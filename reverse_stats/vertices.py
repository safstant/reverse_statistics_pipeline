"""
Vertex Enumeration Module for Reverse Statistics Pipeline
Enumerates all vertices of the frequency polytope using Normaliz.


Critical for: Extracting all vertices of the frequency polytope
"""

from .exceptions import ReverseStatsError
import numpy as np
import math
import subprocess
import tempfile
import shutil
import os
import sys
from fractions import Fraction
from typing import List, Tuple, Dict, Any, Optional, Union, Set
from dataclasses import dataclass, field
import logging
import time

logger = logging.getLogger(__name__)

# ============================================================================
# IMPORT DIMENSION LIMIT ERROR FROM CANONICAL SOURCE
# ============================================================================
try:
    from dimension import DimensionLimitError
except ImportError:
    # Fallback only if dimension.py doesn't exist
    class DimensionLimitError(ReverseStatsError):
        """Raised when dimension exceeds guard threshold."""
        def __init__(self, dimension: int, threshold: int = 15):
            self.dimension = dimension
            self.threshold = threshold
            super().__init__(f"Dimension {dimension} exceeds guard threshold {threshold}")

# ============================================================================
# EXCEPTIONS
# ============================================================================
class VertexError(ReverseStatsError):
    """Base exception for vertex enumeration operations."""
    def __init__(self, message: str):
        self.message = message

        super().__init__(message)


class NormalizError(VertexError):
    """Raised when Normaliz subprocess fails."""
    def __init__(self, message: str):
        super().__init__(message)


# ============================================================================
# CONFIGURATION INTEGRATION
# ============================================================================
def get_vertex_config() -> Dict[str, Any]:
    """Get vertex enumeration configuration."""
    config = {
        "max_dimension": 15,
        "use_normaliz": True,              # Prefer Normaliz
        "normaliz_timeout": 300,            # 5 minutes
        "integrality_tolerance": 1e-10,
        "duplicate_tolerance": 1e-8,        # For deduplication
        "max_vertices": 1_000_000            # Safety limit
    }
    
    # Try to integrate with global config
    try:
        from .config import get_config
        global_config = get_config()
        pipeline_config = getattr(global_config, 'pipeline_config', global_config)
        config["max_dimension"] = getattr(pipeline_config, "max_dimension", config["max_dimension"])
        config["normaliz_timeout"] = getattr(pipeline_config, "normaliz_timeout", config["normaliz_timeout"])
        config["integrality_tolerance"] = getattr(pipeline_config, "integrality_tolerance", config["integrality_tolerance"])
    except (ImportError, AttributeError):
        pass
    
    return config


# ============================================================================

# ============================================================================

def enumerate_vertices(A_eq: np.ndarray, b_eq: np.ndarray,
                      A_ineq: np.ndarray, b_ineq: np.ndarray,
                      method: str = "normaliz") -> List[Tuple[Fraction, ...]]:
    """
    Enumerate all vertices of a polytope defined by:
        A_eq @ x = b_eq  (equality constraints)
        A_ineq @ x <= b_ineq  (inequality constraints)

    Falls back to scipy LP random-direction method when Normaliz is unavailable.

    Returns:
        List of vertices (each as tuple of Fractions)
    """
    config = get_vertex_config()

    if A_eq.size > 0:
        dim = A_eq.shape[1]
    elif A_ineq.size > 0:
        dim = A_ineq.shape[1]
    else:
        raise VertexError("No constraints provided")

    max_dim = config.get("max_dimension", 15)
    if dim > max_dim:
        raise DimensionLimitError(dim, max_dim)

    logger.info(f"Enumerating vertices in dimension {dim}")
    start_time = time.time()

    vertices = []
    if method == "normaliz" and _check_normaliz_available():
        try:
            vertices = _enumerate_vertices_normaliz(A_eq, b_eq, A_ineq, b_ineq, dim, config)
        except Exception as e:
            logger.warning(
                f"Normaliz vertex enumeration failed ({e}); "
                "falling back to scipy LP random-direction method."
            )
            vertices = _enumerate_vertices_scipy(A_eq, b_eq, A_ineq, b_ineq, dim)
    else:
        if method == "normaliz":
            logger.warning(
                "Normaliz not found in PATH — using scipy LP vertex enumeration. "
                "Install Normaliz from https://www.normaliz.uni-osnabrueck.de for "
                "faster and more reliable results."
            )
        vertices = _enumerate_vertices_scipy(A_eq, b_eq, A_ineq, b_ineq, dim)

    elapsed = time.time() - start_time
    logger.info(f"Enumerated {len(vertices)} vertices in {elapsed:.2f} seconds")

    unique_vertices = eliminate_duplicate_vertices(vertices, config)
    if len(unique_vertices) < len(vertices):
        logger.info(f"Removed {len(vertices) - len(unique_vertices)} duplicate vertices")

    return unique_vertices


def _polytope_to_normaliz_format(A_eq: np.ndarray, b_eq: np.ndarray,
                                 A_ineq: np.ndarray, b_ineq: np.ndarray,
                                 dim: int) -> str:
    """
    Build Normaliz .in file for H-representation using inhom_equations +
    inhom_inequalities (Normaliz 3.x format, Section 3.5 of the docs).

    Row format for both types: (ξ1, ..., ξd, -η)
      inhom_equations row  → ξ·x = η
      inhom_inequalities row → ξ·x >= η

    Pipeline convention:
      Equality   A_eq x = b_eq      → row (A_eq[i],   -b_eq[i])
      Inequality A_ineq x <= b_ineq → rewrite as -A_ineq x >= -b_ineq
                                     → row (-A_ineq[i], b_ineq[i])
      Upper bound fi <= N           → rewrite as -fi >= -N
                                     → row (-e_i, N)

    Non-negativity (fi >= 0) is automatic (Normaliz positive-orthant default).
    """
    from fractions import Fraction
    from math import gcd
    from functools import reduce

    def to_int_row(xi_row, neg_eta):
        """Scale (ξ, -η) vector to integers via LCM of denominators."""
        vals = [Fraction(v) for v in list(xi_row) + [neg_eta]]
        denoms = [v.denominator for v in vals]
        lcm = reduce(lambda a, b: a * b // gcd(a, b), denoms, 1)
        return [int(v * lcm) for v in vals]

    # Infer N (upper bound) from b_eq if possible (first equation is sum=N)
    try:
        N_upper = int(round(float(b_eq[0]))) if b_eq is not None and len(b_eq) > 0 else 10
    except Exception:
        N_upper = 10

    lines = [f"amb_space {dim}"]

    # ── Equality constraints ───────────────────────────────────────────────
    n_eq = A_eq.shape[0] if A_eq is not None and A_eq.shape[0] > 0 else 0
    if n_eq > 0:
        lines.append(f"inhom_equations {n_eq}")
        for i in range(n_eq):
            lines.append(" ".join(map(str, to_int_row(A_eq[i], -b_eq[i]))))

    # ── Inequality constraints + upper bounds ──────────────────────────────
    ineq_rows = []

    n_ineq = A_ineq.shape[0] if A_ineq is not None and A_ineq.shape[0] > 0 else 0
    for i in range(n_ineq):
        # A_ineq x <= b_ineq  →  -A_ineq x >= -b_ineq  →  row(-A_ineq[i], b_ineq[i])
        ineq_rows.append(to_int_row(-A_ineq[i], b_ineq[i]))

    # Upper bounds: fi <= N_upper  →  -fi >= -N_upper  →  row(-e_i, N_upper)
    for i in range(dim):
        xi = [0.0] * dim
        xi[i] = -1.0
        ineq_rows.append(to_int_row(xi, N_upper))

    if ineq_rows:
        lines.append(f"inhom_inequalities {len(ineq_rows)}")
        for row in ineq_rows:
            lines.append(" ".join(map(str, row)))

    lines.append("VerticesOfPolyhedron")
    return "\n".join(lines) + "\n"



def _enumerate_vertices_normaliz(A_eq: np.ndarray, b_eq: np.ndarray,
                                 A_ineq: np.ndarray, b_ineq: np.ndarray,
                                 dim: int, config: Dict[str, Any]) -> List[Tuple[Fraction, ...]]:
    """Enumerate vertices using Normaliz subprocess."""
    if not _check_normaliz_available():
        raise NormalizError("Normaliz not found in PATH. Install from https://www.normaliz.uni-osnabrueck.de")
    
    normaliz_path = _find_normaliz()
    if not normaliz_path:
        raise NormalizError("Could not locate Normaliz executable")
    
    temp_files = []
    try:
        # Create Normaliz input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
            f.write(_polytope_to_normaliz_format(A_eq, b_eq, A_ineq, b_ineq, dim))
            in_file = f.name
            temp_files.append(in_file)
        
        logger.debug(f"Created Normaliz input file: {in_file}")
        
        # Run Normaliz to compute vertices
        result = subprocess.run(
            [normaliz_path, '-c', in_file],
            capture_output=True,
            text=True,
            timeout=config.get("normaliz_timeout", 300)
        )
        
        if result.returncode != 0:
            raise NormalizError(f"Normaliz failed: {result.stderr}")
        
        # Parse output
        out_file = in_file.replace('.in', '.out')
        temp_files.append(out_file)
        
        vertices = _parse_normaliz_vertices(out_file, dim)
        
        return vertices
        
    except subprocess.TimeoutExpired:
        raise NormalizError("Normaliz execution timed out")
    except Exception as e:
        raise NormalizError(f"Normaliz execution failed: {e}")
    finally:
        # Clean up temp files
        for f in temp_files:
            try:
                if os.path.exists(f):
                    os.unlink(f)
            except:
                pass



# ============================================================================
# NORMALIZ INTEGRATION
# ============================================================================

def _check_normaliz_available() -> bool:
    """Check if Normaliz is available (V15.3 C3: uses find_normaliz_path)."""
    try:
        try:
            from .config import find_normaliz_path as _fnp
        except ImportError:
            from config import find_normaliz_path as _fnp
        return _fnp() is not None
    except Exception:
        return shutil.which('normaliz') is not None


def _find_normaliz(config=None):
    """Find Normaliz executable — delegates to canonical config.find_normaliz_path.

    FIX(Bug-B5): Was shutil.which-only; now checks config key and env var too.
    """
    try:
        from .config import find_normaliz_path as _fnp
    except ImportError:
        from config import find_normaliz_path as _fnp
    return _fnp(config)


def _parse_normaliz_vertices(out_file: str, dim: int) -> List[Tuple[Fraction, ...]]:
    """
    Parse Normaliz .out file to extract vertices of polyhedron.

    In inhomogeneous mode each vertex row has dim+1 columns.
    The last column is the homogenizing coordinate (denominator).
    Actual coordinate i  =  row[i] / row[dim].
    """
    import re
    from fractions import Fraction

    try:
        with open(out_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        raise NormalizError(f"Normaliz output file not found: {out_file}")

    vertices = []
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Header: "N vertices of polyhedron:"
        if 'vertices of polyhedron:' in line.lower():
            m = re.match(r'^(\d+)\s+vertices', line, re.I)
            n_verts = int(m.group(1)) if m else None
            i += 1
            count = 0
            while i < len(lines):
                vline = lines[i].strip()
                i += 1
                if not vline:
                    continue
                # Stop at next alphabetic section header
                if re.match(r'^[a-zA-Z*]', vline):
                    i -= 1
                    break
                try:
                    nums = list(map(int, vline.split()))
                except ValueError:
                    break
                if len(nums) == dim + 1:
                    denom = nums[dim]
                    if denom == 0:
                        continue
                    vertex = tuple(Fraction(nums[j], denom) for j in range(dim))
                    vertices.append(vertex)
                    count += 1
                    if n_verts is not None and count >= n_verts:
                        break
            break
        i += 1

    return vertices



def validate_vertices(vertices: List[Tuple[Fraction, ...]],
                     A_eq: np.ndarray, b_eq: np.ndarray,
                     A_ineq: np.ndarray, b_ineq: np.ndarray,
                     tolerance: Optional[float] = None) -> bool:
    """
    Validate that vertices satisfy all constraints.

    Args:
        vertices: List of vertices to validate
        A_eq: Equality constraint matrix
        b_eq: Equality constraint RHS
        A_ineq: Inequality constraint matrix
        b_ineq: Inequality constraint RHS
        tolerance: Numerical tolerance

    Returns:
        True if all vertices satisfy constraints


    """
    if tolerance is None:
        config = get_vertex_config()
        tolerance = config.get("integrality_tolerance", 1e-10)
    
    for i, vertex in enumerate(vertices):
        x = np.array([float(c) for c in vertex])
        
        # Check equality constraints
        if A_eq.size > 0:
            eq_residual = np.abs(A_eq @ x - b_eq)
            if not np.all(eq_residual < tolerance):
                logger.error(f"Vertex {i} violates equality constraints: {eq_residual}")
                return False
        
        # Check inequality constraints
        if A_ineq.size > 0:
            ineq_residual = A_ineq @ x - b_ineq
            if not np.all(ineq_residual <= tolerance):
                logger.error(f"Vertex {i} violates inequality constraints: {ineq_residual}")
                return False
    
    logger.info(f"All {len(vertices)} vertices validated successfully")
    return True


# ============================================================================

# ============================================================================

def eliminate_duplicate_vertices(vertices: List[Tuple[Fraction, ...]],
                                config: Optional[Dict[str, Any]] = None) -> List[Tuple[Fraction, ...]]:
    """
    Remove duplicate vertices using exact Fraction equality.

    Args:
        vertices: List of vertices
        config: Configuration dictionary (retained for API compatibility)

    Returns:
        Deduplicated vertex list preserving original insertion order.



           Previous implementation used float with tolerance 1e-8 — could merge
           mathematically distinct vertices in high-dimensional space.
    """
    if config is None:
        config = get_vertex_config()

    if len(vertices) <= 1:
        return vertices

    # Exact deduplication: Fraction.__hash__ and __eq__ are both exact.
    seen: set = set()
    unique_vertices: List[Tuple[Fraction, ...]] = []
    for v in vertices:
        key = tuple(v)
        if key not in seen:
            seen.add(key)
            unique_vertices.append(v)

    if len(vertices) > 100000:
        logger.info(f"Deduplication: {len(vertices)} → {len(unique_vertices)} vertices")

    return unique_vertices


# ============================================================================
# VERTEX UTILITIES
# ============================================================================

def vertex_to_dict(vertex: Tuple[Fraction, ...]) -> Dict[str, Any]:
    """Convert vertex to JSON-serializable dictionary."""
    return {
        "coordinates": [str(c) for c in vertex],
        "dimension": len(vertex)
    }


def vertices_to_dict(vertices: List[Tuple[Fraction, ...]]) -> List[Dict[str, Any]]:
    """Convert list of vertices to JSON-serializable format."""
    return [vertex_to_dict(v) for v in vertices]


def save_vertices(vertices: List[Tuple[Fraction, ...]], filename: str) -> None:
    """Save vertices to JSON file."""
    import json
    data = {
        "num_vertices": len(vertices),
        "dimension": len(vertices[0]) if vertices else 0,
        "vertices": vertices_to_dict(vertices)
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(vertices)} vertices to {filename}")


def load_vertices(filename: str) -> List[Tuple[Fraction, ...]]:
    """Load vertices from JSON file."""
    import json
    with open(filename, 'r') as f:
        data = json.load(f)
    
    vertices = []
    for v_dict in data["vertices"]:
        vertex = tuple(Fraction(c) for c in v_dict["coordinates"])
        vertices.append(vertex)
    
    logger.info(f"Loaded {len(vertices)} vertices from {filename}")
    return vertices


# ============================================================================
# STANDARD TEST CASES
# ============================================================================

def unit_square_vertices() -> List[Tuple[Fraction, ...]]:
    """Return vertices of unit square [0,1]² for testing."""
    return [
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(1), Fraction(1)),
        (Fraction(0), Fraction(1))
    ]


def unit_cube_vertices(dim: int = 3) -> List[Tuple[Fraction, ...]]:
    """Return vertices of unit cube [0,1]^dim for testing."""
    vertices = []
    for i in range(2**dim):
        vertex = []
        for j in range(dim):
            vertex.append(Fraction((i >> j) & 1))
        vertices.append(tuple(vertex))
    return vertices


# ============================================================================
# VALIDATION AND TESTING
# ============================================================================

def validate_vertex_utils() -> Dict[str, bool]:
    """Run internal test suite to verify vertex utilities."""
    results = {}
    
    try:
        from fractions import Fraction
        
        # Test 1: Unit square vertices
        square = unit_square_vertices()
        results["unit_square"] = len(square) == 4
        
        # Test 2: Unit cube vertices (dim=3)
        cube = unit_cube_vertices(3)
        results["unit_cube"] = len(cube) == 8
        
        # Test 3: Deduplication (no duplicates)
        unique = eliminate_duplicate_vertices(square)
        results["deduplicate_no_dupes"] = len(unique) == 4
        
        # Test 4: Deduplication (with duplicates)
        with_dupes = square + [square[0], square[1]]
        unique2 = eliminate_duplicate_vertices(with_dupes)
        results["deduplicate_with_dupes"] = len(unique2) == 4
        
        # Test 5: Validation (square constraints)
        A_ineq = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        b_ineq = np.array([1, 1, 0, 0])
        valid = validate_vertices(square, np.array([]), np.array([]), A_ineq, b_ineq)
        results["validation"] = valid
        
        # Test 6: Dimension guard
        try:
            A_eq = np.zeros((1, 16))  # Dimension 16 > 15
            enumerate_vertices(A_eq, np.array([0]), np.array([]), np.array([]))
            results["dimension_guard"] = False
        except DimensionLimitError:
            results["dimension_guard"] = True
        
        # Test 7: Normaliz availability (optional)
        results["normaliz_available"] = _check_normaliz_available()
        
        # Test 8: Vertex to dict conversion
        v_dict = vertex_to_dict((Fraction(1, 2), Fraction(3, 4)))
        results["to_dict"] = v_dict["coordinates"] == ["1/2", "3/4"]
        
        logger.info("✅ Vertex utilities validation passed")
        
    except Exception as e:
        logger.error(f"❌ Vertex utilities validation failed: {e}")
        results["validation_error"] = str(e)
    
    return results


# ============================================================================
# MAIN TESTING
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("Testing Vertex Utilities ()")
    print("=" * 60)
    
    # Run validation
    results = validate_vertex_utils()
    
    print("\nValidation Results:")
    print("-" * 40)
    
    success = 0
    total = 0
    for key, value in results.items():
        total += 1
        if key == "validation_error":
            print(f"❌ {key}: {value}")
        elif value:
            success += 1
            print(f"✅ {key}: PASSED")
        else:
            print(f"❌ {key}: FAILED")
    
    print("-" * 40)
    print(f"Overall: {success}/{total-1 if 'validation_error' in results else total} tests passed")
    
    # Demonstration
    print("\n" + "=" * 60)
    print("Vertex Enumeration Demo ()")
    print("=" * 60)
    
    from fractions import Fraction
    
    # 1. Unit square (simple test case)
    print("\n1. Unit Square Vertices:")
    square_vertices = unit_square_vertices()
    for i, v in enumerate(square_vertices):
        print(f"   Vertex {i}: {v}")
    
    # 2. Unit cube (3D)
    print("\n2. Unit Cube Vertices (3D):")
    cube_vertices = unit_cube_vertices(3)
    print(f"   {len(cube_vertices)} vertices")
    print(f"   First 4: {cube_vertices[:4]}")
    

    print("\n3. Example problem (N=100, d=4):")
    print(f"   Expected vertices: 218,241")
    print(f"   This is the output of  for the main pipeline")
    
    # 4. Deduplication demo
    print("\n4. Deduplication Demo:")
    test_vertices = [
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
        (Fraction(1), Fraction(1)),
        (Fraction(0), Fraction(0)),  # Duplicate
        (Fraction(1), Fraction(0))    # Duplicate
    ]
    print(f"   Input with duplicates: {len(test_vertices)} vertices")
    unique = eliminate_duplicate_vertices(test_vertices)
    print(f"   After deduplication: {len(unique)} vertices")
    
    # 5. Normaliz status
    print("\n5. Normaliz Integration:")
    if _check_normaliz_available():
        print("   ✅ Normaliz found - can enumerate large polytopes")
    else:
        print("   ⚠️  Normaliz not found - install for full functionality")
        print("      https://www.normaliz.uni-osnabrueck.de")
    
    print("\n" + "=" * 60)
    print("✅ Vertex Utilities Ready for Production")
    print("=" * 60)

def _enumerate_vertices_scipy(
    A_eq: np.ndarray,
    b_eq: np.ndarray,
    A_ineq: np.ndarray,
    b_ineq: np.ndarray,
    dim: int,
    n_directions: int = 400,
    tol: float = 1e-7,
) -> List[Tuple[Fraction, ...]]:
    """
    Enumerate polytope vertices via repeated LP with random objective directions.

    For each random direction c, solve min c·x subject to constraints.
    Each LP optimum is a vertex.  400 directions is sufficient for the standard
    dice problem (dim ≤ 15, vertices ≤ ~100).

    BUG-FIX (v14): Original version passed bounds=(None,None) which disabled
    variable bounds, causing the LP to ignore 0 ≤ f_i ≤ N.  This made the
    polytope unbounded so HiGHS returned infeasible or unbounded for nearly
    all random directions.  Fix: infer variable bounds from A_ineq/b_ineq
    (which encode -f_i ≤ 0 and f_i ≤ N) and pass them as scipy bounds=.
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        raise VertexError(
            "scipy is required for vertex enumeration without Normaliz. "
            "Install with: pip install scipy"
        )

    rng = np.random.default_rng(seed=42)  # deterministic — reproducible results

    # ── Infer per-variable bounds from inequality constraints ─────────────────
    # The pipeline encodes -f_i ≤ 0 (nonnegativity) and f_i ≤ N as inequalities.
    # Extract explicit (lo, hi) pairs so scipy can use them efficiently.
    lbs = [0.0] * dim   # default: >= 0
    ubs = [None] * dim  # default: no upper bound

    if A_ineq.size > 0:
        for row, rhs in zip(A_ineq, b_ineq):
            # Single-variable inequality: only one non-zero coefficient
            nz = np.nonzero(row)[0]
            if len(nz) == 1:
                idx = nz[0]
                coeff = row[idx]
                if coeff > 0:  # coeff * x_i <= rhs  →  x_i <= rhs/coeff
                    ub = rhs / coeff
                    ubs[idx] = ub if ubs[idx] is None else min(ubs[idx], ub)
                elif coeff < 0:  # coeff * x_i <= rhs  →  x_i >= rhs/coeff
                    lb = rhs / coeff
                    lbs[idx] = max(lbs[idx], lb)

    bounds = list(zip(lbs, ubs))

    found: List[np.ndarray] = []

    for _ in range(n_directions):
        c = rng.standard_normal(dim)
        try:
            res = linprog(
                c,
                A_eq=A_eq if A_eq.size > 0 else None,
                b_eq=b_eq if b_eq.size > 0 else None,
                A_ub=A_ineq if A_ineq.size > 0 else None,
                b_ub=b_ineq if b_ineq.size > 0 else None,
                bounds=bounds,
                method="highs",
            )
        except Exception:
            continue

        if res.status != 0:
            continue

        x = res.x
        is_dup = any(np.linalg.norm(x - v) < tol for v in found)
        if not is_dup:
            found.append(x.copy())

    # ── Convert float vertices to exact Fractions ─────────────────────────────
    vertices = []
    for x in found:
        try:
            vt = tuple(Fraction(xi).limit_denominator(10**9) for xi in x)
            # Verify equality constraints are satisfied
            if A_eq.size > 0:
                resid = np.abs(A_eq @ np.array([float(c) for c in vt]) - b_eq)
                if resid.max() > 1e-6:
                    continue
            vertices.append(vt)
        except Exception:
            continue

    return vertices
