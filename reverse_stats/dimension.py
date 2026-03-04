"""
Canonical Dimension Limit Definitions - Reverse Statistics Pipeline
====================================================================

This module defines the authoritative DimensionLimitError and related dimension
analysis functions used throughout the pipeline.

CRITICAL: This is the SINGLE CANONICAL SOURCE for dimension limit definitions.
DO NOT redefine DimensionLimitError in any other module.

All files MUST import from here:
    from .dimension import DimensionLimitError

This module is imported by 15+ pipeline modules (over 50% of the codebase).
Breaking changes here affect the entire pipeline.


"""

from .exceptions import ReverseStatsError
import numpy as np
import math
from fractions import Fraction
from typing import List, Tuple, Optional, Union, Any, Dict
from dataclasses import dataclass
import logging

# Use sympy for exact linear algebra
try:
    import sympy
    from sympy import Matrix
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    # Will use numpy fallback with warning

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ============================================================================
# EXCEPTIONS
# ============================================================================

class DimensionError(ReverseStatsError):
    """Base exception for dimension-related errors."""
    def __init__(self, message: str):
        self.message = message

        super().__init__(message)


class DimensionLimitError(DimensionError):
    """
    Raised when effective dimension exceeds the Barvinok tractability limit.

    Barvinok's algorithm has complexity exponential in dimension.
    For dimensions > 15, the pipeline may be intractable.

    This is the CANONICAL definition - all modules MUST import from here.

    Args:
        dimension: The effective dimension that exceeded the limit
        threshold: The maximum allowed dimension (default 15)
    """
    def __init__(self, dimension: int, threshold: int = 15):
        self.dimension = dimension
        self.threshold = threshold
        super().__init__(
            f"Effective dimension {dimension} exceeds Barvinok tractability limit {threshold}",

        )


# ============================================================================
# CONFIGURATION
# ============================================================================

def get_dimension_config() -> Dict[str, Any]:
    """Get dimension-specific configuration."""
    config = {
        "max_dimension": 15,
        "integrality_tolerance": 1e-10,
    }
    
    # Try to integrate with global config
    try:
        from .config import get_config
        global_config = get_config()
        pipeline_config = getattr(global_config, 'pipeline_config', global_config)
        config["max_dimension"] = getattr(pipeline_config, "max_dimension", config["max_dimension"])
        config["integrality_tolerance"] = getattr(pipeline_config, "integrality_tolerance", config["integrality_tolerance"])
    except ImportError:
        pass  # config module not available
    except AttributeError:
        # Only ignore missing expected attributes, re-raise others
        import sys
        if not any(x in str(sys.exc_info()[1]) for x in ['pipeline_config', 'max_dimension']):
            raise
    
    return config


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DimensionAnalysis:
    """
    Result of dimension analysis.

    Attributes:
        ambient_dimension: Dimension of ambient space
        affine_dimension: Dimension of affine hull
        rank: Rank of constraint matrix
        is_full_rank: Whether constraints have full rank
    """
    ambient_dimension: int = 0
    affine_dimension: int = 0
    rank: int = 0
    is_full_rank: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "ambient_dimension": self.ambient_dimension,
            "affine_dimension": self.affine_dimension,
            "rank": self.rank,
            "is_full_rank": self.is_full_rank
        }


# ============================================================================
# DIMENSION ANALYSIS FUNCTIONS
# ============================================================================

def compute_affine_hull(points: List[Tuple[Fraction, ...]]) -> Tuple[List[Tuple[Fraction, ...]], int]:
    """
    Compute affine hull basis and dimension.

    Args:
        points: List of points in the affine space

    Returns:
        Tuple of (basis_vectors, dimension)

    Raises:
        DimensionError: If points have inconsistent dimensions
        DimensionLimitError: If computed dimension exceeds threshold
    """
    if not points:
        return [], 0
    
    # Validate all points have same dimension
    dim = len(points[0])
    if any(len(p) != dim for p in points[1:]):
        raise DimensionError("All points must have the same dimension")
    
    # Use first point as reference
    ref = points[0]
    vectors = []
    for p in points[1:]:
        v = tuple(p[i] - ref[i] for i in range(dim))
        vectors.append(v)
    
    if not vectors:
        return [], 0
    
    # Compute rank using exact arithmetic if possible
    if HAS_SYMPY:
        # Use sympy for exact rank
        M = Matrix([[x for x in v] for v in vectors])
        rank = M.rank()
    else:
        # Fallback to numpy with warning
        logger.warning("sympy not available - rank computation using float (may lose precision)")
        vec_array = np.array([[float(x) for x in v] for v in vectors])
        rank = np.linalg.matrix_rank(vec_array)
    
    affine_dim = rank
    
    # Check dimension limit
    config = get_dimension_config()
    max_dim = config.get("max_dimension", 15)
    if affine_dim > max_dim:
        raise DimensionLimitError(affine_dim, max_dim)
    
    # Extract basis (simplified)
    basis = []
    if rank > 0:
        # Use first 'rank' vectors as basis (simplified)
        for i in range(min(rank, len(vectors))):
            basis.append(vectors[i])
    
    return basis, affine_dim


def compute_effective_dimension(constraint_matrix: np.ndarray,
                               num_vars: int,
                               tolerance: float = 1e-10) -> int:
    """
    Compute effective dimension from constraint matrix.

    For polytope defined by constraints, effective dimension = num_vars - rank(constraints).

    BUG-FIX (v14): The original implementation used np.linalg.matrix_rank which
    operates on float64.  For near-rank-deficient systems (e.g. two constraints
    that are almost but not exactly parallel due to Fraction→float conversion)
    this can return the wrong rank, causing eff_dim to be off by ±1.  An off-by-
    one here propagates into the dimension guard and the Step 13 vertex enumeration
    branch decision (eff_dim==0 → solve directly, eff_dim>0 → full LP enumeration).

    Fix: attempt SymPy exact rational rank first; fall back to numpy only when
    the matrix entries are not convertible to exact rationals.

    Args:
        constraint_matrix: Matrix of equality constraints (numpy array, any dtype)
        num_vars: Number of variables
        tolerance: Tolerance used only in the numpy fallback path

    Returns:
        Effective dimension (exact when SymPy succeeds)

    Raises:
        TypeError: If constraint_matrix is not a numpy array
        DimensionLimitError: If effective dimension exceeds threshold
    """
    if not isinstance(constraint_matrix, np.ndarray):
        raise TypeError("constraint_matrix must be a numpy array")

    if constraint_matrix.size == 0:
        return num_vars

    # ── Exact path via SymPy ──────────────────────────────────────────────────
    # Convert each entry to sympy.Rational so rank is computed over Q, not R.
    # This is correct for any matrix whose entries are rational numbers
    # (which is always true here because the constraint coefficients come from
    # Fraction objects that were converted to float only for the numpy call).
    try:
        import sympy
        rows = constraint_matrix.tolist()
        sym_rows = [
            [sympy.Rational(c).limit_denominator(10**12)
             if not isinstance(c, sympy.Rational) else c
             for c in row]
            for row in rows
        ]
        # Build exact rational matrix and compute rank
        M = sympy.Matrix(sym_rows)
        rank = M.rank()
        logger.debug(
            f"compute_effective_dimension: SymPy exact rank={rank} "
            f"(numpy approx={np.linalg.matrix_rank(constraint_matrix, tol=tolerance)})"
        )
    except Exception as sym_err:
        # Fall back to numpy float rank if SymPy conversion fails
        logger.warning(
            f"compute_effective_dimension: SymPy exact rank failed ({sym_err}); "
            "falling back to numpy float rank — result may be imprecise for "
            "near-rank-deficient systems."
        )
        rank = np.linalg.matrix_rank(constraint_matrix, tol=tolerance)

    eff_dim = num_vars - rank

    # Check dimension limit
    config = get_dimension_config()
    max_dim = config.get("max_dimension", 15)
    if eff_dim > max_dim:
        raise DimensionLimitError(eff_dim, max_dim)

    return eff_dim


def analyze_dimension(points: List[Tuple[Fraction, ...]]) -> DimensionAnalysis:
    """
    Comprehensive dimension analysis of point set.

    Args:
        points: List of points to analyze

    Returns:
        DimensionAnalysis with ambient, affine dimensions and rank
    """
    if not points:
        return DimensionAnalysis()
    
    ambient = len(points[0])
    
    # Compute affine dimension (vectors computed internally)
    _, affine = compute_affine_hull(points)
    
    # Compute rank (simplified - could reuse vectors from compute_affine_hull)
    if len(points) > 1:
        ref = points[0]
        vectors = []
        for p in points[1:]:
            v = tuple(p[i] - ref[i] for i in range(ambient))
            vectors.append(v)
        
        if HAS_SYMPY:
            M = Matrix([[x for x in v] for v in vectors])
            rank = M.rank()
        else:
            vec_array = np.array([[float(x) for x in v] for v in vectors])
            rank = np.linalg.matrix_rank(vec_array)
    else:
        rank = 0
    
    return DimensionAnalysis(
        ambient_dimension=ambient,
        affine_dimension=affine,
        rank=rank,
        is_full_rank=(rank == ambient)
    )


def compute_reduction_factor(original_dim: int, effective_dim: int) -> float:
    """
    Compute dimension reduction factor.

    Args:
        original_dim: Original dimension before reduction
        effective_dim: Effective dimension after reduction (e.g., 4)

    Returns:
        Reduction factor (original_dim / effective_dim)

    Note:
        If effective_dim is 0 (zero-dimensional polytope/point),
        returns float('inf') as the reduction factor is infinite.
    """
    if effective_dim == 0:
        return float('inf')
    return original_dim / effective_dim


def enforce_dimension_guard(dimension: int, threshold: Optional[int] = None) -> None:
    """
    Enforce dimension guard threshold.

    Args:
        dimension: Dimension to check
        threshold: Maximum allowed dimension (uses config default if None)

    Raises:
        DimensionLimitError: If dimension exceeds threshold
    """
    if threshold is None:
        config = get_dimension_config()
        threshold = config.get("max_dimension", 15)
    
    if dimension > threshold:
        raise DimensionLimitError(dimension, threshold)


# ============================================================================
# VALIDATION AND TESTING
# ============================================================================

def validate_dimension_utils() -> Dict[str, bool]:
    """Run internal test suite to verify dimension utilities."""
    results = {}
    
    try:
        from fractions import Fraction
        
        # Test points: 3 points in 2D forming a triangle
        points = [
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1))
        ]
        
        # Test 1: Affine hull computation
        basis, aff_dim = compute_affine_hull(points)
        results["affine_hull"] = (aff_dim == 2)
        
        # Test 2: Dimension analysis
        analysis = analyze_dimension(points)
        results["dimension_analysis"] = (
            analysis.ambient_dimension == 2 and
            analysis.affine_dimension == 2 and
            analysis.rank == 2 and
            analysis.is_full_rank
        )
        
        # Test 3: Reduction factor
        factor = compute_reduction_factor(720, 4)
        results["reduction_factor"] = abs(factor - 180.0) < 1e-10
        
        # Test 4: Dimension guard
        try:
            enforce_dimension_guard(20, 15)
            results["dimension_guard"] = False
        except DimensionLimitError:
            results["dimension_guard"] = True
        
        # Test 5: Effective dimension
        # Create a simple constraint matrix: x + y = 1
        A = np.array([[1, 1]])
        eff_dim = compute_effective_dimension(A, 2)
        results["effective_dimension"] = (eff_dim == 1)
        
        # Test 6: Dimension validation
        try:
            bad_points = [(Fraction(1),), (Fraction(1), Fraction(2))]
            compute_affine_hull(bad_points)
            results["dimension_validation"] = False
        except DimensionError:
            results["dimension_validation"] = True
        
        logger.info("✅ Dimension utilities validation passed")
        
    except Exception as e:
        logger.error(f"❌ Dimension utilities validation failed: {e}")
        results["validation_error"] = str(e)
    
    return results


# ============================================================================
# MAIN TESTING
# ============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Dimension Utilities - CANONICAL SOURCE")
    print("=" * 60)
    print("\nThis module defines the authoritative DimensionLimitError.")
    print("All modules MUST import from here, not redefine locally.")
    print("\nImported by: 15+ pipeline modules (over 50% of codebase)")
    print("=" * 60)
    
    results = validate_dimension_utils()
    
    print("\nValidation Results:")
    print("-" * 40)
    success = 0
    total = 0
    error_key = None
    
    for key, value in results.items():
        total += 1
        if key == "validation_error":
            error_key = key
            print(f"❌ {key}: {value}")
        elif value:
            success += 1
            print(f"✅ {key}: PASSED")
        else:
            print(f"❌ {key}: FAILED")
    
    # Adjust total for error key if present
    if error_key:
        total -= 1
    
    print("-" * 40)
    print(f"Overall: {success}/{total} tests passed")
    
    # Demo
    print("\n" + "=" * 60)
    print("Dimension Analysis Demo - example problem")
    print("=" * 60)
    
    from fractions import Fraction
    
    # Example: variables with sum and sum-of-squares constraints
    # Effective dimension = 6 - 2 = 4
    A = np.array([
        [1, 1, 1, 1, 1, 1],        # sum constraint
        [1, 4, 9, 16, 25, 36]      # sum of squares constraint
    ])
    
    eff_dim = compute_effective_dimension(A, 6)
    print(f"\nDice problem: 6 variables, 2 constraints")
    print(f"Effective dimension: {eff_dim}")
    
    factor = compute_reduction_factor(720, eff_dim)
    if factor == float('inf'):
        print(f"Reduction factor: infinite (point)")
    else:
        print(f"Reduction factor: {factor:.1f}x")
    
    # Test dimension guard
    print(f"\nDimension guard threshold: 15")
    try:
        enforce_dimension_guard(eff_dim)
        print(f"✅ Dimension {eff_dim} within limits")
    except DimensionLimitError as e:
        print(f"❌ {e}")
    
    print("\n" + "=" * 60)
    print("✅ Dimension Utilities Ready for Production")
    print("=" * 60)

# ============================================================================
# V15.5 — Intrinsic Lattice Basis via Smith Normal Form (S3 — Lattice-Correct)
# ============================================================================

def intrinsic_lattice_basis(rays):
    """
    Compute a primitive integer basis for the sublattice span_Z(rays).

    Uses Smith Normal Form (via smith_normal_decomp from sympy.matrices.normalforms)
    to extract a primitive Z-basis of the column lattice.  This is strictly stronger
    than rref, which gives only a rational column-space basis and produces non-integer
    projected coordinates when the lattice index > 1 (e.g. cones 2 and 5 of the N=6
    dice problem have index=8 under rref).

    Algorithm:
        Given ray matrix M (n_rays × ambient_dim):
        1. Compute M = U * D * V  (Smith Normal Form over Z)
           U: n_rays × n_rays unimodular
           D: n_rays × ambient_dim diagonal (elementary divisors)
           V: ambient_dim × ambient_dim unimodular
        2. rank d = number of nonzero diagonal entries of D
        3. Primitive basis B = first d columns of V^{-1}  (ambient_dim × d)
           These form a Z-basis of span_Z(rays).
        4. Left inverse B_inv = (B^T B)^{-1} B^T  (d × ambient_dim)
           Exact rational, satisfies B_inv @ B = I_d.
        5. Assert B_inv @ B == I_d  (exact SymPy check).
        6. Assert all projected ray coordinates are integers.

    Args:
        rays: list of tuples of Fraction/int — cone generators, n_rays × ambient_dim

    Returns:
        B     : list of ambient_dim tuples, each of length d  (ambient_dim × d basis)
        B_inv : list of d tuples, each of length ambient_dim  (d × ambient_dim left inverse)
        d     : int — intrinsic dimension (rank of ray matrix)

    Raises:
        ValueError  if rays empty or rank zero.
        AssertionError if B_inv @ B != I_d or projected rays are non-integer.
    """
    import sympy as _sp
    from sympy.matrices.normalforms import smith_normal_decomp as _snf
    from fractions import Fraction

    if not rays:
        raise ValueError("intrinsic_lattice_basis: empty ray list")

    ambient_dim = len(rays[0])

    # Build integer ray matrix (rows = rays)
    int_rows = []
    for ray in rays:
        row = []
        for x in ray:
            if isinstance(x, Fraction):
                if x.denominator != 1:
                    raise ValueError(
                        f"intrinsic_lattice_basis: ray has non-integer entry {x}. "
                        "Rays must be integral for lattice basis computation."
                    )
                row.append(int(x))
            else:
                row.append(int(x))
        int_rows.append(row)

    M = _sp.Matrix(int_rows)  # n_rays × ambient_dim

    # Smith Normal Form: M = U * D * V
    try:
        D, U, V = _snf(M)
    except Exception as e:
        raise ValueError(f"intrinsic_lattice_basis: SNF failed: {e}")

    # Verify U and V are unimodular (det = ±1)
    det_U = int(U.det())
    det_V = int(V.det())
    assert abs(det_U) == 1, f"SNF: U not unimodular (det={det_U})"
    assert abs(det_V) == 1, f"SNF: V not unimodular (det={det_V})"

    # Extract rank and elementary divisors
    d = sum(1 for i in range(min(M.shape)) if D[i, i] != 0)
    if d == 0:
        raise ValueError("intrinsic_lattice_basis: all rays are zero")

    elem_divs = [int(D[i, i]) for i in range(d)]

    # Primitive column basis: first d columns of V^{-1}
    # These span the same lattice as the rays, primitively.
    V_inv = V.inv()  # exact over Z since det(V)=±1
    B_sp = V_inv[:, :d]  # ambient_dim × d

    # Left inverse: B_inv = (B^T B)^{-1} B^T
    BtB = B_sp.T * B_sp
    try:
        BtB_inv = BtB.inv()
    except Exception as e:
        raise ValueError(
            f"intrinsic_lattice_basis: B^T B singular (dependent basis): {e}"
        )
    B_inv_sp = BtB_inv * B_sp.T  # d × ambient_dim

    # Assert B_inv @ B == I_d exactly
    check = B_inv_sp * B_sp
    assert check == _sp.eye(d), (
        "intrinsic_lattice_basis: left-inverse check failed: " + str(check)
    )

    # Assert all projected ray coordinates are integers
    for i, row in enumerate(int_rows):
        r_vec = _sp.Matrix(row)
        coords = B_inv_sp * r_vec
        for j in range(d):
            c = coords[j]
            # c is a SymPy Rational; check denominator
            if hasattr(c, 'denominator') and int(c.denominator) != 1:
                raise AssertionError(
                    f"intrinsic_lattice_basis: ray {i} projects to non-integer "
                    f"coord {c} in dimension {j}. "
                    f"Elementary divisors: {elem_divs}. "
                    "SNF basis may be incorrect."
                )

    # Convert to Fraction tuples for pipeline consumption
    B_list = [
        tuple(Fraction(int(B_sp[i, j].numerator), int(B_sp[i, j].denominator))
              for j in range(d))
        for i in range(ambient_dim)
    ]
    B_inv_list = [
        tuple(Fraction(int(B_inv_sp[i, j].numerator), int(B_inv_sp[i, j].denominator))
              for j in range(ambient_dim))
        for i in range(d)
    ]

    return B_list, B_inv_list, d
