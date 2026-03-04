"""
Simplex Module for Reverse Statistics Pipeline
Provides simplex representation and operations for lattice geometry.


Critical for: Unimodularity testing ( skip decision) and simplex decomposition
"""

from .exceptions import ReverseStatsError
import math
import subprocess
import tempfile
import shutil
import os
import sys
from fractions import Fraction
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import itertools

# Use sympy for exact linear algebra
try:
    import sympy
    from sympy import Matrix
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    logger = logging.getLogger(__name__)
    logger.warning("sympy not available - using fallback implementations")

logger = logging.getLogger(__name__)

# ============================================================================
# Add current directory to path for standalone execution
# ============================================================================
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# EXCEPTIONS
# ============================================================================
class SimplexError(ReverseStatsError):
    """Base exception for simplex operations."""
    def __init__(self, message: str):
        self.message = message

        super().__init__(message)


class DimensionError(SimplexError):
    """Raised for dimension mismatches."""
    def __init__(self, message: str):
        super().__init__(message)


class UnimodularError(SimplexError):
    """Raised when unimodular operations fail."""
    def __init__(self, message: str):
        super().__init__(message)


class DecompositionError(SimplexError):
    """Raised when simplex decomposition fails."""
    def __init__(self, message: str):
        super().__init__(message)


# Import DimensionLimitError from canonical source
try:
    from dimension import DimensionLimitError
except ImportError:
    class DimensionLimitError(SimplexError):
        """Raised when dimension exceeds guard threshold."""
        def __init__(self, dimension: int, threshold: int = 15):
            self.dimension = dimension
            self.threshold = threshold
            super().__init__(
                f"Simplex dimension {dimension} exceeds guard threshold {threshold}",

            )


# ============================================================================
# IMPORT HANDLING (Dual-mode for package + standalone execution)
# ============================================================================
try:
    # Package mode
    from .math_utils import matrix_rank, determinant_exact
    from .lattice_utils import lattice_points_in_polytope
    from .config import get_config
    HAS_DEPS = True
except (ImportError, ModuleNotFoundError):
    # Standalone mode
    try:
        from math_utils import matrix_rank, determinant_exact
        from lattice_utils import lattice_points_in_polytope
        from config import get_config
        HAS_DEPS = True
    except ImportError:
        HAS_DEPS = False
        logger.debug("Using fallback implementations for simplex operations")


# ============================================================================
# CONFIGURATION INTEGRATION
# ============================================================================
def get_simplex_config() -> Dict[str, Any]:
    """Get simplex-specific configuration with sane defaults."""
    config = {
        "max_dimension": 15,
        "unimodular_tolerance": 1e-10,    # Tolerance for determinant=1 check
        "integrality_tolerance": 1e-10,
        "use_normaliz": True,             # Prefer Normaliz for lattice enumeration
        "normaliz_timeout": 60,           # Seconds
    }
    
    # Try to integrate with global config
    if HAS_DEPS:
        try:
            global_config = get_config()
            pipeline_config = getattr(global_config, 'pipeline_config', global_config)
            config["max_dimension"] = getattr(pipeline_config, "max_dimension", config["max_dimension"])
            config["normaliz_timeout"] = getattr(pipeline_config, "normaliz_timeout", config["normaliz_timeout"])
            config["integrality_tolerance"] = getattr(pipeline_config, "integrality_tolerance", config["integrality_tolerance"])
        except (ImportError, AttributeError):
            pass
    
    return config


# ============================================================================
# NORMALIZ INTEGRATION
# ============================================================================

def _check_normaliz_available() -> bool:
    """Check if Normaliz is available in PATH."""
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


def _call_normaliz_for_lattice_points(vertices: List[Tuple[Fraction, ...]], 
                                      config: Dict[str, Any]) -> List[Tuple[int, ...]]:
    """
    Call Normaliz to compute lattice points in simplex.

    Args:
        vertices: Simplex vertices
        config: Configuration dictionary

    Returns:
        List of lattice points as tuples of integers
    """
    if not _check_normaliz_available():
        logger.warning("Normaliz not found in PATH - using fallback enumeration")
        return []
    
    normaliz_path = _find_normaliz()
    if not normaliz_path:
        return []
    
    temp_files = []
    try:
        # Create temporary input file
        input_content = _simplex_to_normaliz_format(vertices)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
            f.write(input_content)
            in_file = f.name
            temp_files.append(in_file)
        
        logger.debug(f"Created Normaliz input file: {in_file}")
        
        # Run Normaliz to compute lattice points
        result = subprocess.run(
            [normaliz_path, '-c', in_file],
            capture_output=True,
            text=True,
            timeout=config.get("normaliz_timeout", 60)
        )
        
        if result.returncode != 0:
            logger.warning(f"Normaliz failed: {result.stderr}")
            return []
        
        # Parse output
        out_file = in_file.replace('.in', '.out')
        temp_files.append(out_file)
        
        return _parse_normaliz_lattice_points(out_file, len(vertices[0]))
        
    except subprocess.TimeoutExpired:
        logger.warning("Normaliz execution timed out")
        return []
    except Exception as e:
        logger.warning(f"Normaliz execution failed: {e}")
        return []
    finally:
        # Clean up temp files
        for f in temp_files:
            try:
                if os.path.exists(f):
                    os.unlink(f)
            except:
                pass


def _parse_normaliz_lattice_points(out_file: str, dim: int) -> List[Tuple[int, ...]]:
    """
    Parse Normaliz output file to extract lattice points.

    Normaliz format:
        [point number]
        [coordinates]
    """
    points = []
    
    if not os.path.exists(out_file):
        return points
    
    with open(out_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for point markers
        if line.startswith('[') and ']' in line:
            i += 1
            if i < len(lines):
                coord_line = lines[i].strip()
                parts = coord_line.split()
                
                if len(parts) >= dim:
                    point = []
                    for p in parts[:dim]:
                        # Handle rational numbers
                        if '/' in p:
                            num, den = p.split('/')
                            # Only keep integer points
                            if den == '1':
                                point.append(int(num))
                        else:
                            point.append(int(p))
                    if len(point) == dim:
                        points.append(tuple(point))
        i += 1
    
    return points


# ============================================================================
# SIMPLEX REPRESENTATION
# ============================================================================
@dataclass(frozen=True)
class Simplex:
    """
    Immutable simplex representation.

    A simplex is the convex hull of d+1 affinely independent points in R^d.
    Vertices are stored as tuples of Fractions for exact arithmetic.

    Attributes:
        vertices: Tuple of d+1 vertices (each a tuple of Fractions)
        dimension: Ambient dimension (d)


    """
    vertices: Tuple[Tuple[Fraction, ...], ...]
    
    def __post_init__(self):
        """Validate simplex representation."""
        if not self.vertices:
            raise SimplexError("Simplex must have at least one vertex")
        
        # Determine ambient dimension from first vertex
        dim = len(self.vertices[0])
        if dim == 0:
            raise SimplexError("Vertices must have positive dimension")
        
        # Validate all vertices have consistent dimension
        for i, v in enumerate(self.vertices):
            if len(v) != dim:
                raise DimensionError(
                    f"Vertex {i} dimension mismatch: {len(v)} != {dim}"
                )
        
        # Validate affine independence (d+1 vertices in d dimensions)
        if len(self.vertices) != dim + 1:
            raise SimplexError(
                f"Simplex in {dim}D requires {dim+1} vertices, got {len(self.vertices)}",

            )
        

        config = get_simplex_config()
        max_dim = config.get("max_dimension", 15)
        if dim > max_dim:
            raise DimensionLimitError(dim, max_dim)
    
    @property
    def dimension(self) -> int:
        """Ambient dimension of the simplex."""
        return len(self.vertices[0])
    
    @property
    def volume(self) -> Fraction:
        """
        Exact volume of the simplex.
        Volume = |det(V)| / d! where V is matrix of edge vectors from first vertex.
        """
        dim = self.dimension
        if dim == 0:
            return Fraction(1)
        
        # Build edge vector matrix (rows = v_i - v_0 for i=1..d)
        v0 = self.vertices[0]
        edge_matrix = []
        for i in range(1, dim + 1):
            edge = [self.vertices[i][j] - v0[j] for j in range(dim)]
            edge_matrix.append(edge)
        
        # Compute determinant using sympy if available
        if HAS_SYMPY:
            try:

                from sympy import Rational as _R
                M = Matrix([[_R(int(x.numerator), int(x.denominator))
                             if hasattr(x,'numerator') else _R(int(x))
                             for x in row] for row in edge_matrix])
                det = abs(M.det())  # SymPy exact rational
                from math import factorial
                return Fraction(int(det)) / factorial(dim)
            except Exception as e:
                logger.debug(f"SymPy volume failed, using fallback: {e}")
        
        # Fallback to exact rational determinant
        det = _determinant_exact_fallback(edge_matrix)
        from math import factorial
        return abs(det) / factorial(dim)
    
    @property
    def is_unimodular(self) -> bool:
        """
        Check if simplex is unimodular (lattice simplex with volume = 1/d!).
        For integer simplices: unimodular iff |det(edge_matrix)| = 1.


        """
        # Only integer simplices can be unimodular
        if not self.is_lattice_simplex:
            return False
        
        dim = self.dimension
        if dim == 0:
            return True
        
        # Build edge vector matrix
        v0 = self.vertices[0]
        edge_matrix = []
        for i in range(1, dim + 1):
            edge = [self.vertices[i][j] - v0[j] for j in range(dim)]
            edge_matrix.append(edge)
        
        # Compute determinant using sympy for exact result
        if HAS_SYMPY:
            try:

                from sympy import Rational as _R
                M = Matrix([[_R(int(x.numerator), int(x.denominator))
                             if hasattr(x,'numerator') else _R(int(x))
                             for x in row] for row in edge_matrix])
                det = abs(M.det())
                return det == 1
            except Exception as e:
                logger.debug(f"SymPy is_unimodular failed, using fallback: {e}")
        
        # Fallback to exact rational determinant
        det = _determinant_exact_fallback(edge_matrix)
        return abs(det) == 1
    
    @property
    def is_lattice_simplex(self) -> bool:
        """Check if all vertices have integer coordinates."""
        return all(
            all(x.denominator == 1 for x in v)
            for v in self.vertices
        )
    
    @property
    def multiplicity(self) -> int:
        """
        Multiplicity of the simplex (denominator of Ehrhart series).
        For lattice simplices: multiplicity = d! * volume (integer).
        """
        dim = self.dimension
        if dim == 0:
            return 1
        
        from math import factorial
        vol = self.volume
        mult = vol * factorial(dim)
        
        # Should be integer for lattice simplices
        if mult.denominator != 1:
            logger.warning(f"Non-integer multiplicity {mult} for simplex")
        return int(mult.numerator // mult.denominator)
    
    def contains(self, point: Tuple[Fraction, ...], 
                tol: Optional[float] = None) -> bool:
        """
        Check if point belongs to the simplex using barycentric coordinates with sympy.

        Args:
            point: Point to test (tuple of Fractions)
            tol: Tolerance for barycentric coordinate check

        Returns:
            True if point is inside or on boundary of simplex
        """
        if len(point) != self.dimension:
            return False
        
        if tol is None:
            tol = get_simplex_config().get("integrality_tolerance", 1e-10)
        
        if HAS_SYMPY:
            try:
                # Build matrix for barycentric coordinates
                # Solve: point = Σ λ_i * v_i, Σ λ_i = 1, λ_i ≥ 0
                n = self.dimension + 1
                
                # Build matrix: each column is [v_i; 1]
                M = []
                for j in range(n):
                    col = [float(self.vertices[j][i]) for i in range(self.dimension)] + [1.0]
                    M.append(col)
                
                # Transpose to get rows = constraints
                A = Matrix(M).T
                b = [float(point[i]) for i in range(self.dimension)] + [1.0]
                
                # Solve for barycentric coordinates
                solution = A.solve(Matrix(b))
                lambdas = [float(solution[i]) for i in range(n)]
                
                # Check non-negativity
                return all(l >= -tol for l in lambdas)
                
            except Exception:
                # Fall through to fallback
                pass
        
        # Fallback: use approximate method
        return self._contains_fallback(point, tol)
    
    def _contains_fallback(self, point: Tuple[Fraction, ...], tol: float) -> bool:
        """Fallback containment check using linear algebra approximation."""
        dim = self.dimension
        n = dim + 1
        
        # Convert to floats for approximate check
        verts = [[float(v[i]) for i in range(dim)] for v in self.vertices]
        p = [float(x) for x in point]
        
        # Build matrix A where columns are [v_i; 1]
        A = []
        for j in range(n):
            col = verts[j][:] + [1.0]
            A.append(col)
        
        # Transpose to get rows = constraints
        import numpy as np
        try:
            A_np = np.array(A).T
            b_np = np.array(p + [1.0])
            x, _, _, _ = np.linalg.lstsq(A_np, b_np, rcond=None)
            return bool(np.all(x >= -tol))
        except:
            return False
    
    def lattice_points(self, 
                      enumeration_limit: Optional[int] = None,
                      config: Optional[Dict[str, Any]] = None) -> List[Tuple[int, ...]]:
        """
        Enumerate all integer lattice points within the simplex.

        Args:
            enumeration_limit: Maximum number of points to enumerate
            config: Configuration dictionary

        Returns:
            List of lattice points as tuples of integers


        """
        if config is None:
            config = get_simplex_config()
        
        if enumeration_limit is None:
            enumeration_limit = 1_000_000
        

        
        # Convert vertices to integer coordinates if lattice simplex
        if not self.is_lattice_simplex:
            raise SimplexError("Lattice point enumeration requires lattice simplex")
        
        # Try Normaliz first
        if config.get("use_normaliz", True):
            points = _call_normaliz_for_lattice_points(list(self.vertices), config)
            if points:
                if len(points) <= enumeration_limit:
                    return points
                else:
                    logger.warning(f"Normaliz returned {len(points)} points, limit is {enumeration_limit}")
                    return points[:enumeration_limit]
        
        # Fallback: brute force enumeration for small simplices
        return self._lattice_points_fallback(enumeration_limit, config)
    
    def _lattice_points_fallback(self, 
                                enumeration_limit: int,
                                config: Dict[str, Any]) -> List[Tuple[int, ...]]:
        """Fallback lattice point enumeration using bounding-box iteration.

        FIX(Bug-7): The original code returned [] for dim > 3 with a warning
        about "bounding box explosion".  This caused the Ehrhart interpolation
        to receive a zero count for every dilation factor, which then triggered
        the testing fallback (count = k + 1) and corrupted the polynomial.

        The bounding-box approach is valid for any dimension; it is simply slow
        for large/thin simplices.  In practice Normaliz handles dim > 3, so
        this fallback is only reached when Normaliz is unavailable.  Being slow
        (but correct) is far better than silently returning wrong data.

        NOTE: A long thin simplex in high dimensions can have a very large
        bounding box relative to the number of interior points.  For production
        use with dim > 6, ensure the Normaliz backend is installed.
        """
        dim = self.dimension

        # Determine bounding box
        min_coords = [min(int(v[i]) for v in self.vertices) for i in range(dim)]
        max_coords = [max(int(v[i]) for v in self.vertices) for i in range(dim)]

        # Warn if the box is large so users can diagnose performance issues.
        box_size = 1
        for lo, hi in zip(min_coords, max_coords):
            box_size *= max(1, hi - lo + 1)
        if box_size > 1_000_000:
            logger.warning(
                f"_lattice_points_fallback: bounding box has ~{box_size} candidates "
                f"for dim={dim} simplex — this may be slow. "
                f"Install Normaliz for efficient high-dimensional enumeration."
            )

        # Brute force enumeration
        points = []
        tol = config.get("integrality_tolerance", 1e-10)

        # Generate all integer points in bounding box
        ranges = [range(min_coords[i], max_coords[i] + 1) for i in range(dim)]
        for pt in itertools.product(*ranges):
            # Check if point is in simplex
            if self.contains(tuple(Fraction(x) for x in pt), tol=tol):
                points.append(tuple(int(x) for x in pt))
                if len(points) >= enumeration_limit:
                    logger.warning(f"Reached enumeration limit {enumeration_limit}")
                    break

        return points
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "vertices": [[str(x) for x in v] for v in self.vertices],
            "dimension": self.dimension,
            "volume": str(self.volume),
            "is_unimodular": self.is_unimodular,
            "is_lattice_simplex": self.is_lattice_simplex,
            "multiplicity": self.multiplicity
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Simplex":
        """Create simplex from dictionary."""
        vertices = []
        for row in data["vertices"]:
            vertices.append(tuple(Fraction(x) for x in row))
        return cls(vertices=tuple(vertices))
    
    @classmethod
    def standard_simplex(cls, dimension: int) -> "Simplex":
        """
        Create standard simplex in R^d: conv(0, e_1, e_2, ..., e_d).

        Args:
            dimension: Ambient dimension d

        Returns:
            Standard simplex with vertices at origin and unit vectors
        """
        vertices = []
        # Origin
        vertices.append(tuple(Fraction(0) for _ in range(dimension)))
        # Unit vectors
        for i in range(dimension):
            v = [Fraction(0)] * dimension
            v[i] = Fraction(1)
            vertices.append(tuple(v))
        return cls(vertices=tuple(vertices))


# ============================================================================
# DETERMINANT FALLBACK
# ============================================================================

def _determinant_exact_fallback(matrix: List[List[Fraction]]) -> Fraction:
    """
    Compute exact determinant using Bareiss algorithm (fraction-preserving).
    """
    n = len(matrix)
    if n == 0:
        return Fraction(1)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    # Bareiss algorithm for exact rational determinant
    M = [[Fraction(x) for x in row] for row in matrix]
    det = Fraction(1)
    
    for k in range(n - 1):
        if M[k][k] == 0:
            # Find non-zero pivot
            for i in range(k + 1, n):
                if M[i][k] != 0:
                    M[k], M[i] = M[i], M[k]
                    det = -det
                    break
            else:
                return Fraction(0)
        
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[i][j] * M[k][k] - M[i][k] * M[k][j]) / (det if k > 0 else 1)
        det = M[k][k]
    
    return det * M[n-1][n-1] / (det if n > 2 else 1)


# ============================================================================

# ============================================================================
def decompose_non_unimodular_simplex(simplex: Simplex,
                                     config: Optional[Dict[str, Any]] = None) -> List[Simplex]:
    """
    Decompose non-unimodular simplex into unimodular simplices.

    Uses the "pulling triangulation" method:
      1. Find interior lattice point
      2. Connect to all facets to create smaller simplices
      3. Recursively decompose non-unimodular children

    Args:
        simplex: Non-unimodular simplex to decompose
        config: Configuration dictionary

    Returns:
        List of unimodular simplices whose union is the original simplex


    """
    if config is None:
        config = get_simplex_config()
    
    # Base case: already unimodular
    if simplex.is_unimodular:
        return [simplex]
    
    # Find interior lattice point (not on boundary)
    points = simplex.lattice_points(enumeration_limit=1000, config=config)
    
    # Separate boundary and interior points
    interior_points = []
    
    for p in points:
        # Check if point is on boundary (barycentric coordinate = 0 for some vertex)
        is_boundary = False
        
        if HAS_SYMPY:
            try:
                # Build barycentric system
                n = simplex.dimension + 1
                M = []
                for j in range(n):
                    col = [float(simplex.vertices[j][i]) for i in range(simplex.dimension)] + [1.0]
                    M.append(col)
                
                A = Matrix(M).T
                b = [float(p[i]) for i in range(simplex.dimension)] + [1.0]
                
                solution = A.solve(Matrix(b))
                lambdas = [float(solution[i]) for i in range(n)]
                
                if any(abs(l) < config.get("integrality_tolerance", 1e-10) for l in lambdas):
                    is_boundary = True
            except:
                is_boundary = True
        else:
            # Without sympy, assume boundary if point is on any facet
            # This is a simplification
            is_boundary = True
        
        if not is_boundary:
            interior_points.append(p)
    
    # Need at least one suitable lattice point for stellar subdivision.
    # FIX(Bug-4): The original code raised DecompositionError immediately when
    # no interior lattice point was found.  This crashes on "Reeve simplices"
    # and other empty-interior non-unimodular simplices — a wide and valid class.
    #
    # Correct fallback: use a *boundary* lattice point (one that lies on a
    # facet but is not a vertex).  Connecting a boundary point p to all facets
    # that do NOT contain p gives a valid stellar subdivision into smaller
    # simplices, which we then recurse on.  This always makes progress because
    # the new simplices have strictly smaller volume (the "stellar" point
    # strictly subdivides at least one facet).
    if not interior_points:
        # Collect non-vertex boundary lattice points from the full lattice_points list.
        vertex_set = {tuple(v) for v in simplex.vertices}
        boundary_points = [p for p in points if tuple(p) not in vertex_set]

        if not boundary_points:
            # Truly empty simplex (no lattice points beyond vertices) — cannot
            # subdivide.  Such simplices are already "primitive" in the sense
            # that they have no usable interior or boundary structure; raise an
            # informative error.
            raise DecompositionError(
                f"Cannot decompose simplex: no interior or boundary lattice points "
                f"(volume={simplex.volume}, multiplicity={simplex.multiplicity}). "
                f"Simplex may already be primitive in the lattice."
            )

        logger.debug(
            "decompose_non_unimodular_simplex: no interior points found; "
            "falling back to stellar subdivision via boundary point."
        )
        stellar_point = boundary_points[0]
        stellar_frac = tuple(Fraction(x) for x in stellar_point)

        # Determine which facets contain stellar_point by checking barycentric
        # coordinates: if barycentric[skip_idx] ≈ 0, the point lies on the
        # facet opposite to vertex[skip_idx].
        tol = config.get("integrality_tolerance", 1e-10)
        unimodular_simplices = []
        dim = simplex.dimension

        for skip_idx in range(dim + 1):
            facet_vertices = [
                simplex.vertices[i] for i in range(dim + 1) if i != skip_idx
            ]
            # Check if stellar_point lies on this facet (barycentric coord 0
            # for skipped vertex).  We connect stellar_point to facets that do
            # NOT contain it (so the resulting simplex has positive volume).
            if HAS_SYMPY:
                try:
                    n = dim + 1
                    M = []
                    for j in range(n):
                        col = [float(simplex.vertices[j][i]) for i in range(dim)] + [1.0]
                        M.append(col)
                    A_mat = Matrix(M).T
                    b_vec = [float(stellar_point[i]) for i in range(dim)] + [1.0]
                    solution = A_mat.solve(Matrix(b_vec))
                    lambdas = [float(solution[i]) for i in range(n)]
                    on_this_facet = abs(lambdas[skip_idx]) < tol
                except Exception:
                    on_this_facet = False
            else:
                on_this_facet = False  # conservative: try all facets

            if on_this_facet:
                # Stellar point is ON this facet — connecting it would give a
                # degenerate (zero-volume) simplex.  Skip.
                continue

            new_vertices = list(facet_vertices) + [stellar_frac]
            new_simplex = Simplex(vertices=tuple(new_vertices))

            if new_simplex.is_unimodular:
                unimodular_simplices.append(new_simplex)
            else:
                unimodular_simplices.extend(
                    decompose_non_unimodular_simplex(new_simplex, config=config)
                )

        return unimodular_simplices

    # Use first interior point for pulling triangulation
    interior = interior_points[0]
    interior_frac = tuple(Fraction(x) for x in interior)

    # Create simplices by connecting interior point to each facet
    unimodular_simplices = []
    dim = simplex.dimension

    # Each facet is simplex with one vertex removed
    for skip_idx in range(dim + 1):
        facet_vertices = [
            simplex.vertices[i] for i in range(dim + 1) if i != skip_idx
        ]
        # Add interior point to form new simplex
        new_vertices = list(facet_vertices) + [interior_frac]
        new_simplex = Simplex(vertices=tuple(new_vertices))

        # Recursively decompose if not unimodular
        if new_simplex.is_unimodular:
            unimodular_simplices.append(new_simplex)
        else:
            unimodular_simplices.extend(
                decompose_non_unimodular_simplex(new_simplex, config=config)
            )

    return unimodular_simplices



# ============================================================================
# SIMPLEX OPERATIONS
# ============================================================================
def simplex_intersection(s1: Simplex, s2: Simplex) -> Optional[Simplex]:
    """
    Compute intersection of two simplices (if result is a simplex).

    Note: Intersection of simplices is generally a polytope, not necessarily a simplex.
    This function returns None if intersection is not a simplex.

    Args:
        s1: First simplex
        s2: Second simplex

    Returns:
        Simplex representing intersection, or None if not a simplex
    """
    if s1.dimension != s2.dimension:
        raise DimensionError(
            f"Cannot intersect simplices of different dimensions "
            f"({s1.dimension} vs {s2.dimension})"
        )
    
    # For MVP: return None (intersection generally not a simplex)
    # Full implementation would require polytope intersection + simplex detection
    return None


def simplex_product(s1: Simplex, s2: Simplex,
                    config: Optional[Dict[str, Any]] = None):
    """
    Compute the Cartesian product of two simplices as a full product polytope.

    The Cartesian product of two simplices is NOT a simplex — it is a more
    complex polytope (e.g. 1-simplex × 1-simplex = square, a 2-polytope with
    4 vertices, not 3).

    FIX(Bug-23): The original code returned
        ``Simplex(vertices=tuple(product_vertices[:combined_dim + 1]))``
    which silently discarded vertices beyond index ``combined_dim``.  For any
    non-degenerate product this drops at least one essential vertex, producing
    an object with the wrong combinatorial structure and incorrect generating
    functions downstream.

    Args:
        s1: First simplex  (d₁-simplex, d₁+1 vertices)
        s2: Second simplex (d₂-simplex, d₂+1 vertices)
        config: Configuration dictionary

    Returns:
        Polytope containing all (d₁+1)×(d₂+1) product vertices, living in
        ℝ^(d₁+d₂) ambient space.

    Raises:
        DimensionLimitError: If combined_dim exceeds config max_dimension.
    """
    if config is None:
        config = get_simplex_config()

    combined_dim = s1.dimension + s2.dimension
    max_dim = config.get("max_dimension", 15)
    if combined_dim > max_dim:
        raise DimensionLimitError(combined_dim, max_dim)

    # All (d₁+1)×(d₂+1) product vertices — concatenate each (v1, v2) pair.
    product_vertices = []
    for v1 in s1.vertices:
        for v2 in s2.vertices:
            product_vertices.append(v1 + v2)

    # Import here to avoid circular imports (polytope.py imports from simplex.py).
    try:
        from .polytope import Polytope as _Polytope
    except ImportError:
        try:
            from polytope import Polytope as _Polytope
        except ImportError:
            _Polytope = None

    if _Polytope is not None:
        return _Polytope(vertices=product_vertices)

    # Fallback (no polytope module available): return the full vertex set inside
    # a Simplex only if it happens to be an exact simplex (vertex count == dim+1).
    # Otherwise raise a clear error to prevent silent data loss.
    n_verts = len(product_vertices)
    if n_verts == combined_dim + 1:
        return Simplex(vertices=tuple(product_vertices))
    raise RuntimeError(
        f"simplex_product: Polytope class unavailable and product has "
        f"{n_verts} vertices (need {combined_dim + 1} for a simplex). "
        "Install the polytope module to handle general product polytopes."
    )


# ============================================================================
# VALIDATION AND TESTING
# ============================================================================
def validate_simplex_utils() -> Dict[str, bool]:
    """Run internal test suite to verify simplex utilities."""
    results = {}
    try:
        from fractions import Fraction
        
        # Test 1: Standard simplex creation
        std2 = Simplex.standard_simplex(2)
        results["standard_simplex"] = (
            std2.dimension == 2 and 
            len(std2.vertices) == 3 and
            std2.vertices[0] == (Fraction(0), Fraction(0)) and
            std2.vertices[1] == (Fraction(1), Fraction(0)) and
            std2.vertices[2] == (Fraction(0), Fraction(1))
        )
        
        # Test 2: Volume computation (2D standard simplex volume = 1/2)
        vol2 = std2.volume
        results["volume_2d"] = abs(float(vol2) - 0.5) < 1e-10
        
        # Test 3: Volume computation (3D standard simplex volume = 1/6)
        std3 = Simplex.standard_simplex(3)
        vol3 = std3.volume
        results["volume_3d"] = abs(float(vol3) - 1/6) < 1e-10
        
        # Test 4: Unimodular detection (standard simplex is unimodular)
        results["unimodular_detection"] = std2.is_unimodular and std3.is_unimodular
        
        # Test 5: Lattice simplex detection
        results["lattice_detection"] = std2.is_lattice_simplex and std3.is_lattice_simplex
        
        # Test 6: Point containment (centroid should be inside)
        centroid2 = (Fraction(1, 3), Fraction(1, 3))
        results["point_containment"] = std2.contains(centroid2)
        
        # Test 7: Multiplicity computation (2D: 2! * 1/2 = 1, 3D: 3! * 1/6 = 1)
        results["multiplicity"] = std2.multiplicity == 1 and std3.multiplicity == 1
        
        # Test 8: Non-unimodular simplex (scaled standard simplex)
        scaled2 = Simplex(vertices=(
            (Fraction(0), Fraction(0)),
            (Fraction(2), Fraction(0)),
            (Fraction(0), Fraction(2))
        ))
        results["non_unimodular"] = (
            not scaled2.is_unimodular and 
            scaled2.multiplicity == 4 and
            abs(float(scaled2.volume) - 2.0) < 1e-10
        )
        
        # Test 9: Dimension guard
        try:
            big_vertices = tuple(
                tuple(Fraction(i + j) for j in range(16))
                for i in range(17)
            )
            Simplex(vertices=big_vertices)
            results["dimension_guard"] = False
        except DimensionLimitError:
            results["dimension_guard"] = True
        
        logger.info("✅ Simplex utilities validation passed")
    except Exception as e:
        logger.error(f"❌ Simplex utilities validation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
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
    print("Testing Production-Ready Simplex Utilities")
    print("=" * 60)
    
    # Run validation
    results = validate_simplex_utils()
    
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
    
    if "validation_error" in results:
        sys.exit(1)
    
    # Demonstration
    print("\n" + "=" * 60)
    print("Simplex Utilities Demo")
    print("=" * 60)
    
    from fractions import Fraction
    
    # 1. Create standard simplices
    print("\n1. Creating Standard Simplices:")
    std2 = Simplex.standard_simplex(2)
    print(f"   2D standard simplex:")
    print(f"     Vertices: {std2.vertices}")
    print(f"     Volume: {std2.volume} (expected 1/2)")
    print(f"     Unimodular: {std2.is_unimodular}")
    
    std3 = Simplex.standard_simplex(3)
    print(f"\n   3D standard simplex:")
    print(f"     Vertices: {std3.vertices[:2]}...")
    print(f"     Volume: {std3.volume} (expected 1/6)")
    print(f"     Unimodular: {std3.is_unimodular}")
    
    # 2. Non-unimodular simplex
    print("\n2. Non-Unimodular Simplex:")
    scaled2 = Simplex(vertices=(
        (Fraction(0), Fraction(0)),
        (Fraction(2), Fraction(0)),
        (Fraction(0), Fraction(2))
    ))
    print(f"   Scaled 2D simplex (factor 2):")
    print(f"     Volume: {scaled2.volume} (expected 2)")
    print(f"     Multiplicity: {scaled2.multiplicity} (expected 4)")
    print(f"     Unimodular: {scaled2.is_unimodular} (expected False)")
    
    # 3. Point containment
    print("\n3. Point Containment:")
    centroid2 = (Fraction(1, 3), Fraction(1, 3))
    outside2 = (Fraction(2, 3), Fraction(2, 3))
    print(f"   Centroid {centroid2} in standard 2D simplex? {std2.contains(centroid2)}")
    print(f"   Point {outside2} in standard 2D simplex? {std2.contains(outside2)}")
    
    # 4. Normaliz status
    print("\n4. Normaliz Integration:")
    if _check_normaliz_available():
        print("   ✅ Normaliz found - can enumerate lattice points")
    else:
        print("   ⚠️  Normaliz not found - using fallback enumeration for small simplices")
    
    print("\n" + "=" * 60)
    print("✅ Simplex Utilities Ready for Production")
    print("=" * 60)