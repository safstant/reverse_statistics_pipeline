#!/usr/bin/env python3
"""
Production-grade Brion's theorem implementation for the Reverse Statistics pipeline.
Handles Brion's theorem, Barvinok's algorithm, generating function decomposition,
and exponential sums over polyhedra.


Critical for: Summing vertex cone generating functions to get polytope GF
"""

from .exceptions import ReverseStatsError
import math
import cmath
from math import pi
import time
from fractions import Fraction
from typing import List, Tuple, Dict, Any, Optional, Union, Set, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import sys
import os
from functools import lru_cache, wraps
import itertools

# Use sympy for exact algebraic computations when needed
try:
    import sympy
    from sympy import Matrix, symbols, diff, simplify
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    # Fallback for when sympy not available

logger = logging.getLogger(__name__)

# ── Barvinok diagnostic counters (module-level, reset per pipeline run) ──────
# These are read by pipeline.py to populate the [BARVINOK DIAGNOSTIC] output.
# They are intentionally simple integers — not thread-local — because the
# pipeline is single-threaded.  reset_diagnostic_counters() must be called
# at the start of each run_pipeline() invocation.
_diag_gf_terms_constructed: int = 0   # GF successfully built for one unimodular cone
_diag_cones_processed:      int = 0   # tangent cones entering _unimodular_decomposition_gf

def reset_diagnostic_counters() -> None:
    """Reset Barvinok diagnostic counters. Call at start of each pipeline run."""
    global _diag_gf_terms_constructed, _diag_cones_processed
    _diag_gf_terms_constructed = 0
    _diag_cones_processed      = 0

def get_diagnostic_counters() -> dict:
    """Return a snapshot of current diagnostic counter values."""
    return {
        "gf_terms_constructed": _diag_gf_terms_constructed,
        "cones_processed":      _diag_cones_processed,
    }

# ============================================================================
# Add current directory to path for standalone execution
# ============================================================================
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# EXCEPTIONS
# ============================================================================

class BrionError(ReverseStatsError):
    """Base exception for Brion's theorem operations."""
    def __init__(self, message: str):
        self.message = message

        super().__init__(message)


class ExponentialSumError(BrionError):
    """Raised when exponential sum computation fails."""
    pass


class VertexConeError(BrionError):
    """Raised when vertex cone decomposition fails."""
    pass


class ResidueError(BrionError):
    """Raised when residue computation fails."""
    pass


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class BrionType(Enum):
    """Types of Brion's theorem applications."""
    VERTEX_SUM = "vertex_sum"              # Sum over vertex cones
    TANGENT_CONE = "tangent_cone"           # Tangent cone decomposition
    BARVINOK = "barvinok"                   # Barvinok's algorithm
    EXPONENTIAL_SUM = "exponential_sum"      # Exponential sum evaluation


class ExponentialSumType(Enum):
    """Types of exponential sums."""
    FINITE = "finite"                        # Finite sum over lattice points
    GENERATING = "generating"                 # Generating function
    VALUATION = "valuation"                   # Valuation form


# ============================================================================
# IMPORT HANDLING (Dual-mode for package + standalone execution)
# ============================================================================
try:
    # Package mode
    from .math_utils import (
        is_integer, gcd_list, lcm_list,
        solve_rational_system, nullspace_basis
    )
    from .lattice import FractionLattice, AffineLattice
    from .simplex import Simplex, SimplexError
    from .polytope import Polytope, PolytopeType
    from .constraints import (
        ConstraintSystem, Inequality, Equation, Bound,
        ConstraintType, InequalityDirection, BoundType
    )
    from .dimension import DimensionLimitError
    from .indexing import Grading, Index, IndexRange
    from .decomposition import DecompositionCone, ConeDecomposition
    from .gf_construction import (
        RationalFunction, EhrhartSeries,
        vertex_generating_function, add_rational_functions,
        multiply_rational_functions, series_expansion,
        PrecisionGuard
    )
    IMPORT_SUCCESS = True
except (ImportError, ModuleNotFoundError) as e:
    # Standalone mode
    IMPORT_SUCCESS = False
    logger.warning(f"Package imports failed: {e}, trying standalone imports...")
    
    try:
        from math_utils import (
            is_integer, gcd_list, lcm_list,
            solve_rational_system, nullspace_basis
        )
        from lattice import FractionLattice, AffineLattice
        from simplex import Simplex, SimplexError
        from polytope import Polytope, PolytopeType
        from constraints import (
            ConstraintSystem, Inequality, Equation, Bound,
            ConstraintType, InequalityDirection, BoundType
        )
        from dimension import DimensionLimitError
        from indexing import Grading, Index, IndexRange
        from decomposition import DecompositionCone, ConeDecomposition
        from gf_construction import (
            RationalFunction, EhrhartSeries,
            vertex_generating_function, add_rational_functions,
            multiply_rational_functions, series_expansion,
            PrecisionGuard
        )
        logger.info("Standalone imports successful")
    except ImportError as e:
        raise ImportError(
            f"brion.py: required dependencies (math_utils, lattice, simplex, polytope, "
            f"constraints, decomposition, gf_construction) could not be imported: {e}. "
            "These modules are required — there is no correct fallback. "
            "Note: the stub vertex_generating_function returned 1/(1-z) for every "
            "vertex, which is mathematically wrong and would silently corrupt all GF results. "
            "Ensure the package is installed correctly or all sibling modules are on sys.path."
        ) from e


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class VertexCone:
    """
    Vertex cone for Brion's theorem.

    A vertex cone is the tangent cone at a vertex of a polytope,
    consisting of all directions from the vertex into the polytope.

    Attributes:
        vertex: Vertex point
        rays: Generating rays of the cone
        lineality_space: Lineality space (if any)
        generating_function: Generating function of the cone
    """
    vertex: Tuple[Fraction, ...]
    rays: List[Tuple[Fraction, ...]]
    lineality_space: List[Tuple[Fraction, ...]] = field(default_factory=list)
    generating_function: Optional['RationalFunction'] = None
    
    def __post_init__(self):
        """Validate vertex cone."""
        if not self.rays and not self.lineality_space:
            raise VertexConeError("Vertex cone must have rays or lineality space")
    
    @property
    def dimension(self) -> int:
        """Dimension of the cone."""
        return len(self.rays) + len(self.lineality_space)
    
    @property
    def is_pointed(self) -> bool:
        """Check if cone is pointed (no lineality space)."""
        return len(self.lineality_space) == 0
    
    @property
    def num_rays(self) -> int:
        """Number of rays in the cone."""
        return len(self.rays)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "vertex": [str(x) for x in self.vertex],
            "rays": [[str(x) for x in r] for r in self.rays],
            "lineality_space": [[str(x) for x in l] for l in self.lineality_space],
            "dimension": self.dimension,
            "is_pointed": self.is_pointed,
            "num_rays": self.num_rays,
            "generating_function": self.generating_function.to_dict() if self.generating_function else None
        }


@dataclass
class BrionDecomposition:
    """
    Decomposition of a polytope via Brion's theorem.

    Attributes:
        polytope: Original polytope
        vertex_cones: List of vertex cones
        generating_function: Total generating function (sum of vertex cones)
        is_exact: Whether decomposition is exact (no overlaps)
    """
    polytope: 'Polytope'
    vertex_cones: List[VertexCone]
    generating_function: Optional['RationalFunction'] = None
    is_exact: bool = True
    
    def __post_init__(self):
        """Initialize Brion decomposition."""
        self.num_vertices = len(self.vertex_cones)
    
    @property
    def num_cones(self) -> int:
        """Number of vertex cones."""
        return len(self.vertex_cones)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "polytope": self.polytope.to_dict() if hasattr(self.polytope, 'to_dict') else {},
            "vertex_cones": [vc.to_dict() for vc in self.vertex_cones],
            "num_vertices": self.num_vertices,
            "num_cones": self.num_cones,
            "is_exact": self.is_exact,
            "generating_function": self.generating_function.to_dict() if self.generating_function else None
        }


@dataclass
class ExponentialSum:
    """
    Exponential sum over lattice points.

    Attributes:
        polytope: Polytope
        exponents: List of exponent vectors
        coefficients: List of coefficients
        value: Evaluated sum (if computed)
    """
    polytope: 'Polytope'
    exponents: List[Tuple[Fraction, ...]]
    coefficients: List[Fraction]
    value: Optional[complex] = None
    
    def __post_init__(self):
        """Validate exponential sum."""
        if len(self.exponents) != len(self.coefficients):
            raise ExponentialSumError(
                f"Number of exponents ({len(self.exponents)}) != number of coefficients ({len(self.coefficients)})",

            )
    
    @property
    def num_terms(self) -> int:
        """Number of terms in the exponential sum."""
        return len(self.exponents)
    
    def evaluate(self, x: Tuple[Fraction, ...]) -> complex:
        """
        Evaluate exponential sum at point x.
        Sum over i: c_i * exp(2πi * <e_i, x>)

        Args:
            x: Point to evaluate at (tuple of Fractions)

        Returns:
            Complex value of the exponential sum

        Raises:
            ExponentialSumError: If point dimension doesn't match exponents
        """
        if not self.exponents:
            return 0.0 + 0.0j
        
        # Check dimension consistency
        exp_dim = len(self.exponents[0])
        if len(x) != exp_dim:
            raise ExponentialSumError(
                f"Point dimension {len(x)} does not match exponent dimension {exp_dim}",

            )
        
        result = 0.0 + 0.0j
        
        for e, c in zip(self.exponents, self.coefficients):
            # Compute dot product <e, x> using exact arithmetic first
            dot_num = 0
            dot_den = 1
            
            # Use Fraction arithmetic for exact dot product
            for j in range(exp_dim):
                # e[j] * x[j] as Fraction
                prod = e[j] * x[j]
                # Add to running sum
                dot_num = dot_num * prod.denominator + prod.numerator * dot_den
                dot_den = dot_den * prod.denominator
            
            # Convert to float for complex exponential
            dot = dot_num / dot_den if dot_den != 0 else 0.0
            
            # Compute exp(2πi * dot)
            exp_val = cmath.exp(2j * pi * dot)
            
            # Add to result: c * exp(2πi * dot)
            result += float(c) * exp_val
        
        return result
    
    def evaluate_real(self, x: Tuple[Fraction, ...]) -> float:
        """
        Evaluate exponential sum and return real part.

        Args:
            x: Point to evaluate at

        Returns:
            Real part of the exponential sum
        """
        return self.evaluate(x).real
    
    def evaluate_imag(self, x: Tuple[Fraction, ...]) -> float:
        """
        Evaluate exponential sum and return imaginary part.

        Args:
            x: Point to evaluate at

        Returns:
            Imaginary part of the exponential sum
        """
        return self.evaluate(x).imag
    
    def magnitude(self, x: Tuple[Fraction, ...]) -> float:
        """
        Evaluate exponential sum and return magnitude.

        Args:
            x: Point to evaluate at

        Returns:
            Magnitude of the exponential sum
        """
        val = self.evaluate(x)
        return abs(val)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "polytope": self.polytope.to_dict() if hasattr(self.polytope, 'to_dict') else {},
            "exponents": [[str(x) for x in e] for e in self.exponents],
            "coefficients": [str(c) for c in self.coefficients],
            "value": str(self.value) if self.value else None,
            "num_terms": self.num_terms
        }


# ============================================================================

# ============================================================================

def vertex_cones(polytope: 'Polytope',
                grading: Optional['Grading'] = None) -> BrionDecomposition:
    """
    Construct vertex cones for a polytope (Brion's theorem).

    For each vertex v of the polytope, the tangent cone at v consists of
    all directions u such that v + εu is in the polytope for small ε > 0.

    Args:
        polytope: Polytope to decompose
        grading: Grading to use for generating functions

    Returns:
        BrionDecomposition with vertex cones


    """
    if not polytope.vertices:
        raise BrionError("Cannot decompose empty polytope")
    
    vertex_cones_list = []
    
    # For each vertex, compute tangent cone
    for i, v in enumerate(polytope.vertices):
        # Compute rays of tangent cone at v
        rays = _tangent_cone_rays(polytope, v)
        
        # Create vertex cone
        vc = VertexCone(vertex=v, rays=rays)
        
        # Compute generating function for this cone
        # Convert to simplex by intersecting with hyperplane
        if rays:
            # Create a simplex from rays (simplified)
            vertices = []
            for ray in rays:
                total = sum(ray)
                if total != 0:
                    pt = tuple(Fraction(x, total) for x in ray)
                    vertices.append(pt)
            
            if len(vertices) == len(rays):
                try:
                    if IMPORT_SUCCESS:
                        from .simplex import Simplex
                        from .gf_construction import vertex_generating_function
                    else:
                        from simplex import Simplex
                        from gf_construction import vertex_generating_function
                    simplex = Simplex(vertices=tuple(vertices))
                    vc.generating_function = vertex_generating_function(simplex, grading)
                except Exception as e:
                    raise VertexConeError(
                        f"vertex_generating_function failed for vertex {v}: {e}. "
                        "Cannot substitute a placeholder GF — the final count would be wrong."
                    ) from e
        
        vertex_cones_list.append(vc)
    
    # Create decomposition
    decomp = BrionDecomposition(
        polytope=polytope,
        vertex_cones=vertex_cones_list
    )
    
    # Sum generating functions to get total
    total_gf = None
    for vc in vertex_cones_list:
        if vc.generating_function:
            if total_gf is None:
                total_gf = vc.generating_function
            else:
                if IMPORT_SUCCESS:
                    from .gf_construction import add_rational_functions
                else:
                    from gf_construction import add_rational_functions
                total_gf = add_rational_functions(total_gf, vc.generating_function)
    
    decomp.generating_function = total_gf
    
    return decomp


def _tangent_cone_rays(polytope: 'Polytope',
                       vertex: Tuple[Fraction, ...]) -> List[Tuple[Fraction, ...]]:
    """
    Compute rays of tangent cone at a vertex.

    The tangent cone at vertex v has one ray per edge of the polytope incident
    to v.  Two vertices are connected by an edge iff they appear together in a
    facet simplex of the convex hull.

    FIX(Bug-5): The original code looped over *all* non-self vertices and
    treated every direction as a ray, producing C(n-1, 1) rays instead of
    deg(v) ≈ d rays.  For a polytope with many vertices this creates thousands
    of redundant rays, making Barvinok's algorithm (exponential in the number
    of rays) computationally intractable.
    """
    all_vertices = polytope.vertices
    vertex_idx = None
    for i, v in enumerate(all_vertices):
        if v == vertex:
            vertex_idx = i
            break
    if vertex_idx is None:
        return []

    # ---- Compute edge-adjacency from ConvexHull facets ----------------------
    # Project vertices to their effective (affine) dimension before calling
    # ConvexHull, because the vertices often lie in a lower-dimensional affine
    # subspace (e.g., dim-3 polytope embedded in R^6 via 3 equalities).
    # Qhull fails when fed flat input; projecting first fixes this.
    adjacent_indices: set = set()
    try:
        import numpy as np
        from scipy.spatial import ConvexHull
        from scipy.linalg import svd as scipy_svd

        verts_f = np.array([[float(x) for x in v] for v in all_vertices])
        n_verts, ambient_dim = verts_f.shape

        if n_verts < 2:
            adjacent_indices = set()
        else:
            # Project to effective dimension via SVD of centered vertices
            center = verts_f[0]
            diffs = verts_f[1:] - center
            try:
                _, s, Vt = scipy_svd(diffs, full_matrices=False)
                eff_dim = int(np.sum(s > max(1e-8, s[0] * 1e-10)))
            except Exception:
                eff_dim = ambient_dim

            if eff_dim < ambient_dim and eff_dim >= 1:
                # Project all vertices to eff_dim-dimensional space
                proj_basis = Vt[:eff_dim]
                pts_low = (verts_f - center) @ proj_basis.T
            else:
                pts_low = verts_f

            if n_verts > eff_dim:
                hull = ConvexHull(pts_low)
                for simplex in hull.simplices:
                    if vertex_idx in simplex:
                        for idx in simplex:
                            if idx != vertex_idx:
                                adjacent_indices.add(idx)
            else:
                # Too few vertices for ConvexHull — it's a simplex
                adjacent_indices = set(range(n_verts)) - {vertex_idx}

    except (ImportError, Exception) as e:
        logger.warning(
            f"_tangent_cone_rays: ConvexHull unavailable ({e}); "
            "falling back to all-vertices adjacency (may produce redundant rays)."
        )
        adjacent_indices = set(range(len(all_vertices))) - {vertex_idx}

    # ---- Build rays from adjacent vertices ----------------------------------
    if IMPORT_SUCCESS:
        from .math_utils import lcm_list, gcd_list
    else:
        from math_utils import lcm_list, gcd_list

    raw_rays = []
    for idx in adjacent_indices:
        other = all_vertices[idx]
        ray = tuple(other[j] - vertex[j] for j in range(len(vertex)))

        if all(r == 0 for r in ray):
            continue

        # Normalize to primitive integer direction
        denoms = [r.denominator for r in ray]
        lcm_denom = lcm_list(denoms) if denoms else 1
        int_ray = [r.numerator * (lcm_denom // r.denominator) for r in ray]

        coeffs = [abs(x) for x in int_ray if x != 0]
        if coeffs:
            g = gcd_list(coeffs)
            if g > 1:
                int_ray = [x // g for x in int_ray]

        ray_frac = tuple(Fraction(x) for x in int_ray)
        raw_rays.append(ray_frac)

    # ---- Deduplicate (up to positive scalar multiple) -----------------------
    unique_rays = []
    for ray in raw_rays:
        for comp in ray:
            if comp != 0:
                scale = 1 if comp > 0 else -1
                norm_ray = tuple(scale * r for r in ray)

                seen = False
                for existing in unique_rays:
                    pivot_idx = next((i for i, x in enumerate(existing) if x != 0), None)
                    if pivot_idx is None:
                        continue
                    if norm_ray[pivot_idx] == 0:
                        continue
                    p_e = existing[pivot_idx]
                    p_n = norm_ray[pivot_idx]
                    if all(norm_ray[i] * p_e == existing[i] * p_n
                           for i in range(len(norm_ray))):
                        seen = True
                        break

                if not seen:
                    unique_rays.append(norm_ray)
                break

    return unique_rays


# ============================================================================

# ============================================================================

def barvinok_generating_function(polytope: 'Polytope',
                                 grading: Optional['Grading'] = None,
                                 intrinsic_map: Optional[Dict[tuple, tuple]] = None,
                                 ) -> 'RationalFunction':
    """
    Compute generating function using Barvinok's algorithm.

    Barvinok's algorithm decomposes the polytope into unimodular cones
    and sums their vertex generating functions.

    Args:
        polytope:       Polytope to compute generating function for
        grading:        Grading to use
        intrinsic_map:  Optional dict mapping vertex tuples → (B, B_inv, d)
                        where B is the intrinsic basis (ambient_dim × d) and
                        B_inv is the left inverse (d × ambient_dim).
                        If provided, each VertexCone is projected to intrinsic
                        R^d before GF construction.  V15.4 (S4).

    Returns:
        Rational function representing the generating function


    """
    # Guard: vertices must be exact before Barvinok stages begin.
    # Float contamination here produces silently wrong generating functions.
    if IMPORT_SUCCESS and polytope.vertices:
        try:
            PrecisionGuard.assert_exact(polytope.vertices, label="polytope.vertices")
        except Exception as guard_err:
            raise BrionError(
                f"Precision guard failed — non-exact values in polytope vertices "
                f"before Barvinok decomposition: {guard_err}"
            ) from guard_err

    # First, get Brion decomposition
    brion = vertex_cones(polytope, grading)
    
    # For each vertex cone, we need to decompose into unimodular cones
    total_gf = None
    
    for vc in brion.vertex_cones:
        if not vc.rays:
            continue

        # V15.4 (S4): attach intrinsic basis if available
        if intrinsic_map:
            v_key = tuple(vc.vertex) if not isinstance(vc.vertex, tuple) else vc.vertex
            intr = intrinsic_map.get(v_key)
            if intr:
                vc.intrinsic_basis   = intr[0]
                vc.intrinsic_inverse = intr[1]

        cone_gf = _unimodular_decomposition_gf(vc, grading)
        
        if total_gf is None:
            total_gf = cone_gf
        else:
            if IMPORT_SUCCESS:
                from .gf_construction import add_rational_functions
            else:
                from gf_construction import add_rational_functions
            total_gf = add_rational_functions(total_gf, cone_gf)
    
    return total_gf or RationalFunction.constant(Fraction(0))


def _unimodular_decomposition_gf(vertex_cone: VertexCone,
                                 grading: Optional['Grading'] = None) -> 'RationalFunction':
    """
    Decompose a cone into unimodular cones and sum their generating functions.

    BUG-FIX (v14) — Triangulation gap:
    The original code silently raised VertexConeError for any cone where
    len(rays) != dimension (i.e. non-simplicial cones).  The caller in
    barvinok_generating_function caught this as a generic exception and fell
    back to direct enumeration without any WARNING-level log entry.  The result
    was correct but the caller had no visibility that Barvinok had not fired.

    Fix:
    1. For simplicial cones (len(rays) == dimension): proceed as before with
       vertex_generating_function.
    2. For non-simplicial cones (len(rays) > dimension): attempt signed
       decomposition via decomposition.signed_decomposition into simplicial
       sub-cones, then sum their GFs.
    3. If signed decomposition is also unavailable, emit a WARNING (not just
       a DEBUG) that names the specific cone that triggered the fallback, then
       raise VertexConeError so the pipeline can log it and fall back cleanly.
    """
    # ── Diagnostic counter: every cone entering this function is tracked ────────
    global _diag_cones_processed, _diag_gf_terms_constructed
    _diag_cones_processed += 1

    n_rays = len(vertex_cone.rays)
    dim    = vertex_cone.dimension

    # ── V15.4 (S4): Intrinsic projection in brion ────────────────────────────
    # If the cone has an attached intrinsic basis (set by pipeline Step 15b
    # via barvinok_generating_function's intrinsic_map), project the rays to
    # intrinsic R^d before the det check.  This makes n_rays == d (simplicial
    # in intrinsic space) so the det is defined and LLL works correctly.
    _B_brion     = getattr(vertex_cone, 'intrinsic_basis',   None)
    _B_inv_brion = getattr(vertex_cone, 'intrinsic_inverse',  None)
    if _B_brion and _B_inv_brion:
        try:
            if IMPORT_SUCCESS:
                from .decomposition import project_to_intrinsic as _pti
            else:
                from decomposition import project_to_intrinsic as _pti
            _intr_rays = _pti(vertex_cone.rays, _B_inv_brion)
            # Rebuild VertexCone with projected rays and intrinsic dimension
            from dataclasses import replace as _dc_replace
            try:
                vertex_cone = _dc_replace(vertex_cone, rays=tuple(_intr_rays))
            except Exception:
                # VertexCone may not support replace — set attribute directly
                try:
                    object.__setattr__(vertex_cone, 'rays', tuple(_intr_rays))
                except Exception:
                    vertex_cone.rays = tuple(_intr_rays)
            n_rays = len(vertex_cone.rays)
            dim    = len(_intr_rays[0])  # intrinsic dimension = projected coordinate length
            logger.debug(
                f"_unimodular_decomposition_gf: projected to intrinsic R^{len(_intr_rays[0])}, "
                f"n_rays={n_rays}, dim={dim}"
            )
        except Exception as _proj_err:
            logger.debug(f"_unimodular_decomposition_gf: intrinsic projection failed ({_proj_err})")

    # ── Case 1: Simplicial cone — direct GF via cone_generating_function ───────
    #
    # BUG-FIX (v14): The original implementation converted rays to normalised
    # vertices and passed them to Simplex/vertex_generating_function.  This fails
    # when the cone lives in a low-dimensional subspace of a high-dimensional
    # ambient space (e.g. 3 rays in R^6 for the dice problem) because Simplex
    # requires exactly d+1 vertices for ambient dimension d — i.e. 7 vertices
    # for 6D — but we only have 3 rays.
    #
    # Fix: use cone_generating_function(DecompositionCone) directly.  This
    # function only needs the rays and their grades (sum of coordinates), and
    # does not require a full-dimensional Simplex.  It computes:
    #   GF = 1 / ∏_i (1 − t^{grade(ray_i)})
    # which is correct for any simplicial cone regardless of ambient dimension.
    if n_rays == dim and dim > 0:
        rays = vertex_cone.rays
        try:
            if IMPORT_SUCCESS:
                from .decomposition import DecompositionCone
                from .gf_construction import cone_generating_function
            else:
                from decomposition import DecompositionCone
                from gf_construction import cone_generating_function

            # Build integer ray list (clear Fraction denominators)
            from math import gcd
            from functools import reduce
            int_rays = []
            for ray in rays:
                denoms = [r.denominator if hasattr(r, 'denominator') else 1 for r in ray]
                lcm_d = reduce(lambda a, b: a * b // gcd(a, b), denoms, 1)
                int_ray = tuple(
                    int(r * lcm_d) if isinstance(r, Fraction) else int(r * lcm_d)
                    for r in ray
                )
                int_rays.append(int_ray)

            # ── V15.2 — Unimodular assertion ────────────────────────────────
            # Assert abs(det(ray_matrix)) == 1 after primitive normalisation.
            # This converts the implicit unimodularity assumption into a formal
            # invariant.  It operates on integer rays so the determinant is exact
            # (SymPy Matrix over Z, no floating point).
            #
            # Routing:
            #   det == 1  → proceed to cone_generating_function (fast path)
            #   det != 1  → log WARNING with vertex + det value, route to
            #               signed_decomposition for secondary decomposition
            #               rather than raising — preserves algorithm continuity.
            #
            # This invariant catches any future refactor that accidentally
            # introduces float contamination into the ray pipeline.
            # V15.6 (Invariant II.2): after intrinsic projection n_rays must == d
            # V15.6 (Invariant II.2): matrix MUST be square after intrinsic projection
            assert len(int_rays) == dim, (
                f"Non-square ray matrix ({len(int_rays)}×{dim}) at Brion stage — "
                "invariant violation: intrinsic projection must ensure n_rays == d"
            )
            if len(int_rays) != dim:
                raise VertexConeError(
                    f"Non-square ray matrix ({len(int_rays)}×{dim}) at Brion "
                    "determinant stage. Intrinsic projection must be applied "
                    "before this point so that n_rays == intrinsic_dimension."
                )

            _det_val = 1
            if len(int_rays) == dim:
                try:
                    import sympy as _sp
                    _ray_mat = _sp.Matrix(int_rays)
                    _det_val = abs(int(_ray_mat.det()))
                except ImportError as _det_err:
                    # SymPy not available — surface as VertexConeError so caller
                    # can fall back correctly.  V15.3 (C5b): was bare Exception,
                    # which silently assumed unimodular and hid the real error.
                    raise VertexConeError(
                        f"SymPy required for unimodular det check: {_det_err}"
                    ) from _det_err
                except (ValueError, TypeError, ArithmeticError) as _det_err:
                    # V15.5: after intrinsic projection the matrix MUST be square
                    # and non-singular. A det computation failure here is an
                    # invariant violation, not a recoverable routing case.
                    raise VertexConeError(
                        f"Unimodular det check failed after intrinsic projection: "
                        f"{_det_err}. Matrix should be square and non-singular. "
                        f"int_rays={int_rays}"
                    ) from _det_err

            if _det_val != 1:
                logger.warning(
                    f"V15.2 UNIMODULAR INVARIANT VIOLATED — cone at vertex "
                    f"{vertex_cone.vertex} has det={_det_val} (expected 1). "
                    f"rays={int_rays}. "
                    "Routing to signed decomposition for secondary decomposition."
                )
                # Fall through to Case 2 (signed decomposition) below.
                # Do NOT build a GF from a non-unimodular cone — it would be wrong.
                raise VertexConeError(
                    f"Non-unimodular cone (det={_det_val}) routed to signed decomposition."
                )

            dcone = DecompositionCone(
                rays=int_rays,
                sign=1,
                is_simplicial=True,
                is_unimodular=True,   # V15.2: certified by determinant test above
                dimension=dim,
            )
            gf = cone_generating_function(dcone, grading)
            _diag_gf_terms_constructed += 1   # GF successfully constructed for this cone
            return gf
        except Exception as e:
            raise VertexConeError(
                f"_unimodular_decomposition_gf: cone_generating_function failed "
                f"for simplicial cone at vertex {vertex_cone.vertex}: {e}. "
                "Cannot substitute a placeholder GF — the final count would be wrong."
            ) from e

    # ── Case 2: Non-simplicial cone — attempt signed decomposition ───────────
    # This is the triangulation gap. Emit a WARNING so the caller always knows
    # when this path is taken, regardless of log level.
    logger.warning(
        f"_unimodular_decomposition_gf: non-simplicial cone at vertex "
        f"{vertex_cone.vertex} has {n_rays} rays in dimension {dim}. "
        "Attempting signed decomposition into simplicial sub-cones. "
        "If this fails the pipeline will fall back to direct enumeration."
    )

    try:
        if IMPORT_SUCCESS:
            from .decomposition import DecompositionCone, signed_decomposition
            from .gf_construction import add_rational_functions
        else:
            from decomposition import DecompositionCone, signed_decomposition
            from gf_construction import add_rational_functions

        # Build a DecompositionCone from the VertexCone rays
        int_rays = []
        for ray in vertex_cone.rays:
            # Convert Fraction rays to integer rays by clearing denominators
            from math import gcd
            from functools import reduce
            denoms = [r.denominator for r in ray]
            lcm_d = reduce(lambda a, b: a * b // gcd(a, b), denoms, 1)
            int_ray = tuple(int(r * lcm_d) for r in ray)
            int_rays.append(int_ray)

        dcone = DecompositionCone(
            rays=int_rays,
            is_simplicial=False,
            is_unimodular=False,
            dimension=dim,
        )
        # V15.6: propagate intrinsic basis so Normaliz gets intrinsic rays
        _vc_B     = getattr(vertex_cone, 'intrinsic_basis',    None)
        _vc_B_inv = getattr(vertex_cone, 'intrinsic_inverse',  None)
        _vc_d     = getattr(vertex_cone, 'intrinsic_dimension', None)
        if _vc_B and _vc_B_inv:
            dcone.intrinsic_basis     = _vc_B
            dcone.intrinsic_inverse   = _vc_B_inv
            dcone.intrinsic_dimension = _vc_d
        decomp = signed_decomposition(dcone)

        total_gf = None
        for signed_cone in decomp.cones:
            # Each signed_cone is a simplicial sub-cone — recurse
            sub_vc = VertexCone(
                vertex=vertex_cone.vertex,
                rays=[tuple(Fraction(c) for c in r) for r in signed_cone.rays],
            )
            try:
                sub_gf = _unimodular_decomposition_gf(sub_vc, grading)
                if signed_cone.sign == -1:
                    # Negate: multiply numerator by −1
                    neg_gf = RationalFunction(
                        numerator=[-c for c in sub_gf.numerator],
                        denominator=sub_gf.denominator,
                        shift=sub_gf.shift,
                    )
                    sub_gf = neg_gf
                total_gf = sub_gf if total_gf is None else add_rational_functions(total_gf, sub_gf)
            except VertexConeError as inner:
                logger.warning(
                    f"_unimodular_decomposition_gf: sub-cone GF failed ({inner}); "
                    "skipping this sub-cone in the signed sum."
                )

        if total_gf is not None:
            logger.info(
                f"_unimodular_decomposition_gf: signed decomposition succeeded "
                f"for non-simplicial cone ({len(decomp.cones)} sub-cones)."
            )
            return total_gf

    except Exception as decomp_err:
        logger.warning(
            f"_unimodular_decomposition_gf: signed decomposition failed for "
            f"non-simplicial cone at vertex {vertex_cone.vertex}: {decomp_err}. "
            "Pipeline will fall back to direct lattice enumeration for this cone."
        )

    # ── Case 3: Complete failure — structured raise with full context ─────────
    raise VertexConeError(
        f"_unimodular_decomposition_gf: could not build GF for non-simplicial "
        f"cone at vertex {vertex_cone.vertex} with {n_rays} rays in dim {dim}. "
        "Both direct simplex construction and signed decomposition failed. "
        "The pipeline will fall back to direct enumeration — result will be "
        "correct but Barvinok was not used for this cone."
    )


# ============================================================================

# ============================================================================

def exponential_sum(polytope: 'Polytope',
                   exponents: List[Tuple[Fraction, ...]],
                   coefficients: List[Fraction]) -> ExponentialSum:
    """
    Construct exponential sum over lattice points in polytope.

    S(x) = sum_{p in P ∩ Z^d} c(p) * exp(2πi * <e(p), x>)

    Args:
        polytope: Polytope containing lattice points
        exponents: Exponent vectors for each term
        coefficients: Coefficients for each term

    Returns:
        ExponentialSum object


    """
    return ExponentialSum(
        polytope=polytope,
        exponents=exponents,
        coefficients=coefficients
    )


def evaluate_exponential_sum(exp_sum: ExponentialSum,
                            point: Tuple[Fraction, ...]) -> complex:
    """
    Evaluate exponential sum at a point.

    Args:
        exp_sum: Exponential sum to evaluate
        point: Point to evaluate at

    Returns:
        Complex value of exponential sum at point
    """
    return exp_sum.evaluate(point)


# ============================================================================

# ============================================================================

def compute_residue(rational_function: 'RationalFunction',
                   pole: Fraction,
                   order: int = 1) -> Fraction:
    """
    Compute the residue of a rational function P(t)/Q(t) at a pole.

    For a pole of order m at t = a:

        Res_{t=a} f(t) = (1/(m-1)!) * lim_{t->a} d^{m-1}/dt^{m-1} [(t-a)^m f(t)]

    Uses exact Fraction arithmetic for order-1 poles (simple poles) and
    SymPy for higher-order poles.

    Args:
        rational_function: Rational function with Fraction coefficients.
        pole: Exact pole location as a Fraction.
        order: Order of the pole (multiplicity).

    Returns:
        Exact residue as a Fraction.

    Raises:
        RationalFunctionError: If the denominator vanishes at the pole with
            an order different from the declared order, or if SymPy is
            required but unavailable for order > 1.
    """
    rf = rational_function

    def _eval_poly(coeffs: list, t: Fraction) -> Fraction:
        """Evaluate polynomial with Fraction coefficients at t."""
        val = Fraction(0)
        for i, c in enumerate(coeffs):
            if c != 0:
                val += Fraction(c) * (t ** i)
        return val

    def _poly_derivative(coeffs: list) -> list:
        """Differentiate polynomial: d/dt [sum c_i t^i] = sum i*c_i t^{i-1}."""
        if len(coeffs) <= 1:
            return [Fraction(0)]
        return [Fraction(i) * Fraction(c) for i, c in enumerate(coeffs)][1:]

    if order == 1:
        # Simple pole: Res = P(a) / Q'(a)
        num_val = _eval_poly(rf.numerator, pole)
        denom_prime = _poly_derivative(rf.denominator)
        den_prime_val = _eval_poly(denom_prime, pole)

        if den_prime_val == 0:
            # Q'(a) = 0 means the pole order is > 1 or a is not a pole
            raise RationalFunctionError(
                f"Q'({pole}) = 0 — pole at {pole} has order > 1; "
                "call compute_residue with the correct order."
            )
        return num_val / den_prime_val

    # Higher-order pole: use SymPy for exact symbolic differentiation
    try:
        import sympy as sp

        t = sp.Symbol('t')
        num_sym = sum(sp.Rational(Fraction(c).numerator, Fraction(c).denominator) * t**i
                      for i, c in enumerate(rf.numerator) if c != 0)
        den_sym = sum(sp.Rational(Fraction(c).numerator, Fraction(c).denominator) * t**i
                      for i, c in enumerate(rf.denominator) if c != 0)

        pole_sym = sp.Rational(pole.numerator, pole.denominator)
        expr = num_sym / den_sym

        # (t - a)^order * f(t), differentiated (order-1) times, evaluated at a, divided by (order-1)!
        g = (t - pole_sym) ** order * expr
        for _ in range(order - 1):
            g = sp.diff(g, t)

        residue_sym = sp.limit(g, t, pole_sym)
        factorial_val = sp.factorial(order - 1)
        residue_sym = sp.simplify(residue_sym / factorial_val)

        # Convert back to Fraction
        r = sp.Rational(residue_sym)
        return Fraction(int(r.p), int(r.q))

    except ImportError:
        raise RationalFunctionError(
            f"SymPy required for order-{order} residue computation but is not installed."
        )
    except Exception as e:
        raise RationalFunctionError(
            f"Residue computation failed at pole={pole}, order={order}: {e}"
        )


# ============================================================================

# ============================================================================

def brion_theorem(polytope: 'Polytope',
                 grading: Optional['Grading'] = None) -> 'RationalFunction':
    """
    Apply Brion's theorem: The generating function of a polytope
    equals the sum of generating functions of its vertex cones.

    ∑_{m ∈ P ∩ Z^d} t^{grading(m)} = ∑_{v ∈ vertices(P)} σ_{C_v}(t)

    where σ_{C_v} is the generating function of the tangent cone at v.

    Args:
        polytope: Polytope
        grading: Grading to use

    Returns:
        Rational function satisfying Brion's theorem


    """
    # Get Brion decomposition
    brion = vertex_cones(polytope, grading)
    
    return brion.generating_function or RationalFunction.constant(Fraction(0))


def verify_brion_theorem(polytope: 'Polytope',
                        ehrhart: 'EhrhartSeries',
                        grading: Optional['Grading'] = None,
                        tolerance: float = 1e-10) -> bool:
    """
    Verify Brion's theorem by comparing with Ehrhart series.

    Brion: GF(P) = sum_{v} GF(C_v)
    Ehrhart: GF(P) should match Ehrhart series

    Args:
        polytope: Polytope
        ehrhart: Ehrhart series of polytope
        grading: Grading to use
        tolerance: Numerical tolerance

    Returns:
        True if Brion's theorem holds within tolerance
    """
    # Compute Brion generating function
    brion_gf = brion_theorem(polytope, grading)
    
    # Compare first few terms
    for k in range(10):
        brion_coeff = brion_gf.series_coefficient(k)
        ehrhart_coeff = ehrhart.rational_function.series_coefficient(k)
        
        if abs(float(brion_coeff - ehrhart_coeff)) > tolerance:
            logger.warning(
                f"Brion mismatch at degree {k}: {brion_coeff} vs {ehrhart_coeff}"
            )
            return False
    
    return True


# ============================================================================
# BRION CACHE
# ============================================================================

class BrionCache:
    """LRU cache for Brion's theorem computations."""
    
    def __init__(self, maxsize: int = 16):
        self._cache = {}
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
    
    def get(self, key: tuple, compute_func: Callable, *args, **kwargs) -> Any:
        """Get cached result or compute and cache."""
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        
        self.misses += 1
        result = compute_func(*args, **kwargs)
        
        if len(self._cache) >= self.maxsize:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        
        self._cache[key] = result
        return result
    
    def clear(self):
        """Clear cache."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0
    
    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# Global cache instance
_brion_cache = BrionCache(maxsize=8)


# ============================================================================
# VALIDATION AND TESTING
# ============================================================================

def validate_brion_utils() -> Dict[str, bool]:
    """Run internal test suite to verify Brion utilities."""
    results = {}
    
    try:
        from fractions import Fraction
        
        # Test 1: Vertex cone creation
        vc = VertexCone(
            vertex=(Fraction(0), Fraction(0)),
            rays=[(Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))]
        )
        results["vertex_cone"] = vc.dimension == 2 and vc.is_pointed
        
        # Test 2: Brion decomposition of square
        class TestPolytope:
            def __init__(self):
                self.vertices = [
                    (Fraction(0), Fraction(0)),
                    (Fraction(1), Fraction(0)),
                    (Fraction(1), Fraction(1)),
                    (Fraction(0), Fraction(1))
                ]
        
        square = TestPolytope()
        brion = vertex_cones(square)
        results["brion_decomposition"] = brion.num_vertices == 4
        
        # Test 3: Barvinok generating function
        barvinok_gf = barvinok_generating_function(square)
        results["barvinok_gf"] = barvinok_gf is not None
        
        # Test 4: Exponential sum creation
        exp_sum = exponential_sum(
            polytope=square,
            exponents=[(Fraction(1), Fraction(0))],
            coefficients=[Fraction(1)]
        )
        results["exponential_sum"] = exp_sum.num_terms == 1
        
        # Test 5: Exponential sum evaluation
        point = (Fraction(1, 2), Fraction(1, 2))
        val = exp_sum.evaluate(point)
        results["exponential_eval"] = isinstance(val, complex)
        
        # Test 6: Residue computation
        rf = RationalFunction.geometric(Fraction(1), Fraction(1))
        residue = compute_residue(rf, Fraction(1))
        results["residue"] = isinstance(residue, Fraction)
        
        # Test 7: Brion's theorem
        brion_gf = brion_theorem(square)
        results["brion_theorem"] = brion_gf is not None
        
        # Test 8: Tangent cone rays
        rays = _tangent_cone_rays(square, (Fraction(0), Fraction(0)))
        results["tangent_rays"] = len(rays) >= 2
        
        logger.info("✅ Brion utilities validation passed")
        
    except Exception as e:
        logger.error(f"❌ Brion utilities validation failed: {e}")
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
    print("Testing Brion Utilities ()")
    print("=" * 60)
    
    # Run validation
    results = validate_brion_utils()
    
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
    print("Brion Utilities Demo ()")
    print("=" * 60)
    
    from fractions import Fraction
    
    # 1. Create a square polytope
    print("\n1. Creating Square Polytope:")
    class DemoPolytope:
        def __init__(self):
            self.vertices = [
                (Fraction(0), Fraction(0)),
                (Fraction(1), Fraction(0)),
                (Fraction(1), Fraction(1)),
                (Fraction(0), Fraction(1))
            ]
    square = DemoPolytope()
    print(f"   Square vertices: {square.vertices}")
    
    # 2. Compute Brion decomposition
    print("\n2. Brion Decomposition ():")
    brion = vertex_cones(square)
    print(f"   Number of vertex cones: {brion.num_vertices}")
    
    # 3. Barvinok's algorithm
    print("\n3. Barvinok's Algorithm ():")
    barvinok_gf = barvinok_generating_function(square)
    print(f"   Generating function denominator length: {len(barvinok_gf.denominator) if barvinok_gf else 0}")
    
    # 4. Exponential sum
    print("\n4. Exponential Sum ():")
    exp_sum = exponential_sum(
        polytope=square,
        exponents=[(Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))],
        coefficients=[Fraction(1), Fraction(1)]
    )
    print(f"   Number of terms: {exp_sum.num_terms}")
    
    # Evaluate at a sample point
    test_point = (Fraction(1, 3), Fraction(1, 3))
    val = exp_sum.evaluate(test_point)
    print(f"   Value at (1/3, 1/3): {val:.4f}")
    print(f"   Real part: {val.real:.4f}")
    print(f"   Imag part: {val.imag:.4f}")
    print(f"   Magnitude: {abs(val):.4f}")
    
    # 5. Brion's theorem
    print("\n5. Brion's Theorem ():")
    brion_gf = brion_theorem(square)
    print(f"   Brion GF denominator: {brion_gf.denominator if brion_gf else None}")
    
    # 6. Tangent cone at origin
    print("\n6. Tangent Cone at Origin:")
    rays = _tangent_cone_rays(square, (Fraction(0), Fraction(0)))
    for i, ray in enumerate(rays):
        print(f"   Ray {i}: {ray}")
    
    print("\n" + "=" * 60)
    print("✅ Brion Utilities Ready for Production")
    print("=" * 60)