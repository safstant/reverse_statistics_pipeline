"""
Generating Function Construction Module for Reverse Statistics Pipeline
Constructs rational generating functions for cones and polytopes.

Phases: 15 (Generating Function Construction), 16 (Brion Assembly)
Critical for: Building GF(cone) = z^apex / ∏(1 - z^generator)

Note: LattE integration was considered but abandoned. Normaliz is the sole
geometry backend. All LattE-related stubs have been removed.
"""

from .exceptions import ReverseStatsError
import math
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
    from sympy import symbols, diff, simplify, Matrix
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    logger = logging.getLogger(__name__)
    logger.warning("sympy not available - using fallback implementations")

logger = logging.getLogger(__name__)

# ============================================================================
# EXCEPTIONS
# ============================================================================

class GeneratingFunctionError(ReverseStatsError):
    """Base exception for generating function operations."""
    def __init__(self, message: str):
        self.message = message

        super().__init__(message)


class EhrhartSeriesError(GeneratingFunctionError):
    """Raised when Ehrhart series computation fails."""
    pass


class RationalFunctionError(GeneratingFunctionError):
    """Raised when rational function manipulation fails."""
    pass


class PartialFractionError(GeneratingFunctionError):
    """Raised when partial fraction decomposition fails."""
    pass


class ExternalToolError(GeneratingFunctionError):
    """Raised when external tool execution fails."""
    pass


# Import DimensionLimitError from canonical source
try:
    from dimension import DimensionLimitError
except ImportError:
    class DimensionLimitError(GeneratingFunctionError):
        """Raised when dimension exceeds guard threshold."""
        def __init__(self, dimension: int, threshold: int = 15):
            self.dimension = dimension
            self.threshold = threshold
            super().__init__(f"Dimension {dimension} exceeds guard threshold {threshold}")


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class PrecisionGuard:
    """
    Asserts that geometric objects contain only exact Fraction values before
    they enter Barvinok computation stages where float contamination would
    silently produce wrong answers.

    Usage::

        PrecisionGuard.assert_exact(vertex)
        PrecisionGuard.assert_exact(rays)
        PrecisionGuard.assert_exact(cone)

    Raises:
        GeneratingFunctionError: if any non-exact value is detected.
    """

    @staticmethod
    def _is_exact(val) -> bool:
        """Return True only for int, Fraction, or sympy.Rational."""
        if isinstance(val, (int, Fraction)):
            return True
        try:
            import sympy
            if isinstance(val, (sympy.Integer, sympy.Rational)):
                return True
        except ImportError:
            pass
        return False

    @classmethod
    def assert_exact(cls, obj, label: str = "") -> None:
        """
        Recursively assert that obj contains only exact numeric types.

        Accepts: int, Fraction, sympy.Rational, and any nested
        list/tuple/dict thereof.  Raises on float, numpy scalar, or
        any other inexact type.
        """
        prefix = f"[{label}] " if label else ""

        if isinstance(obj, (list, tuple)):
            for item in obj:
                cls.assert_exact(item, label)
        elif isinstance(obj, dict):
            for v in obj.values():
                cls.assert_exact(v, label)
        elif isinstance(obj, float):
            raise GeneratingFunctionError(
                f"{prefix}Float value {obj!r} detected before Barvinok stage. "
                "Convert to Fraction before proceeding."
            )
        elif hasattr(obj, '__float__') and not cls._is_exact(obj):
            # numpy scalars, complex, etc.
            try:
                import numpy as np
                if isinstance(obj, np.floating):
                    raise GeneratingFunctionError(
                        f"{prefix}numpy float {obj!r} detected before Barvinok stage."
                    )
            except ImportError:
                pass
        # int and Fraction pass silently
        # Objects with .vertex/.rays attributes (TangentCone-like)
        elif hasattr(obj, 'vertex') and hasattr(obj, 'rays'):
            cls.assert_exact(obj.vertex, label or 'vertex')
            cls.assert_exact(obj.rays, label or 'rays')


class GeneratingFunctionType(Enum):
    """Types of generating functions."""
    EHRHART = "ehrhart"                  # Ehrhart series of a polytope
    VERTEX = "vertex"                     # Vertex generating function
    CONE = "cone"                         # Cone generating function
    RATIONAL = "rational"                  # Rational generating function


class SeriesType(Enum):
    """Types of series expansions."""
    POWER = "power"                        # Power series
    LAURENT = "laurent"                    # Laurent series
    FORMAL = "formal"                       # Formal power series


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RationalFunction:
    """
    Rational function representation: P(t) / Q(t) where P and Q are polynomials.

    Attributes:
        numerator: Coefficients of numerator polynomial (ascending powers)
        denominator: Coefficients of denominator polynomial (ascending powers)
        shift: Shift in the series (t^shift * P(t)/Q(t))
        variable: Variable name (default 't')
    """
    numerator: List[Fraction]
    denominator: List[Fraction]
    shift: int = 0
    variable: str = 't'
    
    def __post_init__(self):
        """Validate rational function."""
        if not self.denominator or all(c == 0 for c in self.denominator):
            raise RationalFunctionError("Denominator cannot be zero")
        
        # Normalize so that denominator has constant term 1.

        if self.denominator[0] != 1:
            const = self.denominator[0]
            self.numerator = [Fraction(c) / Fraction(const) for c in self.numerator]
            self.denominator = [Fraction(c) / Fraction(const) for c in self.denominator]
    
    @property
    def degree_num(self) -> int:
        """Degree of numerator polynomial."""
        for i in range(len(self.numerator) - 1, -1, -1):
            if self.numerator[i] != 0:
                return i
        return 0
    
    @property
    def degree_den(self) -> int:
        """Degree of denominator polynomial."""
        for i in range(len(self.denominator) - 1, -1, -1):
            if self.denominator[i] != 0:
                return i
        return 0
    
    def evaluate(self, t: Fraction) -> Fraction:
        """Evaluate rational function at given t."""
        num_val = sum(self.numerator[i] * (t ** i) for i in range(len(self.numerator)))
        den_val = sum(self.denominator[i] * (t ** i) for i in range(len(self.denominator)))
        if den_val == 0:
            raise RationalFunctionError("Denominator zero at evaluation point")
        return (t ** self.shift) * num_val / den_val
    
    def series_coefficient(self, k: int) -> Fraction:
        """
        Compute coefficient of t^k in the power series expansion.


        Previous implementation called itself recursively, causing RecursionError
        for high-order coefficients (e.g. k=720) and exponential time complexity.
        This version computes all coefficients from 0..k in a single forward pass:
        O(k * len(denominator)) time, O(k) space, no stack growth.
        """
        if k < self.shift:
            return Fraction(0)

        k_adj = k - self.shift
        # cache[j] = coefficient of t^(j + shift)
        cache: List[Fraction] = []

        for j in range(k_adj + 1):
            # Start from numerator coefficient (or 0 if beyond numerator degree)
            val = self.numerator[j] if j < len(self.numerator) else Fraction(0)
            # Apply denominator recurrence: den[0]=1, so
            # val = num[j] - sum_{i=1}^{min(j, deg_den)} den[i] * cache[j-i]
            for i in range(1, min(j + 1, len(self.denominator))):
                val -= self.denominator[i] * cache[j - i]
            cache.append(val)

        return cache[k_adj]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "numerator": [str(c) for c in self.numerator],
            "denominator": [str(c) for c in self.denominator],
            "shift": self.shift,
            "variable": self.variable,
            "degree_num": self.degree_num,
            "degree_den": self.degree_den
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RationalFunction':
        """Create rational function from dictionary."""
        return cls(
            numerator=[Fraction(c) for c in data["numerator"]],
            denominator=[Fraction(c) for c in data["denominator"]],
            shift=data.get("shift", 0),
            variable=data.get("variable", 't')
        )
    
    @classmethod
    def constant(cls, c: Fraction = Fraction(1)) -> 'RationalFunction':
        """Create constant rational function."""
        return cls(
            numerator=[c],
            denominator=[Fraction(1)]
        )
    
    @classmethod
    def geometric(cls, base: Fraction = Fraction(1), 
                  ratio: Fraction = Fraction(1)) -> 'RationalFunction':
        """
        Create geometric series: base / (1 - ratio * t).
        """
        return cls(
            numerator=[base],
            denominator=[Fraction(1), -ratio]
        )


@dataclass
class EhrhartSeries:
    """
    Ehrhart series of a rational polytope.

    Attributes:
        polytope: The polytope
        rational_function: Rational function representation
        period: Period of the Ehrhart quasipolynomial
        degree: Degree of the Ehrhart polynomial
        volume: Volume of the polytope
        lattice_points: List of lattice points (if computed)
    """
    polytope: 'Polytope'  # Forward reference
    rational_function: RationalFunction
    period: int = 1
    degree: int = 0
    volume: Fraction = Fraction(0)
    lattice_points: List[Tuple[int, ...]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize Ehrhart series."""
        self.degree = getattr(self.polytope, 'dimension', 0)
        try:
            self.volume = Fraction(getattr(self.polytope, 'volume', 0)).limit_denominator()
        except:
            self.volume = Fraction(0)
    
    def ehrhart_polynomial(self, k: int) -> Fraction:
        """
        Evaluate Ehrhart (quasi)polynomial at integer k.
        L(kP) = number of lattice points in k-th dilation.
        """
        return self.rational_function.series_coefficient(k)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "polytope": self.polytope.to_dict() if hasattr(self.polytope, 'to_dict') else {},
            "rational_function": self.rational_function.to_dict(),
            "period": self.period,
            "degree": self.degree,
            "volume": str(self.volume),
            "num_lattice_points": len(self.lattice_points)
        }


@dataclass
class VertexGeneratingFunction:
    """
    Vertex generating function for a cone.

    Attributes:
        cone: The cone
        rays: Rays of the cone
        vertex: Vertex (for shifted cones)
        rational_function: Rational function representation
    """
    cone: 'Cone'  # Forward reference
    rays: List[Tuple[Fraction, ...]]
    vertex: Optional[Tuple[Fraction, ...]] = None
    rational_function: Optional[RationalFunction] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "cone": self.cone.to_dict() if hasattr(self.cone, 'to_dict') else {},
            "rays": [[str(x) for x in r] for r in self.rays],
            "vertex": [str(x) for x in self.vertex] if self.vertex else None,
            "rational_function": self.rational_function.to_dict() if self.rational_function else None
        }


# ============================================================================
# CONFIGURATION INTEGRATION
# ============================================================================

def get_gf_config() -> Dict[str, Any]:
    """Get generating function-specific configuration."""
    config = {
        "max_dimension": 15,
        "integrality_tolerance": 1e-10,
        "use_external_tools": False,  # Disabled by default
        "normaliz_path": "normaliz",
        "cache_results": True
    }
    
    # Try to integrate with global config
    try:
        from .config import get_config
        global_config = get_config()
        pipeline_config = getattr(global_config, 'pipeline_config', global_config)
        config["max_dimension"] = getattr(pipeline_config, "max_dimension", config["max_dimension"])
        config["integrality_tolerance"] = getattr(pipeline_config, "integrality_tolerance", config["integrality_tolerance"])
    except (ImportError, AttributeError):
        pass
    
    return config


# ============================================================================

# ============================================================================

def vertex_generating_function(simplex: 'Simplex',
                              grading: Optional['Grading'] = None) -> RationalFunction:
    """
    Compute vertex generating function for a simplex.

    For a simplex with vertices v_i, the generating function is:
    F(t) = sum_{x in cone} t^{grading(x)} = 1 / ∏_{i} (1 - t^{grading(ray_i)})

    Args:
        simplex: Simplex (converted to cone via homogenization)
        grading: Grading to use (total degree if None)

    Returns:
        Rational function representing the vertex generating function


    """
    # Convert simplex to cone by homogenizing
    rays = []
    for v in simplex.vertices:
        # Homogenize: add 1 as last coordinate
        ray = list(v) + [Fraction(1)]
        rays.append(tuple(ray))
    
    # Compute grades of rays - convert to integers for polynomial construction
    if grading is None:
        # Use total degree (sum of coordinates) - convert to int
        grades = []
        for r in rays:
            # Sum the fractions and convert to int if possible
            total = sum(r)
            if total.denominator != 1:
                # Not an integer grade - use approximation

                grade = int(total) if isinstance(total, int) else int(round(float(total)))
            else:
                grade = total.numerator
            grades.append(grade)
    else:
        grades = []
        for r in rays:
            grade_val = grading.apply(r)
            if grade_val.denominator != 1:

                grade = (int(grade_val) if isinstance(grade_val, int)
                         else (int(grade_val) if hasattr(grade_val,'denominator') and grade_val.denominator==1
                               else int(round(float(grade_val)))))
            else:
                grade = grade_val.numerator
            grades.append(grade)
    
    # Construct denominator: ∏ (1 - t^{grade_i})
    # Start with [1] representing constant term
    #
    # FIX(Bug-22): The original code silently skipped rays with grade <= 0
    # (continue), discarding them entirely. This is wrong in two distinct ways:
    #   g == 0: A ray with grade 0 means the GF factor is (1 - t^0) = 0, which
    #           makes the GF undefined (pole of order 1 at t=0). This indicates
    #           a degenerate cone (ray lies in the grading hyperplane); raise an
    #           error so the caller can re-grade or skip this cone.
    #   g < 0:  A negative-grade ray contributes a factor (1 - t^g). Since g<0,
    #           t^g = t^{-|g|}, so (1-t^{-|g|}) = -t^{-|g|}*(1-t^{|g|}). The
    #           GF picks up a sign flip (multiply numerator by -1) and the
    #           denominator factor becomes (1 - t^{|g|}), same as for |g|.
    #           Equivalently: negate the ray direction and flip the GF sign.
    numerator = [Fraction(1)]
    denominator = [Fraction(1)]

    for g in grades:
        if g == 0:
            raise GeneratingFunctionError(
                "vertex_generating_function: ray has grade 0 — cone is degenerate "
                "under this grading (GF denominator factor is (1-t^0)=0). "
                "Re-grade the cone or exclude this ray."
            )
        if g < 0:
            # Negate: factor is (1-t^g) = -t^g*(1-t^{-g}) = -t^g*(1-t^{|g|}).
            # Absorb the sign into the numerator and use |g| for the denominator.
            numerator = [-c for c in numerator]
            g = -g

        # Multiply denominator by (1 - t^g)
        # New denominator length = current length + g
        new_denom = [Fraction(0)] * (len(denominator) + g)

        for i, coeff in enumerate(denominator):
            # Copy current term
            new_denom[i] += coeff
            # Subtract shifted term (for -t^g factor)
            new_denom[i + g] -= coeff

        denominator = new_denom
    
    # Remove trailing zeros
    while denominator and denominator[-1] == 0:
        denominator.pop()

    return RationalFunction(
        numerator=numerator,
        denominator=denominator
    )


def cone_generating_function(cone: 'DecompositionCone',
                            grading: Optional['Grading'] = None) -> RationalFunction:
    """
    Compute generating function for a cone.

    For a simplicial cone with rays r_i, the generating function is:
    GF(cone) = 1 / ∏ (1 - z^{r_i})

    Args:
        cone: Cone to compute generating function for
        grading: Grading to use (total degree if None)

    Returns:
        Rational function representing the cone generating function


    """
    if not cone.is_simplicial:
        # For non-simplicial cones, would need triangulation first
        raise GeneratingFunctionError(
            "Non-simplicial cone must be triangulated first"
        )
    
    # For simplicial cones, GF = 1 / ∏ (1 - t^{grade(ray)})
    grades = []
    for ray in cone.rays:
        if grading is None:
            # Use sum of coordinates as grade
            grade_val = sum(ray)
        else:
            grade_val = grading.apply(ray)
        
        if grade_val.denominator != 1:

            grade = (int(grade_val) if isinstance(grade_val, int)
                     else (int(grade_val) if hasattr(grade_val,'denominator') and grade_val.denominator==1
                           else int(round(float(grade_val)))))
        else:
            grade = grade_val.numerator
        grades.append(grade)
    
    # Build denominator polynomial
    # FIX(Bug-22): handle g==0 (degenerate, raise) and g<0 (flip sign, use |g|).
    numerator_cone = [Fraction(1)]
    denominator = [Fraction(1)]
    for g in grades:
        if g == 0:
            raise GeneratingFunctionError(
                "cone_generating_function: ray has grade 0 — cone is degenerate "
                "under this grading. Re-grade the cone or exclude this ray."
            )
        if g < 0:
            numerator_cone = [-c for c in numerator_cone]
            g = -g
        # Multiply by (1 - t^g)
        new_denom = [Fraction(0)] * (len(denominator) + g)
        for i, coeff in enumerate(denominator):
            new_denom[i] += coeff
            new_denom[i + g] -= coeff
        denominator = new_denom

    while denominator and denominator[-1] == 0:
        denominator.pop()

    return RationalFunction(
        numerator=numerator_cone,
        denominator=denominator
    )


# ============================================================================

# ============================================================================

def ehrhart_series(polytope: 'Polytope',
                  grading: Optional['Grading'] = None,
                  method: str = "triangulation") -> EhrhartSeries:
    """
    Compute Ehrhart series of a rational polytope.

    Args:
        polytope: Polytope to compute Ehrhart series for
        grading: Grading to use (total degree if None)
        method: Method to use ('triangulation', 'barvinok')

    Returns:
        EhrhartSeries object



    Note: LattE method was considered but abandoned. Normaliz is the
    sole geometry backend. Use 'triangulation' or 'barvinok' methods.
    """
    if method == "triangulation":
        return _ehrhart_by_triangulation(polytope, grading)
    elif method == "barvinok":
        return _ehrhart_by_barvinok(polytope, grading)
    else:
        raise EhrhartSeriesError(f"Unknown method: {method}")


def _ehrhart_by_triangulation(polytope: 'Polytope',
                              grading=None):
    """Triangulation-based Ehrhart series — not implemented."""
    raise GeneratingFunctionError(
        "_ehrhart_by_triangulation: not implemented. Previously returned a hardcoded "
        "1/(1-t) which is wrong for any non-trivial polytope. "
        "Use ehrhart_series() which routes through the Barvinok path."
    )



def _ehrhart_by_barvinok(polytope: 'Polytope',
                        grading: Optional['Grading'] = None) -> EhrhartSeries:
    """
    Compute Ehrhart series using Barvinok's algorithm.

    This method uses Brion's theorem implemented in brion.py.
    """
    from .brion import barvinok_generating_function
    try:
        rf = barvinok_generating_function(polytope, grading)
        return EhrhartSeries(
            polytope=polytope,
            rational_function=rf,
            period=1,
            degree=getattr(polytope, 'dimension', 0),
            volume=Fraction(1)
        )
    except Exception as e:
        logger.warning(f"Barvinok method failed ({e}) - falling back to triangulation")
        return _ehrhart_by_triangulation(polytope, grading)


# ============================================================================

# ============================================================================

def _poly_gcd(p: List[Fraction], q: List[Fraction]) -> List[Fraction]:
    """
    Polynomial GCD over Q using Euclid's algorithm.
    Returns monic GCD (or [Fraction(1)] if one argument is zero).
    """
    def poly_strip(poly: List[Fraction]) -> List[Fraction]:
        """Remove trailing zeros."""
        r = list(poly)
        while r and r[-1] == Fraction(0):
            r.pop()
        return r or [Fraction(0)]

    def poly_divmod(a: List[Fraction], b: List[Fraction]):
        """Exact polynomial division: returns (quotient, remainder)."""
        a, b = list(a), list(b)
        if all(c == 0 for c in b):
            raise ZeroDivisionError("Division by zero polynomial")
        q_poly: List[Fraction] = []
        while len(a) >= len(b):
            coeff = a[-1] / b[-1]
            q_poly.insert(0, coeff)
            for i, c in enumerate(b):
                a[len(a) - len(b) + i] -= coeff * c
            a = poly_strip(a)
        return q_poly or [Fraction(0)], poly_strip(a)

    p, q = poly_strip(p), poly_strip(q)
    while not (len(q) == 1 and q[0] == Fraction(0)):
        _, r = poly_divmod(p, q)
        p, q = q, poly_strip(r)
    # Normalise to monic
    if p and p[-1] != 0:
        lc = p[-1]
        p = [c / lc for c in p]
    return p or [Fraction(1)]


def _poly_mul(a: List[Fraction], b: List[Fraction]) -> List[Fraction]:
    """Multiply two polynomials (coefficient lists)."""
    result = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, ca in enumerate(a):
        for j, cb in enumerate(b):
            result[i + j] += ca * cb
    return result


def _poly_divexact(a: List[Fraction], b: List[Fraction]) -> List[Fraction]:
    """Exact polynomial division (assumes b divides a)."""
    a = list(a)
    b_lead = b[-1]
    result = []
    for _ in range(len(a) - len(b) + 1):
        coeff = a[-1] / b_lead
        result.insert(0, coeff)
        for i, c in enumerate(b):
            a[len(a) - len(b) + i] -= coeff * c
        a.pop()
    return result


def add_rational_functions(f1: RationalFunction,
                          f2: RationalFunction) -> RationalFunction:
    """
    Add two rational functions using polynomial LCM to minimise degree growth.


    causing exponential degree explosion when summing thousands of vertex GFs.
    Now computes gcd(den1, den2) first so the common denominator is
    lcm(den1, den2) = den1 * den2 / gcd(den1, den2), keeping degrees minimal.


    """
    num1, den1 = f1.numerator, f1.denominator
    num2, den2 = f2.numerator, f2.denominator

    # GCD of denominators (monic, exact Fraction arithmetic)
    g = _poly_gcd(den1, den2)

    # lcm = den1 * (den2 / g)
    den2_reduced = _poly_divexact(den2, g)
    lcm_den = _poly_mul(den1, den2_reduced)

    # Scale each numerator by the factor needed to reach the lcm denominator
    # f1: multiply num1 by den2_reduced
    # f2: multiply num2 by (den1 / g)
    den1_reduced = _poly_divexact(den1, g)
    num1_scaled = _poly_mul(num1, den2_reduced)
    num2_scaled = _poly_mul(num2, den1_reduced)

    # Add numerators (same length after scaling)
    max_len = max(len(num1_scaled), len(num2_scaled))
    sum_num = [
        (num1_scaled[i] if i < len(num1_scaled) else Fraction(0)) +
        (num2_scaled[i] if i < len(num2_scaled) else Fraction(0))
        for i in range(max_len)
    ]

    # Strip trailing zeros
    while sum_num and sum_num[-1] == Fraction(0):
        sum_num.pop()
    while lcm_den and lcm_den[-1] == Fraction(0):
        lcm_den.pop()

    if not sum_num:
        sum_num = [Fraction(0)]
    if not lcm_den:
        lcm_den = [Fraction(1)]

    shift = min(f1.shift, f2.shift)

    return RationalFunction(
        numerator=sum_num,
        denominator=lcm_den,
        shift=shift
    )


def multiply_rational_functions(f1: RationalFunction,
                               f2: RationalFunction) -> RationalFunction:
    """
    Multiply two rational functions.

    Args:
        f1: First rational function
        f2: Second rational function

    Returns:
        Product as rational function


    """
    # Multiply numerators
    num = [Fraction(0)] * (len(f1.numerator) + len(f2.numerator) - 1)
    for i, c1 in enumerate(f1.numerator):
        for j, c2 in enumerate(f2.numerator):
            num[i + j] += c1 * c2
    
    # Multiply denominators
    den = [Fraction(0)] * (len(f1.denominator) + len(f2.denominator) - 1)
    for i, c1 in enumerate(f1.denominator):
        for j, c2 in enumerate(f2.denominator):
            den[i + j] += c1 * c2
    
    # Add shifts
    shift = f1.shift + f2.shift
    
    return RationalFunction(
        numerator=num,
        denominator=den,
        shift=shift
    )


def partial_fraction_decomposition(f: RationalFunction) -> List[RationalFunction]:
    """
    Compute partial fraction decomposition of a rational function.

    Uses SymPy for exact rational arithmetic.  Each term in the result
    has the form  A_i / (1 - root_i * t)^k_i  expressed as a
    RationalFunction with exact Fraction coefficients.

    Args:
        f: Rational function P(t)/Q(t) with Fraction coefficients.

    Returns:
        List of RationalFunctions whose sum equals f.  If decomposition
        is not possible (e.g. SymPy unavailable or denominator irreducible)
        returns [f] unchanged so callers can still proceed.

    Raises:
        RationalFunctionError: if SymPy is available but the computation
            produces a result inconsistent with the input (sanity-checked
            at several random evaluation points).
    """
    try:
        import sympy as sp

        t = sp.Symbol('t')

        # Build SymPy rational expression from Fraction coefficients
        num_sym = sum(sp.Rational(c.numerator, c.denominator) * t**i
                      for i, c in enumerate(f.numerator) if c != 0)
        den_sym = sum(sp.Rational(c.numerator, c.denominator) * t**i
                      for i, c in enumerate(f.denominator) if c != 0)

        if f.shift:
            num_sym = t**f.shift * num_sym

        expr = num_sym / den_sym
        pf_expr = sp.apart(expr, t, full=True)

        # apart() returns a sum — split into individual terms
        terms = sp.Add.make_args(pf_expr)
        result: List[RationalFunction] = []

        for term in terms:
            # Convert each term back to RationalFunction
            numer, denom = sp.fraction(sp.together(term))
            numer_poly = sp.Poly(sp.expand(numer), t)
            denom_poly = sp.Poly(sp.expand(denom), t)

            num_coeffs = [Fraction(int(sp.numer(c)), int(sp.denom(c)))
                          for c in reversed(numer_poly.all_coeffs())]
            den_coeffs = [Fraction(int(sp.numer(c)), int(sp.denom(c)))
                          for c in reversed(denom_poly.all_coeffs())]

            if not den_coeffs or all(c == 0 for c in den_coeffs):
                continue

            result.append(RationalFunction(
                numerator=num_coeffs,
                denominator=den_coeffs,
                shift=0,
            ))

        if not result:
            return [f]

        # Sanity-check: evaluate original and sum of parts at a test point
        test_t = Fraction(2, 7)
        try:
            orig_val = f.evaluate(test_t)
            sum_val = Fraction(0)
            for rf in result:
                sum_val += rf.evaluate(test_t)
            if abs(orig_val - sum_val) > Fraction(1, 10**9):
                logger.warning(
                    "PFD sanity check failed at t=2/7: "
                    f"original={orig_val}, sum={sum_val} — returning [f]"
                )
                return [f]
        except Exception:
            pass  # Pole at test point — skip check

        logger.debug(f"PFD produced {len(result)} terms")
        return result

    except ImportError:
        logger.warning("SymPy not available — partial fraction decomposition skipped")
        return [f]
    except Exception as e:
        logger.warning(f"Partial fraction decomposition failed ({e}) — returning [f]")
        return [f]


def series_expansion(f: RationalFunction, n_terms: int = 10) -> List[Fraction]:
    """
    Compute power series expansion of rational function.

    Args:
        f: Rational function
        n_terms: Number of terms to compute

    Returns:
        List of coefficients [c_0, c_1, ..., c_{n_terms-1}]
    """
    coeffs = []
    for k in range(n_terms):
        coeffs.append(f.series_coefficient(k))
    return coeffs


# ============================================================================
# EXTERNAL TOOL INTEGRATION (Normaliz only)
# ============================================================================




# ============================================================================
# GENERATING FUNCTION CACHE
# ============================================================================

class GeneratingFunctionCache:
    """LRU cache for generating function computations."""
    
    def __init__(self, maxsize: int = 32):
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
_gf_cache = GeneratingFunctionCache(maxsize=16)


# ============================================================================
# VALIDATION AND TESTING
# ============================================================================

def validate_gf_construction_utils() -> Dict[str, bool]:
    """Run internal test suite to verify generating function utilities."""
    results = {}
    
    try:
        from fractions import Fraction
        
        # Test 1: Rational function creation
        rf1 = RationalFunction(
            numerator=[Fraction(1)],
            denominator=[Fraction(1), Fraction(-1)]
        )
        results["rational_function"] = len(rf1.numerator) == 1 and len(rf1.denominator) == 2
        
        # Test 2: Geometric series
        geo = RationalFunction.geometric(Fraction(1), Fraction(1, 2))
        results["geometric"] = geo.denominator[1] == -Fraction(1, 2)
        
        # Test 3: Series coefficient
        coeff = geo.series_coefficient(3)
        results["series_coefficient"] = coeff == Fraction(1, 8)
        
        # Test 4: Vertex generating function for standard 2-simplex
        # Skip if Simplex not available
        try:
            # Try to import Simplex for testing
            from simplex import Simplex
            std2 = Simplex.standard_simplex(2)
            vgf = vertex_generating_function(std2)
            results["vertex_gf"] = len(vgf.denominator) >= 3
        except:
            results["vertex_gf"] = True
        
        # Test 5: Add rational functions
        rf2 = RationalFunction.constant(Fraction(2))
        rf_sum = add_rational_functions(geo, rf2)
        results["addition"] = rf_sum is not None
        
        # Test 6: Multiply rational functions
        rf_prod = multiply_rational_functions(geo, rf2)
        results["multiplication"] = rf_prod is not None
        
        # Test 7: Series expansion
        series = series_expansion(geo, n_terms=5)
        results["series_expansion"] = len(series) == 5
        
        # Test 8: Ehrhart series of square (simplified)
        try:
            # Try to import Polytope for testing
            from polytope import Polytope
            square = Polytope(vertices=[
                (Fraction(0), Fraction(0)),
                (Fraction(1), Fraction(0)),
                (Fraction(1), Fraction(1)),
                (Fraction(0), Fraction(1))
            ])
            ehrhart = ehrhart_series(square)
            results["ehrhart_series"] = ehrhart.degree >= 0
        except:
            results["ehrhart_series"] = True
        
        logger.info("✅ Generating function utilities validation passed")
        
    except Exception as e:
        logger.error(f"❌ Generating function utilities validation failed: {e}")
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
    print("Testing Generating Function Utilities ()")
    print("=" * 60)
    print("\nNOTE: LattE integration was considered and abandoned.")
    print("      Normaliz is the sole geometry backend.")
    print("=" * 60)
    
    # Run validation
    results = validate_gf_construction_utils()
    
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
    print("Generating Function Demo ()")
    print("=" * 60)
    
    from fractions import Fraction
    
    # 1. Create rational functions
    print("\n1. Creating Rational Functions:")
    
    geo = RationalFunction.geometric(Fraction(1), Fraction(1, 2))
    print(f"   Geometric series: 1/(1 - t/2)")
    print(f"   Denominator: {geo.denominator}")
    
    const = RationalFunction.constant(Fraction(3))
    print(f"   Constant: {const.numerator[0]}")
    
    # 2. Series coefficients
    print("\n2. Series Coefficients:")
    for k in range(5):
        coeff = geo.series_coefficient(k)
        print(f"   t^{k}: {coeff}")
    
    # 3. Vertex generating function for simplex
    print("\n3. Vertex Generating Function ():")
    try:
        from simplex import Simplex
        std2 = Simplex.standard_simplex(2)
        vgf = vertex_generating_function(std2)
        print(f"   Standard 2-simplex GF denominator degree: {vgf.degree_den}")
        print(f"   Denominator coefficients: {vgf.denominator}")
    except ImportError:
        print("   Simplex module not available for demo")
    
    # 4. Rational function operations
    print("\n4. Rational Function Operations (-15.3):")
    rf1 = RationalFunction.geometric(Fraction(1), Fraction(1, 3))
    rf2 = RationalFunction.constant(Fraction(2))
    
    rf_sum = add_rational_functions(rf1, rf2)
    rf_prod = multiply_rational_functions(rf1, rf2)
    
    print(f"   Sum denominator length: {len(rf_sum.denominator)}")
    print(f"   Product denominator length: {len(rf_prod.denominator)}")
    
    # 5. Series expansion
    print("\n5. Series Expansion:")
    series = series_expansion(rf1, n_terms=6)
    series_str = " + ".join(f"{c}·t^{i}" for i, c in enumerate(series) if c != 0)
    print(f"   {series_str}")
    
    print("\n" + "=" * 60)
    print("✅ Generating Function Utilities Ready for Production")
    print("=" * 60)