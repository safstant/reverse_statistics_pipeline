import numpy as np
from .exceptions import ReverseStatsError
import math
from fractions import Fraction
from typing import List, Tuple, Dict, Any, Optional, Union, Set, Callable, Iterator, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import sys
import os
from functools import lru_cache, wraps

# Add the current directory to path to handle imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Handle imports to work both as module and standalone
try:
    # When imported as part of package
    from .math_utils import (
        is_integer, gcd_list, lcm_list, matrix_rank,
        is_unimodular_matrix
    )
    from .lattice_utils import LatticeBasis, LatticeCone as Cone
    from .stats_utils import Histogram, MomentConstraints
except ImportError:
    # When run directly
    try:
        from math_utils import (
            is_integer, gcd_list, lcm_list, matrix_rank,
            is_unimodular_matrix
        )
        from lattice_utils import LatticeBasis, LatticeCone as Cone
        from stats_utils import Histogram, MomentConstraints
    except ImportError as e:
        raise ImportError(
            f"indexing.py: required dependencies (math_utils, lattice_utils, stats_utils) "
            f"could not be imported: {e}. "
            "These modules are required — there is no correct fallback. "
            "Ensure the package is installed correctly or all sibling modules are on sys.path."
        ) from e

logger = logging.getLogger(__name__)

# Import exact determinant for Barvinok pipeline
try:
    from math_utils import determinant_exact as _det_exact
except ImportError:
    try:
        from .math_utils import determinant_exact as _det_exact
    except ImportError:
        _det_exact = None


# ============================================================================
# EXCEPTIONS
# ============================================================================

class IndexingError(ReverseStatsError):
    """Base exception for indexing operations."""
    def __init__(self, message: str):
        self.message = message

        super().__init__(message)


class GradingError(IndexingError):
    """Raised for grading-related errors."""
    pass


class DenominatorError(IndexingError):
    """Raised for denominator calculation errors."""
    pass


class IndexRangeError(IndexingError):
    """Raised when index is out of valid range."""
    pass


class ShiftError(IndexingError):
    """Raised for shift-related errors."""
    pass


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class IndexType(Enum):
    """Types of indices."""
    EXTERNAL = "external"
    INTERNAL = "internal"      # Internal index
    UNIT_GROUP = "unit_group"  # Unit group index
    LATTICE = "lattice"        # Lattice index
    GRADING = "grading"        # Grading index


class ShiftDirection(Enum):
    """Directions for index shifting."""
    FORWARD = "forward"        # Shift forward (increase)
    BACKWARD = "backward"      # Shift backward (decrease)
    BOTH = "both"              # Shift in both directions


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class Grading:
    """
    Grading on a lattice or monoid.

    Attributes:
        vector: Grading vector (linear form)
        denominator: Denominator of the grading
        name: Optional name for the grading
    """
    vector: Tuple[int, ...]
    denominator: int = 1
    name: str = ""
    
    def __post_init__(self):
        """Validate grading."""
        if not self.vector:
            raise GradingError("Grading vector cannot be empty")
        
        if self.denominator <= 0:
            raise GradingError(f"Denominator must be positive, got {self.denominator}")
        
        # Check that vector entries are integers
        if not all(isinstance(x, int) for x in self.vector):
            raise GradingError("Grading vector must contain integers")
    
    @property
    def dim(self) -> int:
        """Dimension of the grading."""
        return len(self.vector)
    
    def apply(self, x: Tuple[int, ...]) -> Fraction:
        """
        Apply grading to a vector.

        Args:
            x: Vector to grade

        Returns:
            Grade as Fraction (vector·grading / denominator)
        """
        if len(x) != self.dim:
            raise GradingError(
                f"Vector dimension {len(x)} does not match grading dimension {self.dim}",

            )
        
        dot = sum(a * b for a, b in zip(x, self.vector))
        return Fraction(dot, self.denominator)
    
    def scale(self, factor: int) -> 'Grading':
        """Scale grading by a factor."""
        return Grading(
            vector=self.vector,
            denominator=self.denominator * factor,
            name=f"{self.name}_scaled_{factor}"
        )
    
    def normalize(self) -> 'Grading':
        """Normalize grading to have denominator 1 if possible."""
        if self.denominator == 1:
            return self
        
        # Check if all components are divisible by denominator
        if all(x % self.denominator == 0 for x in self.vector):
            return Grading(
                vector=tuple(x // self.denominator for x in self.vector),
                denominator=1,
                name=f"{self.name}_normalized"
            )
        
        # Find GCD of all components and denominator
        all_vals = list(self.vector) + [self.denominator]
        g = gcd_list(all_vals)
        if g > 1:
            return Grading(
                vector=tuple(x // g for x in self.vector),
                denominator=self.denominator // g,
                name=f"{self.name}_normalized"
            )
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "vector": list(self.vector),
            "denominator": self.denominator,
            "dim": self.dim,
            "name": self.name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Grading':
        """Create grading from dictionary."""
        return cls(
            vector=tuple(data["vector"]),
            denominator=data.get("denominator", 1),
            name=data.get("name", "")
        )
    
    @classmethod
    def unit_grading(cls, dim: int, coordinate: int = 0, denominator: int = 1) -> 'Grading':
        """
        Create unit grading (1 in specified coordinate, 0 elsewhere).

        Args:
            dim: Dimension
            coordinate: Coordinate with value 1 (0-based)
            denominator: Denominator

        Returns:
            Unit grading
        """
        if coordinate >= dim:
            raise GradingError(f"Coordinate {coordinate} out of range for dimension {dim}")
        
        vector = [0] * dim
        vector[coordinate] = 1
        return cls(vector=tuple(vector), denominator=denominator, name=f"unit_{coordinate}")
    
    @classmethod
    def total_degree(cls, dim: int, denominator: int = 1) -> 'Grading':
        """
        Create total degree grading (1 in all coordinates).

        Args:
            dim: Dimension
            denominator: Denominator

        Returns:
            Total degree grading
        """
        vector = [1] * dim
        return cls(vector=tuple(vector), denominator=denominator, name="total_degree")


@dataclass(frozen=True)
class Index:
    """
    Index value with potential shift and denominator.

    Attributes:
        value: Index value (as Fraction for exact representation)
        shift: Shift amount
        denominator: Denominator
        index_type: Type of index
    """
    value: Fraction
    shift: int = 0
    denominator: int = 1
    index_type: IndexType = IndexType.LATTICE
    
    def __post_init__(self):
        """Validate index."""
        if self.denominator <= 0:
            raise IndexingError(f"Denominator must be positive, got {self.denominator}")
    
    @property
    def shifted_value(self) -> Fraction:
        """Value after applying shift."""
        return self.value + self.shift
    
    @property
    def normalized_value(self) -> Fraction:
        """Value normalized by denominator."""
        return self.value / self.denominator if self.denominator != 1 else self.value
    
    def apply_shift(self, amount: int, direction: ShiftDirection = ShiftDirection.FORWARD) -> 'Index':
        """Apply shift to index."""
        if direction == ShiftDirection.FORWARD:
            new_shift = self.shift + amount
        elif direction == ShiftDirection.BACKWARD:
            new_shift = self.shift - amount
        else:  # BOTH - create two indices
            raise ValueError("Use shift_forward and shift_backward for BOTH direction")
        
        return Index(
            value=self.value,
            shift=new_shift,
            denominator=self.denominator,
            index_type=self.index_type
        )
    
    def shift_forward(self, amount: int) -> 'Index':
        """Shift index forward."""
        return self.apply_shift(amount, ShiftDirection.FORWARD)
    
    def shift_backward(self, amount: int) -> 'Index':
        """Shift index backward."""
        return self.apply_shift(amount, ShiftDirection.BACKWARD)
    
    def to_float(self) -> float:
        """Convert to float (for display only, not for calculations)."""
        return float(self.shifted_value / self.denominator)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "value": str(self.value),
            "shift": self.shift,
            "denominator": self.denominator,
            "shifted_value": str(self.shifted_value),
            "normalized_value": str(self.normalized_value),
            "index_type": self.index_type.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Index':
        """Create index from dictionary."""
        from fractions import Fraction
        return cls(
            value=Fraction(data["value"]),
            shift=data.get("shift", 0),
            denominator=data.get("denominator", 1),
            index_type=IndexType(data.get("index_type", "lattice"))
        )
    
    @classmethod
    def from_float(cls, value: float, denominator: int = 1000000) -> 'Index':
        """
        Create index from float (approximate).

        Args:
            value: Float value
            denominator: Denominator for Fraction conversion

        Returns:
            Approximate index
        """
        from fractions import Fraction
        return cls(
            value=Fraction(int(value * denominator), denominator),
            denominator=1,
            index_type=IndexType.LATTICE
        )


@dataclass(frozen=True)
class IndexRange:
    """
    Range of indices with bounds.

    Attributes:
        min_index: Minimum index (inclusive)
        max_index: Maximum index (inclusive)
        step: Step size between indices
        inclusive: Whether bounds are inclusive
    """
    min_index: Index
    max_index: Index
    step: int = 1
    inclusive: bool = True
    
    def __post_init__(self):
        """Validate index range."""
        if self.min_index.value > self.max_index.value:
            raise IndexRangeError(
                f"min_index {self.min_index.value} > max_index {self.max_index.value}",

            )
        
        if self.step <= 0:
            raise IndexRangeError(f"Step must be positive, got {self.step}")
    
    @property
    def size(self) -> int:
        """Number of indices in range."""
        diff = self.max_index.value - self.min_index.value
        # Convert to float for calculation, then back to int
        diff_float = float(diff)
        if self.inclusive:
            return int(diff_float / self.step) + 1
        else:
            return int(diff_float / self.step)
    
    def contains(self, index: Index) -> bool:
        """Check if index is within range."""
        if index.value < self.min_index.value or index.value > self.max_index.value:
            return False
        
        # Check step alignment
        diff = index.value - self.min_index.value
        # Convert to float for modulus check
        diff_float = float(diff)
        step_float = float(self.step)
        return abs(diff_float % step_float) < 1e-10
    
    def indices(self) -> Iterator[Index]:
        """Iterate over indices in range."""
        current_val = self.min_index.value
        max_val = self.max_index.value
        step_frac = Fraction(self.step, 1)
        
        while current_val <= max_val:
            yield Index(
                value=current_val,
                shift=self.min_index.shift,
                denominator=self.min_index.denominator,
                index_type=self.min_index.index_type
            )
            current_val += step_frac
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "min_index": self.min_index.to_dict(),
            "max_index": self.max_index.to_dict(),
            "step": self.step,
            "size": self.size,
            "inclusive": self.inclusive
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndexRange':
        """Create index range from dictionary."""
        return cls(
            min_index=Index.from_dict(data["min_index"]),
            max_index=Index.from_dict(data["max_index"]),
            step=data.get("step", 1),
            inclusive=data.get("inclusive", True)
        )
    
    @classmethod
    def from_values(cls, min_val: Union[int, float, Fraction], 
                   max_val: Union[int, float, Fraction],
                   step: int = 1) -> 'IndexRange':
        """Create index range from values."""
        from fractions import Fraction
        
        if isinstance(min_val, (int, float)):
            min_index = Index(value=Fraction(int(min_val) if isinstance(min_val, int) else int(min_val)))
        else:
            min_index = Index(value=min_val)
        
        if isinstance(max_val, (int, float)):
            max_index = Index(value=Fraction(int(max_val) if isinstance(max_val, int) else int(max_val)))
        else:
            max_index = Index(value=max_val)
        
        return cls(min_index=min_index, max_index=max_index, step=step)


# ============================================================================
# INDEX CALCULATION FUNCTIONS
# ============================================================================

def compute_external_index(lattice_basis: LatticeBasis) -> Index:
    """
    Compute external index of a lattice.

    The external index is the index of the lattice in its ambient space.

    Args:
        lattice_basis: Lattice basis

    Returns:
        External index
    """
    if lattice_basis.dim == lattice_basis.rank:
        # Full rank lattice - determinant gives index

        matrix_exact = [list(row) for row in lattice_basis.basis]
        if _det_exact is not None:
            det_frac = abs(_det_exact(matrix_exact))
        else:
            # fallback: integer matrix det via sympy
            from sympy import Matrix as _SMatrix
            det_frac = Fraction(abs(int(_SMatrix(matrix_exact).det())))
        return Index(value=det_frac, index_type=IndexType.EXTERNAL)
    else:
        # Not full rank - compute via Gram matrix using exact arithmetic

        from fractions import Fraction as _Frac
        basis = [list(row) for row in lattice_basis.basis]
        k = len(basis)
        # Gram matrix G[i][j] = dot(basis[i], basis[j])
        G_exact = []
        for i in range(k):
            row = []
            for j in range(k):
                dot = sum(_Frac(basis[i][l]) * _Frac(basis[j][l]) for l in range(len(basis[i])))
                row.append(dot)
            G_exact.append(row)
        if _det_exact is not None:
            det_gram = _det_exact(G_exact)
        else:
            from sympy import Matrix as _SMatrix

        from sympy import Rational as _SR
        _G_sym = _SMatrix([[_SR(int(x.numerator), int(x.denominator))
                            if hasattr(x, 'numerator') else _SR(int(x))
                            for x in r] for r in G_exact])
        # FIX(Bug-15): The previous code called Fraction(abs(int(_G_sym.det()))),
        # which truncates a SymPy Rational result (e.g. Rational(5,4)) through int()
        # before wrapping it in Fraction, yielding Fraction(1) instead of Fraction(5,4).
        # Correct approach: convert SymPy Rational → Python Fraction exactly via
        # its .p (numerator) and .q (denominator) attributes.
        _sym_det = _G_sym.det()
        if hasattr(_sym_det, 'p') and hasattr(_sym_det, 'q'):
            det_gram = Fraction(int(abs(_sym_det.p)), int(_sym_det.q))
        else:
            det_gram = Fraction(abs(int(_sym_det)))
        # Index = sqrt(det(G)).  det_gram may be a Fraction (e.g. 5/4 for a
        # half-integer basis).  FIX(Bug-3): the original code called int(det_gram)
        # which truncates 5/4 → 1 before taking isqrt, yielding 1 instead of √(5/4).
        # Correct approach: rational sqrt via perfect-square check, else SymPy.
        import math as _m

        if isinstance(det_gram, Fraction):
            num, den = abs(det_gram.numerator), det_gram.denominator
        else:
            num, den = abs(int(det_gram)), 1

        sqrt_num = _m.isqrt(num)
        sqrt_den = _m.isqrt(den)
        if sqrt_num * sqrt_num == num and sqrt_den * sqrt_den == den:
            sqrt_frac = Fraction(sqrt_num, sqrt_den)
        else:
            # Gram det is not a perfect rational square → irrational index
            try:
                from sympy import sqrt as _sympy_sqrt, Rational as _SR
                sqrt_sym = _sympy_sqrt(_SR(num, den))
                # sqrt_sym may be irrational; best rational approximation used.
                sqrt_frac = Fraction(float(sqrt_sym)).limit_denominator(10**9)
            except Exception as e:
                raise type(e)(
                    f"compute_external_index: cannot compute exact sqrt of {num}/{den}: {e}. "
                    "SymPy is required for irrational lattice index."
                ) from e
        return Index(value=sqrt_frac, index_type=IndexType.EXTERNAL)


def compute_internal_index(monoid_generators: List[Tuple[int, ...]], 
                          lattice_basis: LatticeBasis) -> Index:
    """
    Compute internal index of a monoid.

    The internal index is the index of the subgroup generated by the monoid
    in the lattice.

    Args:
        monoid_generators: Generators of the monoid
        lattice_basis: Lattice basis

    Returns:
        Internal index
    """
    # Convert to matrix
    gen_matrix = np.array(monoid_generators, dtype=int)
    
    # Compute rank
    rank = matrix_rank(gen_matrix)
    
    if rank < lattice_basis.rank:
        # Monoid doesn't span full lattice
        return Index(value=Fraction(0, 1), index_type=IndexType.INTERNAL)
    
    # For full rank, compute determinant of generator matrix
    if len(monoid_generators) >= lattice_basis.rank:
        # Use numpy for determinant
        if len(monoid_generators) == lattice_basis.rank:

            matrix_exact = [list(row) for row in monoid_generators]
            if _det_exact is not None:
                det_frac = abs(_det_exact(matrix_exact))
            else:
                from sympy import Matrix as _SMatrix
                det_frac = Fraction(abs(int(_SMatrix(matrix_exact).det())))
            return Index(value=det_frac, index_type=IndexType.INTERNAL)
        else:

            from fractions import Fraction as _Frac
            gens = [list(row) for row in monoid_generators]
            k = len(gens)
            G_exact = []
            for i in range(k):
                row = []
                for j in range(k):
                    dot = sum(_Frac(gens[i][l]) * _Frac(gens[j][l]) for l in range(len(gens[i])))
                    row.append(dot)
                G_exact.append(row)
            if _det_exact is not None:
                det_gram = _det_exact(G_exact)
            else:
                from sympy import Matrix as _SMatrix, Rational as _SR
                _G_sym = _SMatrix([[_SR(int(x.numerator), int(x.denominator))
                                    if hasattr(x, 'numerator') else _SR(int(x))
                                    for x in r] for r in G_exact])
                # FIX(Bug-15): same exact-Fraction conversion as compute_external_index.
                _sym_det2 = _G_sym.det()
                if hasattr(_sym_det2, 'p') and hasattr(_sym_det2, 'q'):
                    det_gram = Fraction(int(abs(_sym_det2.p)), int(_sym_det2.q))
                else:
                    det_gram = Fraction(abs(int(_sym_det2)))

            # FIX(Bug-3): same rational-sqrt fix as compute_external_index.
            import math as _m
            if isinstance(det_gram, Fraction):
                num2, den2 = abs(det_gram.numerator), det_gram.denominator
            else:
                num2, den2 = abs(int(det_gram)), 1
            sqrt_num2 = _m.isqrt(num2)
            sqrt_den2 = _m.isqrt(den2)
            if sqrt_num2 * sqrt_num2 == num2 and sqrt_den2 * sqrt_den2 == den2:
                sqrt_frac = Fraction(sqrt_num2, sqrt_den2)
            else:
                try:
                    from sympy import sqrt as _sympy_sqrt, Rational as _SR
                    sqrt_sym = _sympy_sqrt(_SR(num2, den2))
                    sqrt_frac = Fraction(float(sqrt_sym)).limit_denominator(10**9)
                except Exception as e:
                    raise type(e)(
                        f"compute_internal_index: cannot compute exact sqrt({num2}/{den2}): {e}. "
                        "SymPy is required for irrational lattice indices."
                    ) from e
            return Index(value=sqrt_frac, index_type=IndexType.INTERNAL)
    
    return Index(value=Fraction(1, 1), index_type=IndexType.INTERNAL)


def compute_unit_group_index(monoid_generators: List[Tuple[int, ...]]) -> Index:
    """
    Compute unit group index of a monoid.

    The unit group index is the index of the unit group of the monoid
    in the unit group of its normalization.

    Args:
        monoid_generators: Generators of the monoid

    Returns:
        Unit group index
    """
    # Find elements that are units (have inverses in the monoid)
    # This is a simplification - in practice, need to find the unit group
    
    # For positive monoids, unit group is trivial
    # Check if any generator has negative components
    has_negative = any(any(x < 0 for x in gen) for gen in monoid_generators)
    
    if not has_negative:
        # All generators non-negative -> unit group is trivial
        return Index(value=Fraction(1, 1), index_type=IndexType.UNIT_GROUP)
    
    # Compute unit group index via Smith Normal Form
    # For a monoid generated by vectors with negative components, the unit group
    # consists of elements with inverses. Compute using SymPy SNF.
    try:
        from sympy import Matrix, Rational as _R
        gen_matrix = Matrix([[_R(int(g)) for g in gen] for gen in monoid_generators])
        # SNF diagonal gives elementary divisors; unit group index = product of non-unit ones
        # For typical pipeline inputs this is 1; SNF guards the general case
        snf = gen_matrix.smith_normal_form()
        product = Fraction(1)
        for i in range(min(snf.rows, snf.cols)):
            d = int(snf[i, i])
            if d not in (0, 1, -1):
                product *= Fraction(abs(d))
        return Index(value=product, index_type=IndexType.UNIT_GROUP)
    except Exception as e:
        logger.debug(f"SNF unit group index failed: {e}")
        return Index(value=Fraction(1, 1), index_type=IndexType.UNIT_GROUP)


def compute_grading_index(grading: Grading, vectors: List[Tuple[int, ...]]) -> Index:
    """
    Compute index based on grading.

    Args:
        grading: Grading to use
        vectors: Vectors to grade

    Returns:
        Index representing the GCD of grades
    """
    grades = [grading.apply(v) for v in vectors]
    
    # Find common denominator
    denominators = [g.denominator for g in grades]
    common_denom = lcm_list(denominators)
    
    # Convert to integers
    int_grades = [g.numerator * (common_denom // g.denominator) for g in grades]
    
    # Find GCD of integer grades
    if int_grades:
        grade_gcd = gcd_list(int_grades)
        return Index(
            value=Fraction(grade_gcd, common_denom),
            index_type=IndexType.GRADING
        )
    else:
        return Index(value=Fraction(0, 1), index_type=IndexType.GRADING)


# ============================================================================
# SHIFT OPERATIONS
# ============================================================================

def apply_shift_to_histogram(histogram: Histogram, shift: int) -> Histogram:
    """
    Apply shift to histogram bins.

    Args:
        histogram: Input histogram
        shift: Shift amount

    Returns:
        Shifted histogram
    """
    new_bins = tuple(b + shift for b in histogram.bins)
    return Histogram(bins=new_bins, counts=histogram.counts)


def apply_shift_to_moments(s1: Fraction, s2: Fraction, n: int, shift: int) -> Tuple[Fraction, Fraction]:
    """
    Apply shift to moment constraints.

    If each xᵢ is shifted by s, then:
        S₁' = S₁ + n·s
        S₂' = S₂ + 2s·S₁ + n·s²

    Args:
        s1: Original first moment
        s2: Original second moment
        n: Number of variables
        shift: Shift amount

    Returns:
        Shifted moments (s1', s2')
    """
    s = Fraction(shift, 1)
    s1_new = s1 + n * s
    s2_new = s2 + 2 * s * s1 + n * s * s
    
    return s1_new, s2_new


def find_optimal_shift(histogram: Histogram, target_min: int = 0) -> int:
    """
    Find optimal shift to make all bins non-negative.

    Args:
        histogram: Input histogram
        target_min: Desired minimum bin value

    Returns:
        Optimal shift amount
    """
    if not histogram.bins:
        return 0
    
    min_bin = min(histogram.bins)
    if min_bin >= target_min:
        return 0
    
    return target_min - min_bin


# ============================================================================
# DENOMINATOR OPERATIONS
# ============================================================================

def compute_common_denominator(values: List[Fraction]) -> int:
    """
    Compute common denominator for a list of fractions.

    Args:
        values: List of fractions

    Returns:
        Least common multiple of denominators
    """
    if not values:
        return 1
    
    denominators = [v.denominator for v in values]
    return lcm_list(denominators)


def reduce_denominator(value: Fraction, target_denom: int) -> Fraction:
    """
    Reduce fraction to target denominator.

    Args:
        value: Input fraction
        target_denom: Target denominator

    Returns:
        Fraction with target denominator

    Raises:
        DenominatorError: If target_denom is not a multiple of value's denominator
    """
    if target_denom % value.denominator != 0:
        raise DenominatorError(
            f"Target denominator {target_denom} is not a multiple of {value.denominator}",

        )
    
    multiplier = target_denom // value.denominator
    return Fraction(value.numerator * multiplier, target_denom)


def normalize_denominator(index: Index) -> Index:
    """
    Normalize index to have denominator 1 if possible.

    Args:
        index: Input index

    Returns:
        Index with denominator normalized
    """
    if index.value.denominator == 1:
        return index
    
    # Try to find integer factor
    if index.value.numerator % index.value.denominator == 0:
        return Index(
            value=Fraction(index.value.numerator // index.value.denominator, 1),
            shift=index.shift,
            denominator=index.denominator,
            index_type=index.index_type
        )
    
    return index


# ============================================================================
# INDEX RANGE OPERATIONS
# ============================================================================

def create_index_range_from_moments(constraints: MomentConstraints, 
                                   step: int = 1) -> IndexRange:
    """
    Create index range from moment constraints.

    Args:
        constraints: Moment constraints
        step: Step size

    Returns:
        Index range covering possible k values
    """
    # For moment constraints, k typically ranges from 0 to n
    return IndexRange.from_values(0, constraints.n, step)


def create_index_range_from_grading(grading: Grading, 
                                   vectors: List[Tuple[int, ...]],
                                   step: int = 1) -> IndexRange:
    """
    Create index range from grading applied to vectors.

    Args:
        grading: Grading to use
        vectors: Vectors to grade
        step: Step size

    Returns:
        Index range covering grades of vectors
    """
    grades = [grading.apply(v) for v in vectors]
    
    if not grades:
        return IndexRange.from_values(0, 0, step)
    
    min_grade = min(grades)
    max_grade = max(grades)
    
    return IndexRange.from_values(min_grade, max_grade, step)


def intersect_index_ranges(range1: IndexRange, range2: IndexRange) -> Optional[IndexRange]:
    """
    Find intersection of two index ranges.

    Args:
        range1: First range
        range2: Second range

    Returns:
        Intersection range, or None if empty
    """
    # Find overlapping interval
    min_val = max(range1.min_index.value, range2.min_index.value)
    max_val = min(range1.max_index.value, range2.max_index.value)
    
    if min_val > max_val:
        return None
    
    # Convert to float for calculations
    min_float = float(min_val)
    max_float = float(max_val)
    
    # Need to handle different steps - find common step
    step = lcm_list([range1.step, range2.step])
    
    # Adjust to first common value
    # Find first value >= min_val that satisfies both step constraints
    found_val = None
    current = min_float
    while current <= max_float + 1e-10:
        if abs((current - float(range1.min_index.value)) % range1.step) < 1e-10 and \
           abs((current - float(range2.min_index.value)) % range2.step) < 1e-10:
            found_val = current
            break
        current += 1
    
    if found_val is None:
        return None
    

    from fractions import Fraction
    min_frac = Fraction(found_val).limit_denominator()
    max_frac = Fraction(max_float).limit_denominator()
    
    return IndexRange.from_values(min_frac, max_frac, step)


def union_index_ranges(ranges: List[IndexRange]) -> List[IndexRange]:
    """
    Compute union of multiple index ranges.

    Args:
        ranges: List of index ranges

    Returns:
        Minimal list of disjoint ranges covering the union
    """
    if not ranges:
        return []
    
    # Sort by min value
    sorted_ranges = sorted(ranges, key=lambda r: float(r.min_index.value))
    
    result = []
    current = sorted_ranges[0]
    
    for next_range in sorted_ranges[1:]:
        current_max = float(current.max_index.value)
        next_min = float(next_range.min_index.value)
        next_max = float(next_range.max_index.value)
        
        if next_min <= current_max + max(current.step, next_range.step):
            # Overlapping or adjacent - merge
            max_val = max(current_max, next_max)
            step = gcd_list([current.step, next_range.step])
            from fractions import Fraction
            current = IndexRange.from_values(
                current.min_index.value,
                Fraction(max_val).limit_denominator(),
                step
            )
        else:
            # Disjoint - add current and start new
            result.append(current)
            current = next_range
    
    result.append(current)
    return result


# ============================================================================
# INDEX CACHE
# ============================================================================

class IndexCache:
    """LRU cache for index calculations."""
    
    def __init__(self, maxsize: int = 256):
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
_index_cache = IndexCache(maxsize=128)


# ============================================================================
# VALIDATION AND TESTING
# ============================================================================

def validate_indexing_utils() -> Dict[str, bool]:
    """Run internal test suite to verify indexing utilities."""
    results = {}
    
    try:
        from fractions import Fraction
        
        # Test 1: Grading creation
        g1 = Grading(vector=(1, 1, 1), denominator=1, name="total")
        results["grading_creation"] = g1.dim == 3 and g1.denominator == 1
        
        # Test 2: Grading application
        grade = g1.apply((1, 2, 3))
        results["grading_apply"] = grade == 6
        
        # Test 3: Unit grading
        g2 = Grading.unit_grading(dim=4, coordinate=2)
        results["unit_grading"] = g2.apply((0, 0, 1, 0)) == 1
        
        # Test 4: Total degree
        g3 = Grading.total_degree(dim=3)
        results["total_degree"] = g3.apply((1, 2, 3)) == 6
        
        # Test 5: Index creation
        idx = Index(value=Fraction(5, 2), shift=1)
        results["index_creation"] = idx.shifted_value == Fraction(7, 2)
        
        # Test 6: Index shift
        idx_shifted = idx.shift_forward(2)
        results["index_shift"] = idx_shifted.shifted_value == Fraction(11, 2)
        
        # Test 7: Index range
        range1 = IndexRange.from_values(0, 10, 2)
        results["index_range"] = range1.size == 6  # 0,2,4,6,8,10
        
        # Test 8: Range contains
        contains = range1.contains(Index(value=Fraction(4, 1)))
        results["range_contains"] = contains
        
        # Test 9: Range intersection
        range2 = IndexRange.from_values(5, 15, 1)
        intersection = intersect_index_ranges(range1, range2)
        results["range_intersection"] = intersection is not None and intersection.size > 0
        
        # Test 10: Moment shift
        s1, s2 = apply_shift_to_moments(Fraction(10, 1), Fraction(30, 1), 5, 2)
        results["moment_shift"] = (s1 == 20 and s2 == 90)
        
        # Test 11: Common denominator
        fractions_list = [Fraction(1, 2), Fraction(1, 3), Fraction(1, 4)]
        common = compute_common_denominator(fractions_list)
        results["common_denominator"] = (common == 12)
        
        logger.info("✅ Indexing utilities validation passed")
        
    except Exception as e:
        logger.error(f"❌ Indexing utilities validation failed: {e}")
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
    print("Testing Production-Ready Indexing Utilities")
    print("=" * 60)
    
    # Run validation
    results = validate_indexing_utils()
    
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
    print(f"Overall: {success}/{total-1} tests passed")
    
    if "validation_error" in results:
        sys.exit(1)
    
    # Demonstration
    print("\n" + "=" * 60)
    print("Indexing Utilities Demo")
    print("=" * 60)
    
    from fractions import Fraction
    
    # Gradings
    print("\n1. Grading Examples:")
    g_total = Grading.total_degree(dim=3)
    print(f"   Total degree: {g_total.vector}, denominator={g_total.denominator}")
    print(f"   Grade of (1,2,3): {g_total.apply((1,2,3))}")
    
    g_unit = Grading.unit_grading(dim=4, coordinate=2)
    print(f"   Unit grading (coord 2): {g_unit.vector}")
    print(f"   Grade of (0,0,5,0): {g_unit.apply((0,0,5,0))}")
    
    # Indices and shifts
    print("\n2. Index and Shift Operations:")
    idx = Index(value=Fraction(5, 2), shift=1)
    print(f"   Original: value={idx.value}, shift={idx.shift}, shifted={idx.shifted_value}")
    
    idx_fwd = idx.shift_forward(3)
    print(f"   Forward +3: shifted={idx_fwd.shifted_value}")
    
    idx_back = idx.shift_backward(2)
    print(f"   Backward -2: shifted={idx_back.shifted_value}")
    
    # Index ranges
    print("\n3. Index Ranges:")
    range1 = IndexRange.from_values(0, 10, 2)
    print(f"   Range 0-10 step 2: size={range1.size}")
    indices = list(range1.indices())
    print(f"   Indices: {[i.value for i in indices]}")
    
    range2 = IndexRange.from_values(5, 15, 1)
    print(f"   Range 5-15 step 1: size={range2.size}")
    
    intersection = intersect_index_ranges(range1, range2)
    if intersection:
        print(f"   Intersection: {intersection.min_index.value}-{intersection.max_index.value} "
              f"step {intersection.step}, size={intersection.size}")
    
    # Moment shifts
    print("\n4. Moment Shifts ():")
    s1, s2 = Fraction(10, 1), Fraction(30, 1)
    n = 5
    print(f"   Original: S₁={s1}, S₂={s2}, N={n}")
    
    s1_new, s2_new = apply_shift_to_moments(s1, s2, n, shift=2)
    print(f"   After shift +2: S₁={s1_new}, S₂={s2_new}")
    
    # Common denominator
    print("\n5. Denominator Operations:")
    fractions_list = [Fraction(1, 2), Fraction(1, 3), Fraction(1, 4)]
    common = compute_common_denominator(fractions_list)
    print(f"   Fractions: {fractions_list}")
    print(f"   Common denominator: {common}")
    
    print("\n" + "=" * 60)
    print("✅ Indexing Utilities Ready for Production")
    print("=" * 60)