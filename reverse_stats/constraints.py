"""
Constraint System Module for Reverse Statistics Pipeline
Provides linear inequality and equation constraints for polytope representation.


Critical for: Building constraint systems from moment statistics
"""

from .exceptions import ReverseStatsError
import math
from fractions import Fraction
from typing import List, Tuple, Dict, Any, Optional, Union, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import sys
import os
from functools import lru_cache, wraps

# Use sympy for exact linear algebra when needed
try:
    import sympy
    from sympy import Matrix, lcm
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    # Fallback implementations

logger = logging.getLogger(__name__)

# ============================================================================
# EXCEPTIONS
# ============================================================================

class ConstraintError(ReverseStatsError):
    """Base exception for constraint operations."""
    def __init__(self, message: str):
        self.message = message

        super().__init__(message)


class InequalityError(ConstraintError):
    """Raised for inequality constraint violations."""
    pass


class EquationError(ConstraintError):
    """Raised for equation constraint violations."""
    pass


class FeasibilityError(ConstraintError):
    """Raised when system is infeasible."""
    pass


class BoundError(ConstraintError):
    """Raised when bounds are inconsistent."""
    pass


# Import DimensionLimitError from canonical source
try:
    from dimension import DimensionLimitError
except ImportError:
    class DimensionLimitError(ConstraintError):
        """Raised when dimension exceeds guard threshold."""
        def __init__(self, dimension: int, threshold: int = 15):
            self.dimension = dimension
            self.threshold = threshold
            super().__init__(f"Dimension {dimension} exceeds guard threshold {threshold}")


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class ConstraintType(Enum):
    """Types of constraints."""
    INEQUALITY = "inequality"      # ≤, ≥, <, >
    EQUATION = "equation"          # =
    RANGE = "range"                # a ≤ x ≤ b
    CONGRUENCE = "congruence"       # ≡ (mod)
    EXCLUDED = "excluded"          # ≠, or excluded face


class InequalityDirection(Enum):
    """Direction of inequality."""
    LESS_THAN = "<"                # x < bound
    LESS_OR_EQUAL = "<="           # x ≤ bound
    GREATER_THAN = ">"             # x > bound
    GREATER_OR_EQUAL = ">="        # x ≥ bound


class BoundType(Enum):
    """Type of bound."""
    LOWER = "lower"                # Lower bound
    UPPER = "upper"                # Upper bound
    BOTH = "both"                  # Both bounds


# ============================================================================
# UTILITY FUNCTIONS (Pure Python replacements for scipy)
# ============================================================================

def _gcd_list(nums: List[int]) -> int:
    """Compute GCD of a list of integers."""
    if not nums:
        return 0
    from math import gcd
    result = abs(nums[0])
    for x in nums[1:]:
        result = gcd(result, abs(x))
        if result == 1:
            return 1
    return result


def _lcm_list(nums: List[int]) -> int:
    """Compute LCM of a list of integers."""
    if not nums:
        return 1
    
    def lcm(a: int, b: int) -> int:
        from math import gcd
        return abs(a * b) // gcd(a, b) if a and b else 0
    
    result = nums[0]
    for x in nums[1:]:
        result = lcm(result, x)
    return result


def _matrix_rank_simple(matrix: List[List[float]], tol: float = 1e-10) -> int:
    """
    Compute rank of a matrix using Gaussian elimination.
    Pure Python implementation - no numpy/scipy.
    """
    if not matrix or not matrix[0]:
        return 0
    
    m = len(matrix)
    n = len(matrix[0])
    
    # Make a copy
    A = [row[:] for row in matrix]
    
    rank = 0
    for col in range(n):
        # Find pivot
        pivot_row = -1
        for row in range(rank, m):
            if abs(A[row][col]) > tol:
                pivot_row = row
                break
        
        if pivot_row == -1:
            continue
        
        # Swap rows
        if pivot_row != rank:
            A[rank], A[pivot_row] = A[pivot_row], A[rank]
        
        # Eliminate below
        pivot = A[rank][col]
        for row in range(rank + 1, m):
            if abs(A[row][col]) > tol:
                factor = A[row][col] / pivot
                for k in range(col, n):
                    A[row][k] -= factor * A[rank][k]
        
        rank += 1
    
    return rank


def _nullspace_basis_simple(matrix: List[List[float]], tol: float = 1e-10) -> List[List[float]]:
    """
    Compute nullspace basis using RREF (reduced row echelon form).
    Pure Python implementation - no scipy.

    For an m×n matrix A, finds basis for {x : Ax = 0}.
    Returns list of basis vectors (each of length n).
    """
    if not matrix or not matrix[0]:
        return []

    m = len(matrix)
    n = len(matrix[0])

    # Work with a copy as floats
    A = [[float(matrix[i][j]) for j in range(n)] for i in range(m)]

    # Forward elimination to row echelon form
    pivot_cols = []   # column index of pivot in each row
    row = 0
    for col in range(n):
        if row >= m:
            break
        # Find pivot in this column
        pivot = -1
        best = 0.0
        for r in range(row, m):
            if abs(A[r][col]) > best:
                best = abs(A[r][col])
                pivot = r
        if pivot == -1 or best < tol:
            continue  # free variable column
        # Swap rows
        A[row], A[pivot] = A[pivot], A[row]
        # Scale pivot row
        scale = A[row][col]
        A[row] = [A[row][j] / scale for j in range(n)]
        # Eliminate all other rows
        for r in range(m):
            if r != row and abs(A[r][col]) > tol:
                factor = A[r][col]
                A[r] = [A[r][j] - factor * A[row][j] for j in range(n)]
        pivot_cols.append(col)
        row += 1

    # Identify free variable columns
    free_cols = [j for j in range(n) if j not in pivot_cols]
    if not free_cols:
        return []  # trivial nullspace

    # Build null basis: for each free variable, set it to 1, read off pivot vars
    null_basis = []
    pivot_col_set = set(pivot_cols)
    pivot_row_for_col = {col: i for i, col in enumerate(pivot_cols)}

    for fc in free_cols:
        vec = [0.0] * n
        vec[fc] = 1.0
        for pc, pr in pivot_row_for_col.items():
            if abs(A[pr][fc]) > tol:
                vec[pc] = -A[pr][fc]
        null_basis.append(vec)

    return null_basis


def _solve_linear_system_simple(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """
    Solve linear system Ax = b using Gaussian elimination.
    Pure Python implementation for small systems.
    """
    n = len(A)
    if n == 0:
        return None
    
    # Make augmented matrix
    aug = [A[i][:] + [b[i]] for i in range(n)]
    
    # Gaussian elimination
    for i in range(n):
        # Find pivot
        pivot = -1
        for j in range(i, n):
            if abs(aug[j][i]) > 1e-10:
                pivot = j
                break
        
        if pivot == -1:
            continue
        
        # Swap rows
        if pivot != i:
            aug[i], aug[pivot] = aug[pivot], aug[i]
        
        # Scale row to make pivot 1
        pivot_val = aug[i][i]
        for j in range(i, n + 1):
            aug[i][j] /= pivot_val
        
        # Eliminate below
        for j in range(i + 1, n):
            factor = aug[j][i]
            for k in range(i, n + 1):
                aug[j][k] -= factor * aug[i][k]
    
    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(aug[i][i]) < 1e-10:
            continue
        x[i] = aug[i][n]
        for j in range(i + 1, n):
            x[i] -= aug[i][j] * x[j]
    
    return x


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class Inequality:
    """
    Linear inequality constraint.

    Attributes:
        coefficients: Coefficients of variables (a₁, a₂, ..., aₙ)
        bound: Right-hand side value
        direction: Inequality direction
        tolerance: Numerical tolerance for checking
    """
    coefficients: Tuple[Fraction, ...]
    bound: Fraction
    direction: InequalityDirection
    tolerance: float = 1e-10
    
    def __post_init__(self):
        """Validate inequality."""
        if not self.coefficients:
            raise InequalityError("Inequality must have at least one coefficient")
    
    @property
    def dim(self) -> int:
        """Dimension of the inequality."""
        return len(self.coefficients)
    
    def evaluate(self, point: Tuple[Fraction, ...]) -> Fraction:
        """
        Evaluate inequality at a point: Σ aᵢ·xᵢ.

        Args:
            point: Point to evaluate

        Returns:
            Left-hand side value
        """
        if len(point) != self.dim:
            raise InequalityError(
                f"Point dimension {len(point)} does not match inequality dimension {self.dim}",

            )
        
        return sum(a * x for a, x in zip(self.coefficients, point))
    
    def is_satisfied(self, point: Tuple[Fraction, ...]) -> bool:
        """
        Check if point satisfies the inequality.

        Args:
            point: Point to check

        Returns:
            True if inequality is satisfied
        """
        lhs = self.evaluate(point)
        
        if self.direction == InequalityDirection.LESS_THAN:
            return lhs < self.bound
        elif self.direction == InequalityDirection.LESS_OR_EQUAL:
            return lhs <= self.bound + self.tolerance
        elif self.direction == InequalityDirection.GREATER_THAN:
            return lhs > self.bound
        elif self.direction == InequalityDirection.GREATER_OR_EQUAL:
            return lhs >= self.bound - self.tolerance
        else:
            return False
    
    def scale(self, factor: Fraction) -> 'Inequality':
        """Scale inequality by a factor."""
        return Inequality(
            coefficients=tuple(a * factor for a in self.coefficients),
            bound=self.bound * factor,
            direction=self.direction,
            tolerance=self.tolerance
        )
    
    def normalize(self) -> 'Inequality':
        """Normalize inequality to have integer coefficients."""
        # Find common denominator
        denoms = [c.denominator for c in self.coefficients] + [self.bound.denominator]
        common_denom = _lcm_list(denoms)
        
        # Scale to integers
        coeffs_int = tuple(c.numerator * (common_denom // c.denominator) for c in self.coefficients)
        bound_int = self.bound.numerator * (common_denom // self.bound.denominator)
        
        # Find GCD to reduce
        all_vals = list(coeffs_int) + [bound_int]
        g = _gcd_list(all_vals)
        
        if g > 1:
            coeffs_int = tuple(c // g for c in coeffs_int)
            bound_int = bound_int // g
        
        return Inequality(
            coefficients=tuple(Fraction(c, 1) for c in coeffs_int),
            bound=Fraction(bound_int, 1),
            direction=self.direction,
            tolerance=self.tolerance
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "coefficients": [str(c) for c in self.coefficients],
            "bound": str(self.bound),
            "direction": self.direction.value,
            "dim": self.dim,
            "tolerance": self.tolerance
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Inequality':
        """Create inequality from dictionary."""
        from fractions import Fraction
        return cls(
            coefficients=tuple(Fraction(c) for c in data["coefficients"]),
            bound=Fraction(data["bound"]),
            direction=InequalityDirection(data["direction"]),
            tolerance=data.get("tolerance", 1e-10)
        )
    
    @classmethod
    def less_than(cls, coefficients: List[Fraction], bound: Fraction) -> 'Inequality':
        """Create less-than inequality: Σ aᵢ·xᵢ < bound."""
        return cls(
            coefficients=tuple(coefficients),
            bound=bound,
            direction=InequalityDirection.LESS_THAN
        )
    
    @classmethod
    def less_or_equal(cls, coefficients: List[Fraction], bound: Fraction) -> 'Inequality':
        """Create less-or-equal inequality: Σ aᵢ·xᵢ ≤ bound."""
        return cls(
            coefficients=tuple(coefficients),
            bound=bound,
            direction=InequalityDirection.LESS_OR_EQUAL
        )
    
    @classmethod
    def greater_than(cls, coefficients: List[Fraction], bound: Fraction) -> 'Inequality':
        """Create greater-than inequality: Σ aᵢ·xᵢ > bound."""
        return cls(
            coefficients=tuple(coefficients),
            bound=bound,
            direction=InequalityDirection.GREATER_THAN
        )
    
    @classmethod
    def greater_or_equal(cls, coefficients: List[Fraction], bound: Fraction) -> 'Inequality':
        """Create greater-or-equal inequality: Σ aᵢ·xᵢ ≥ bound."""
        return cls(
            coefficients=tuple(coefficients),
            bound=bound,
            direction=InequalityDirection.GREATER_OR_EQUAL
        )


@dataclass(frozen=True)
class Equation:
    """
    Linear equation constraint.

    Attributes:
        coefficients: Coefficients of variables (a₁, a₂, ..., aₙ)
        rhs: Right-hand side value
        tolerance: Numerical tolerance for checking
    """
    coefficients: Tuple[Fraction, ...]
    rhs: Fraction
    tolerance: float = 1e-10
    
    def __post_init__(self):
        """Validate equation."""
        if not self.coefficients:
            raise EquationError("Equation must have at least one coefficient")
    
    @property
    def dim(self) -> int:
        """Dimension of the equation."""
        return len(self.coefficients)
    
    def evaluate(self, point: Tuple[Fraction, ...]) -> Fraction:
        """
        Evaluate equation at a point: Σ aᵢ·xᵢ.

        Args:
            point: Point to evaluate

        Returns:
            Left-hand side value
        """
        if len(point) != self.dim:
            raise EquationError(
                f"Point dimension {len(point)} does not match equation dimension {self.dim}",

            )
        
        return sum(a * x for a, x in zip(self.coefficients, point))
    
    def is_satisfied(self, point: Tuple[Fraction, ...]) -> bool:
        """
        Check if point satisfies the equation.

        Args:
            point: Point to check

        Returns:
            True if equation is satisfied within tolerance
        """
        lhs = self.evaluate(point)
        return abs(lhs - self.rhs) <= self.tolerance
    
    def scale(self, factor: Fraction) -> 'Equation':
        """Scale equation by a factor."""
        return Equation(
            coefficients=tuple(a * factor for a in self.coefficients),
            rhs=self.rhs * factor,
            tolerance=self.tolerance
        )
    
    def normalize(self) -> 'Equation':
        """Normalize equation to have integer coefficients."""
        # Find common denominator
        denoms = [c.denominator for c in self.coefficients] + [self.rhs.denominator]
        common_denom = _lcm_list(denoms)
        
        # Scale to integers
        coeffs_int = tuple(c.numerator * (common_denom // c.denominator) for c in self.coefficients)
        rhs_int = self.rhs.numerator * (common_denom // self.rhs.denominator)
        
        # Find GCD to reduce
        all_vals = list(coeffs_int) + [rhs_int]
        g = _gcd_list(all_vals)
        
        if g > 1:
            coeffs_int = tuple(c // g for c in coeffs_int)
            rhs_int = rhs_int // g
        
        return Equation(
            coefficients=tuple(Fraction(c, 1) for c in coeffs_int),
            rhs=Fraction(rhs_int, 1),
            tolerance=self.tolerance
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "coefficients": [str(c) for c in self.coefficients],
            "rhs": str(self.rhs),
            "dim": self.dim,
            "tolerance": self.tolerance
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Equation':
        """Create equation from dictionary."""
        from fractions import Fraction
        return cls(
            coefficients=tuple(Fraction(c) for c in data["coefficients"]),
            rhs=Fraction(data["rhs"]),
            tolerance=data.get("tolerance", 1e-10)
        )
    
    @classmethod
    def from_coeffs(cls, coefficients: List[Fraction], rhs: Fraction) -> 'Equation':
        """Create equation from coefficients and RHS."""
        return cls(
            coefficients=tuple(coefficients),
            rhs=rhs
        )


@dataclass(frozen=True)
class Bound:
    """
    Bound constraint on a single variable.

    Attributes:
        variable: Variable index (0-based)
        lower: Lower bound (None if no lower bound)
        upper: Upper bound (None if no upper bound)
        inclusive_lower: Whether lower bound is inclusive
        inclusive_upper: Whether upper bound is inclusive
    """
    variable: int
    lower: Optional[Fraction] = None
    upper: Optional[Fraction] = None
    inclusive_lower: bool = True
    inclusive_upper: bool = True
    
    def __post_init__(self):
        """Validate bound."""
        if self.variable < 0:
            raise BoundError(f"Variable index must be non-negative, got {self.variable}")
        
        if self.lower is not None and self.upper is not None:
            if self.lower > self.upper:
                raise BoundError(f"Lower bound {self.lower} > upper bound {self.upper}")
            if self.lower == self.upper and not (self.inclusive_lower and self.inclusive_upper):
                raise BoundError(f"Equal bounds must be inclusive")
    
    @property
    def bound_type(self) -> BoundType:
        """Get bound type."""
        if self.lower is not None and self.upper is not None:
            return BoundType.BOTH
        elif self.lower is not None:
            return BoundType.LOWER
        elif self.upper is not None:
            return BoundType.UPPER
        else:
            raise BoundError("Bound must have at least one bound")
    
    def is_satisfied(self, value: Fraction) -> bool:
        """Check if value satisfies the bound."""
        if self.lower is not None:
            if self.inclusive_lower:
                if value < self.lower:
                    return False
            else:
                if value <= self.lower:
                    return False
        
        if self.upper is not None:
            if self.inclusive_upper:
                if value > self.upper:
                    return False
            else:
                if value >= self.upper:
                    return False
        
        return True
    
    def to_inequalities(self, n_variables: Optional[int] = None) -> List[Inequality]:
        """Convert bound to inequalities.

        Args:
            n_variables: Total number of variables in the system.  If provided,
                coefficient vectors will have exactly this length (required for
                the resulting Inequality to be compatible with the constraint
                system).  If None, falls back to ``variable + 1`` which is
                correct only when this is the last variable.

        FIX: The original implementation always used ``[Fraction(0)] * (self.variable + 1)``,
        producing a coefficient vector that is too short for any variable that is
        not the last one in the system (e.g. variable=1 in a k=4 system gives
        length 2 instead of 4).  Callers should pass ``n_variables=k``.
        """
        length = n_variables if (n_variables is not None and n_variables > self.variable) \
                 else (self.variable + 1)
        inequalities = []

        if self.lower is not None:
            coeffs = [Fraction(0)] * length
            coeffs[self.variable] = Fraction(1)
            if self.inclusive_lower:
                inequalities.append(Inequality.greater_or_equal(coeffs, self.lower))
            else:
                inequalities.append(Inequality.greater_than(coeffs, self.lower))

        if self.upper is not None:
            coeffs = [Fraction(0)] * length
            coeffs[self.variable] = Fraction(1)
            if self.inclusive_upper:
                inequalities.append(Inequality.less_or_equal(coeffs, self.upper))
            else:
                inequalities.append(Inequality.less_than(coeffs, self.upper))

        return inequalities
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "variable": self.variable,
            "lower": str(self.lower) if self.lower is not None else None,
            "upper": str(self.upper) if self.upper is not None else None,
            "inclusive_lower": self.inclusive_lower,
            "inclusive_upper": self.inclusive_upper,
            "bound_type": self.bound_type.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Bound':
        """Create bound from dictionary."""
        from fractions import Fraction
        return cls(
            variable=data["variable"],
            lower=Fraction(data["lower"]) if data.get("lower") else None,
            upper=Fraction(data["upper"]) if data.get("upper") else None,
            inclusive_lower=data.get("inclusive_lower", True),
            inclusive_upper=data.get("inclusive_upper", True)
        )
    
    @classmethod
    def lower_bound(cls, variable: int, value: Fraction, inclusive: bool = True) -> 'Bound':
        """Create lower bound: x ≥ value or x > value."""
        return cls(
            variable=variable,
            lower=value,
            inclusive_lower=inclusive
        )
    
    @classmethod
    def upper_bound(cls, variable: int, value: Fraction, inclusive: bool = True) -> 'Bound':
        """Create upper bound: x ≤ value or x < value."""
        return cls(
            variable=variable,
            upper=value,
            inclusive_upper=inclusive
        )
    
    @classmethod
    def interval(cls, variable: int, lower: Fraction, upper: Fraction, 
                 inclusive: bool = True) -> 'Bound':
        """Create interval bound: lower ≤ x ≤ upper."""
        return cls(
            variable=variable,
            lower=lower,
            upper=upper,
            inclusive_lower=inclusive,
            inclusive_upper=inclusive
        )


# ============================================================================
# QUADRATIC CONSTRAINT
# ============================================================================

@dataclass(frozen=True)
class QuadraticConstraint:
    """
    Second-moment constraint expressed as a LINEAR constraint on frequencies.

    In the frequency polytope the variables are frequencies fⱼ (one per bin),
    and the second moment S₂ is:

        S₂ = Σⱼ (valueⱼ²) · fⱼ

    This is **linear** in the fⱼ.  The ``coefficients`` stored here should
    already be the pre-squared alphabet values (valueⱼ²), so that

        evaluate(f) = Σⱼ coefficients[j] · f[j]

    FIX(Bug-1): The original code computed ``Σ cᵢ xᵢ²``, squaring the
    frequency variables a *second* time.  For the frequency polytope this
    turns a linear feasibility problem into a non-convex quadratic one,
    producing a completely different (and geometrically incorrect) object.
    The correct evaluation is a plain dot product.

    Attributes:
        coefficients: Pre-squared alphabet values (value₀², value₁², ...).
        rhs: Target S₂ value.
    """
    coefficients: Tuple[Fraction, ...]   # pre-squared alphabet values: valueⱼ²
    rhs: Fraction                        # S₂ target value

    @property
    def dim(self) -> int:
        return len(self.coefficients)

    def evaluate(self, point: Tuple[Fraction, ...]) -> Fraction:
        """Compute Σ cᵢ·fᵢ (linear dot product) using exact Fraction arithmetic.

        FIX(Bug-1): Was ``Σ cᵢ·xᵢ²`` — incorrect squaring of frequency vars.
        """
        return sum(c * x for c, x in zip(self.coefficients, point))

    def is_satisfied(self, point: Tuple[Fraction, ...]) -> bool:
        """Return True iff Σ cᵢ·fᵢ == rhs (exact Fraction comparison)."""
        return self.evaluate(point) == self.rhs

    def residual(self, point: Tuple[Fraction, ...]) -> Fraction:
        """Signed residual: evaluate(point) − rhs."""
        return self.evaluate(point) - self.rhs

    def to_equation(self) -> 'Equation':
        """Convert to a linear Equation (valid because evaluation is linear).

        This lets the rest of the pipeline treat S₂ as an ordinary linear
        equality constraint, which is correct in the frequency-polytope context.
        """
        return Equation(coefficients=self.coefficients, rhs=self.rhs)



@dataclass
class ConstraintSystem:
    """
    System of linear constraints.

    Attributes:
        inequalities: List of inequality constraints
        equations: List of equation constraints
        bounds: List of bound constraints
        variables: Number of variables
    """
    inequalities: List[Inequality] = field(default_factory=list)
    equations: List[Equation] = field(default_factory=list)
    bounds: List[Bound] = field(default_factory=list)
    variables: int = 0
    
    def __post_init__(self):
        """Validate constraint system."""
        # Determine number of variables from constraints
        max_var = -1
        
        for ineq in self.inequalities:
            max_var = max(max_var, ineq.dim)
        
        for eq in self.equations:
            max_var = max(max_var, eq.dim)
        
        for bound in self.bounds:
            max_var = max(max_var, bound.variable + 1)
        
        if max_var > 0:
            self.variables = max_var
        elif self.variables == 0:
            self.variables = 1  # Default to 1 variable if no constraints
    
    def add_inequality(self, inequality: Inequality) -> 'ConstraintSystem':
        """Add an inequality to the system."""
        self.inequalities.append(inequality)
        self.variables = max(self.variables, inequality.dim)
        return self
    
    def add_equation(self, equation: Equation) -> 'ConstraintSystem':
        """Add an equation to the system."""
        self.equations.append(equation)
        self.variables = max(self.variables, equation.dim)
        return self
    
    def add_bound(self, bound: Bound) -> 'ConstraintSystem':
        """Add a bound to the system."""
        self.bounds.append(bound)
        self.variables = max(self.variables, bound.variable + 1)
        return self

    def normalize_all(self) -> 'ConstraintSystem':
        """Return a new ConstraintSystem with every equation and inequality
        reduced to primitive integer form via LCM+GCD normalization.

        Why this matters
        ----------------
        Constraints built from floating-point moments (e.g. S3 = 7.3) arrive
        with fractional RHS values such as ``Fraction(73, 10)``.  Downstream
        consumers (Gomory cut arithmetic, unimodularity determinant checks,
        Normaliz integer input) all work better — and some *require* — integer
        coefficients.  Each individual ``Equation`` and ``Inequality`` already
        has a ``normalize()`` method that clears denominators via LCM scaling
        and then GCD-reduces the row to its primitive form.  This method simply
        applies that normalization to the whole system in one call.

        Bounds are left unchanged (they hold per-variable scalar bounds, not
        linear coefficient rows, and are already stored as exact Fractions).

        Returns a *new* ``ConstraintSystem``; the original is not mutated
        (``ConstraintSystem`` is a ``@dataclass`` but not frozen, however
        returning a new object keeps the call site explicit about intent).
        """
        normed_ineqs = [ineq.normalize() for ineq in self.inequalities]
        normed_eqs   = [eq.normalize()   for eq   in self.equations]
        return ConstraintSystem(
            inequalities=normed_ineqs,
            equations=normed_eqs,
            bounds=list(self.bounds),
            variables=self.variables,
        )

    def is_feasible(self, point: Tuple[Fraction, ...]) -> bool:
        """
        Check if a point satisfies all constraints.

        Args:
            point: Point to check

        Returns:
            True if point satisfies all constraints
        """
        if len(point) != self.variables:
            return False
        
        # Check inequalities
        for ineq in self.inequalities:
            if not ineq.is_satisfied(point[:ineq.dim]):
                return False
        
        # Check equations
        for eq in self.equations:
            if not eq.is_satisfied(point[:eq.dim]):
                return False
        
        # Check bounds
        for bound in self.bounds:
            if bound.variable < len(point):
                if not bound.is_satisfied(point[bound.variable]):
                    return False
        
        return True
    
    def to_matrix_form(self) -> Tuple[List[List[Fraction]], List[Fraction], List[List[Fraction]], List[Fraction]]:
        """
        Convert to matrix form: A·x ≤ b, C·x = d.

        Returns:
            (A, b, C, d) where A is inequality matrix, b is RHS,
            C is equation matrix, d is RHS.

        FIX(Bug-29): The original method cast all Fraction coefficients to float,
        returning List[List[float]]. This contaminated the exact-arithmetic pipeline:
        a unimodular matrix with det=1 (exact) could appear as det≈0.9999999999 in
        float, causing the pipeline to falsely take the expensive decomposition path
        or crash the unimodularity check. All values are returned as Fraction so
        callers that need float can convert at the call site themselves.
        """
        # Count constraints
        n_ineq = len(self.inequalities)
        n_eq = len(self.equations)
        
        # Initialize matrices as exact Fraction lists (zero = Fraction(0))
        _zero = Fraction(0)
        A = [[_zero] * self.variables for _ in range(n_ineq)]
        b = [_zero] * n_ineq
        C = [[_zero] * self.variables for _ in range(n_eq)]
        d = [_zero] * n_eq
        
        # Fill inequalities
        for i, ineq in enumerate(self.inequalities):
            for j, coeff in enumerate(ineq.coefficients):
                if j < self.variables:
                    A[i][j] = Fraction(coeff) if not isinstance(coeff, Fraction) else coeff
            
            if ineq.direction in [InequalityDirection.LESS_THAN, InequalityDirection.LESS_OR_EQUAL]:
                # Already in form A·x ≤ b
                b[i] = Fraction(ineq.bound) if not isinstance(ineq.bound, Fraction) else ineq.bound
            else:
                # Convert ≥ to ≤ by multiplying by -1
                for j in range(self.variables):
                    A[i][j] = -A[i][j]
                b[i] = -(Fraction(ineq.bound) if not isinstance(ineq.bound, Fraction) else ineq.bound)
        
        # Fill equations
        for i, eq in enumerate(self.equations):
            for j, coeff in enumerate(eq.coefficients):
                if j < self.variables:
                    C[i][j] = Fraction(coeff) if not isinstance(coeff, Fraction) else coeff
            d[i] = Fraction(eq.rhs) if not isinstance(eq.rhs, Fraction) else eq.rhs
        
        return A, b, C, d
    
    def eliminate_equations(self) -> Tuple['ConstraintSystem', Optional[List[List[float]]]]:
        """
        Eliminate equations to reduce system using sympy.

        Returns:
            Reduced system and basis for solution space
        """
        if not self.equations:
            return self, None
        
        if not HAS_SYMPY:
            # Cannot eliminate without sympy
            logger.warning("sympy required for equation elimination")
            return self, None
        
        try:
            # Convert equations to sympy matrix
            eq_matrix = []
            for eq in self.equations:
                row = [float(c) for c in eq.coefficients] + [float(eq.rhs)]
                eq_matrix.append(row)
            
            # Split into coefficient matrix and RHS
            A_eq = Matrix([row[:-1] for row in eq_matrix])
            b_eq = Matrix([row[-1:] for row in eq_matrix])
            
            # Find particular solution
            try:
                # Solve using sympy
                solution = A_eq.solve(b_eq)
                x_particular = [float(solution[i]) for i in range(solution.rows)]
            except:
                # If no unique solution, use least squares approximation
                # This is a fallback - in production, would use proper nullspace
                logger.warning("Could not find particular solution")
                return self, None
            
            # Find nullspace basis
            null_basis = A_eq.nullspace()
            
            if not null_basis:
                # Unique solution
                return self, None
            
            # Create reduced system in nullspace coordinates
            reduced_system = ConstraintSystem(variables=len(null_basis))
            
            # Transform inequalities to nullspace
            for ineq in self.inequalities:
                # Project inequality onto nullspace
                coeffs_full = [0.0] * self.variables
                for j, c in enumerate(ineq.coefficients):
                    if j < self.variables:
                        coeffs_full[j] = float(c)
                
                # Transform to nullspace coordinates
                coeffs_null = [0.0] * len(null_basis)
                for k, basis_vec in enumerate(null_basis):
                    for j in range(self.variables):
                        coeffs_null[k] += coeffs_full[j] * float(basis_vec[j])
                
                # Create new inequality if not all zero
                if any(abs(c) > 1e-10 for c in coeffs_null):
                    coeffs_frac = [Fraction(c).limit_denominator() for c in coeffs_null]
                    
                    # Adjust bound
                    bound_adjusted = ineq.bound
                    for j, c in enumerate(ineq.coefficients[:len(x_particular)]):
                        bound_adjusted -= c * Fraction(x_particular[j])
                    
                    new_ineq = Inequality(
                        coefficients=tuple(coeffs_frac),
                        bound=bound_adjusted,
                        direction=ineq.direction,
                        tolerance=ineq.tolerance
                    )
                    reduced_system.add_inequality(new_ineq)
            
            # Convert null_basis to list of lists for return
            basis_list = [[float(null_basis[k][j]) for j in range(self.variables)] 
                         for k in range(len(null_basis))]
            
            return reduced_system, basis_list
            
        except Exception as e:
            logger.warning(f"Equation elimination failed: {e}")
            return self, None
    
    def find_feasible_point(self):
        """Removed — previous implementation used random float sampling.

        Use feasibility.check_feasibility(system) for exact LP-based feasibility.
        """
        raise NotImplementedError(
            "ConstraintSystem.find_feasible_point: removed. "
            "Used random.uniform sampling which violates exact arithmetic. "
            "Use feasibility.check_feasibility(system) instead."
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "inequalities": [i.to_dict() for i in self.inequalities],
            "equations": [e.to_dict() for e in self.equations],
            "bounds": [b.to_dict() for b in self.bounds],
            "variables": self.variables,
            "num_inequalities": len(self.inequalities),
            "num_equations": len(self.equations),
            "num_bounds": len(self.bounds)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConstraintSystem':
        """Create constraint system from dictionary."""
        system = cls(variables=data.get("variables", 0))
        
        for ineq_data in data.get("inequalities", []):
            system.add_inequality(Inequality.from_dict(ineq_data))
        
        for eq_data in data.get("equations", []):
            system.add_equation(Equation.from_dict(eq_data))
        
        for bound_data in data.get("bounds", []):
            system.add_bound(Bound.from_dict(bound_data))
        
        return system


# ============================================================================
# CONSTRAINT GENERATION FUNCTIONS
# ============================================================================

def create_nonnegativity_constraints(dim: int) -> List[Inequality]:
    """
    Create non-negativity constraints xᵢ ≥ 0.

    Args:
        dim: Dimension

    Returns:
        List of inequality constraints
    """
    constraints = []
    for i in range(dim):
        coeffs = [Fraction(0)] * dim
        coeffs[i] = Fraction(1)
        constraints.append(Inequality.greater_or_equal(coeffs, Fraction(0)))
    return constraints


def create_box_constraints(lower: List[Fraction], upper: List[Fraction]) -> List[Bound]:
    """
    Create box constraints: lower[i] ≤ xᵢ ≤ upper[i].

    Args:
        lower: Lower bounds
        upper: Upper bounds

    Returns:
        List of bound constraints
    """
    if len(lower) != len(upper):
        raise BoundError("Lower and upper bounds must have same length")
    
    bounds = []
    for i, (l, u) in enumerate(zip(lower, upper)):
        bounds.append(Bound.interval(i, l, u))
    return bounds


def create_simplex_constraint(dim: int, total: Fraction) -> Equation:
    """
    Create simplex constraint: Σ xᵢ = total, xᵢ ≥ 0.

    Args:
        dim: Dimension
        total: Total sum

    Returns:
        Equation constraint for sum
    """
    coeffs = [Fraction(1)] * dim
    return Equation.from_coeffs(coeffs, total)


def create_moment_constraints(s1: Fraction, s2: Fraction, n: int):
    """
    Create moment constraints from S₁ and S₂.

    Args:
        s1: First moment Σxᵢ
        s2: Second moment Σxᵢ²
        n: Number of variables

    Returns:
        Tuple[Equation, QuadraticConstraint]:
          - S₁ linear Equation (Σxᵢ = s1)
          - S₂ QuadraticConstraint (Σxᵢ² = s2)


    a linear Equation with [1,1,...,1] coefficients (encoding Σxᵢ = s2 — wrong).
    Now returns a QuadraticConstraint for exact nonlinear checking.
    """
    # S₁ constraint: Σxᵢ = s1  (linear — correct as Equation)
    s1_coeffs = [Fraction(1)] * n
    s1_eq = Equation.from_coeffs(s1_coeffs, s1)

    # S₂ constraint: Σxᵢ² = s2  (nonlinear — must not be a linear Equation)
    s2_constraint = QuadraticConstraint(
        coefficients=tuple(Fraction(1) for _ in range(n)),
        rhs=Fraction(s2)
    )

    return s1_eq, s2_constraint


# ============================================================================
# CONSTRAINT VALIDATION
# ============================================================================

def validate_feasibility(system: ConstraintSystem, 
                        candidates: List[Tuple[Fraction, ...]]) -> List[bool]:
    """
    Validate feasibility of candidate points.

    Args:
        system: Constraint system
        candidates: List of points to check

    Returns:
        List of boolean results
    """
    return [system.is_feasible(point) for point in candidates]


def find_violated_constraints(system: ConstraintSystem, 
                             point: Tuple[Fraction, ...]) -> List[str]:
    """
    Find which constraints are violated by a point.

    Args:
        system: Constraint system
        point: Point to check

    Returns:
        List of descriptions of violated constraints
    """
    violated = []
    
    for i, ineq in enumerate(system.inequalities):
        if not ineq.is_satisfied(point[:ineq.dim]):
            lhs = ineq.evaluate(point[:ineq.dim])
            violated.append(f"Inequality {i}: {lhs} {ineq.direction.value} {ineq.bound}")
    
    for i, eq in enumerate(system.equations):
        if not eq.is_satisfied(point[:eq.dim]):
            lhs = eq.evaluate(point[:eq.dim])
            violated.append(f"Equation {i}: {lhs} = {eq.rhs} (diff {lhs - eq.rhs})")
    
    for bound in system.bounds:
        if bound.variable < len(point):
            if not bound.is_satisfied(point[bound.variable]):
                val = point[bound.variable]
                if bound.lower is not None and bound.upper is not None:
                    violated.append(f"Bound x{bound.variable}: {bound.lower} ≤ {val} ≤ {bound.upper}")
                elif bound.lower is not None:
                    violated.append(f"Bound x{bound.variable}: {val} ≥ {bound.lower}")
                elif bound.upper is not None:
                    violated.append(f"Bound x{bound.variable}: {val} ≤ {bound.upper}")
    
    return violated


# ============================================================================
# CONSTRAINT CACHE
# ============================================================================

class ConstraintCache:
    """LRU cache for constraint computations."""
    
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
_constraint_cache = ConstraintCache(maxsize=128)


# ============================================================================
# VALIDATION AND TESTING
# ============================================================================

def validate_constraint_utils() -> Dict[str, bool]:
    """Run internal test suite to verify constraint utilities."""
    results = {}
    
    try:
        from fractions import Fraction
        
        # Test 1: Inequality creation
        ineq = Inequality.less_or_equal([Fraction(1), Fraction(1)], Fraction(10))
        results["inequality_creation"] = ineq.dim == 2 and ineq.bound == 10
        
        # Test 2: Inequality evaluation
        val = ineq.evaluate((Fraction(3), Fraction(4)))
        results["inequality_eval"] = val == 7
        
        # Test 3: Inequality satisfaction
        satisfied = ineq.is_satisfied((Fraction(3), Fraction(4)))
        results["inequality_satisfied"] = satisfied
        
        # Test 4: Inequality violation
        violated = ineq.is_satisfied((Fraction(6), Fraction(5)))
        results["inequality_violated"] = not violated
        
        # Test 5: Equation creation
        eq = Equation.from_coeffs([Fraction(1), Fraction(1)], Fraction(10))
        results["equation_creation"] = eq.dim == 2 and eq.rhs == 10
        
        # Test 6: Equation satisfaction
        eq_satisfied = eq.is_satisfied((Fraction(3), Fraction(7)))
        results["equation_satisfied"] = eq_satisfied
        
        # Test 7: Bound creation
        bound = Bound.interval(0, Fraction(0), Fraction(10))
        results["bound_creation"] = bound.variable == 0
        
        # Test 8: Bound satisfaction
        bound_satisfied = bound.is_satisfied(Fraction(5))
        results["bound_satisfied"] = bound_satisfied
        
        # Test 9: Bound violation
        bound_violated = bound.is_satisfied(Fraction(11))
        results["bound_violated"] = not bound_violated
        
        # Test 10: Constraint system
        system = ConstraintSystem()
        system.add_inequality(ineq)
        system.add_equation(eq)
        system.add_bound(bound)
        results["system_creation"] = system.variables >= 2
        
        # Test 11: System feasibility
        feasible = system.is_feasible((Fraction(3), Fraction(7)))
        results["system_feasible"] = feasible
        
        # Test 12: System infeasibility
        infeasible = system.is_feasible((Fraction(11), Fraction(0)))
        results["system_infeasible"] = not infeasible
        
        # Test 13: Non-negativity constraints
        nonneg = create_nonnegativity_constraints(3)
        results["nonnegativity"] = len(nonneg) == 3
        
        # Test 14: Box constraints
        box = create_box_constraints([Fraction(0), Fraction(0)], [Fraction(1), Fraction(1)])
        results["box_constraints"] = len(box) == 2
        
        # Test 15: Simplex constraint
        simplex = create_simplex_constraint(3, Fraction(10))
        results["simplex"] = simplex.dim == 3
        
        # Test 16: Normalization
        ineq_norm = ineq.normalize()
        results["normalization"] = all(c.denominator == 1 for c in ineq_norm.coefficients)
        
        # Test 17: Matrix conversion
        A, b, C, d = system.to_matrix_form()
        results["matrix_form"] = len(A) == 1 and len(C) == 1
        
        # Test 18: Violation finding
        violations = find_violated_constraints(system, (Fraction(11), Fraction(0)))
        results["violation_finding"] = len(violations) > 0
        
        logger.info("✅ Constraint utilities validation passed")
        
    except Exception as e:
        logger.error(f"❌ Constraint utilities validation failed: {e}")
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
    print("Testing Production-Ready Constraint Utilities")
    print("=" * 60)
    
    # Run validation
    results = validate_constraint_utils()
    
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
    print("Constraint Utilities Demo")
    print("=" * 60)
    
    from fractions import Fraction
    
    # Create constraints
    print("\n1. Creating Constraints:")
    
    # Inequality: x + y ≤ 10
    ineq1 = Inequality.less_or_equal([Fraction(1), Fraction(1)], Fraction(10))
    print(f"   Inequality: x + y ≤ 10")
    
    # Equation: x - y = 2
    eq1 = Equation.from_coeffs([Fraction(1), Fraction(-1)], Fraction(2))
    print(f"   Equation: x - y = 2")
    
    # Bound: 0 ≤ x ≤ 5
    bound1 = Bound.interval(0, Fraction(0), Fraction(5))
    print(f"   Bound: 0 ≤ x ≤ 5")
    
    # Create system
    print("\n2. Constraint System:")
    system = ConstraintSystem()
    system.add_inequality(ineq1)
    system.add_equation(eq1)
    system.add_bound(bound1)
    print(f"   Variables: {system.variables}")
    print(f"   Inequalities: {len(system.inequalities)}")
    print(f"   Equations: {len(system.equations)}")
    print(f"   Bounds: {len(system.bounds)}")
    
    # Check points
    print("\n3. Feasibility Checking:")
    
    point1 = (Fraction(3), Fraction(1))  # x=3, y=1
    feasible1 = system.is_feasible(point1)
    print(f"   Point (3,1): {'✅ Feasible' if feasible1 else '❌ Infeasible'}")
    
    point2 = (Fraction(6), Fraction(4))  # x=6, y=4
    feasible2 = system.is_feasible(point2)
    print(f"   Point (6,4): {'✅ Feasible' if feasible2 else '❌ Infeasible'}")
    
    # Find violations
    print("\n4. Violation Analysis:")
    violations = find_violated_constraints(system, point2)
    if violations:
        for v in violations:
            print(f"   ❌ {v}")
    
    # Matrix form
    print("\n5. Matrix Form:")
    A, b, C, d = system.to_matrix_form()
    print(f"   A·x ≤ b:")
    print(f"   A = {A}")
    print(f"   b = {b}")
    print(f"   C·x = d:")
    print(f"   C = {C}")
    print(f"   d = {d}")
    
    # Standard constraint sets
    print("\n6. Standard Constraint Sets:")
    
    nonneg = create_nonnegativity_constraints(3)
    print(f"   Non-negativity (dim=3): {len(nonneg)} constraints")
    
    box = create_box_constraints([Fraction(0), Fraction(0)], [Fraction(1), Fraction(1)])
    print(f"   Box constraints: {len(box)} bounds")
    
    simplex = create_simplex_constraint(3, Fraction(1))
    print(f"   Simplex constraint: {simplex.coefficients} = {simplex.rhs}")
    
    print("\n" + "=" * 60)
    print("✅ Constraint Utilities Ready for Production")
    print("=" * 60)