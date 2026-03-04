from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Union, Set
from enum import Enum
import numpy as np
from fractions import Fraction


# ============================================================================
# Core Data Structures
# ============================================================================
@dataclass
class ObservedStatistics:
    """
    Observed statistics from data.

    Attributes:
        s1: First moment (sum)
        s2: Second moment (sum of squares)
        n: Number of observations
        frequencies: Frequency distribution if available
    """
    s1: int
    s2: int
    n: int
    frequencies: Optional[List[int]] = None
    
    def __post_init__(self):
        """Validate statistics."""
        if self.n < 0:
            raise ValueError(f"n must be non-negative, got {self.n}")
    
    @property
    def mean(self) -> float:
        """Sample mean."""
        return self.s1 / self.n if self.n > 0 else 0.0
    
    @property
    def variance(self) -> float:
        """Sample variance."""
        if self.n <= 1:
            return 0.0
        return (self.s2 - self.s1**2 / self.n) / (self.n - 1)


@dataclass
class AlphabetInfo:
    """
    Information about an alphabet.

    Attributes:
        letters: The actual letters/symbols
        size: Size of alphabet
        is_binary: Whether alphabet is binary (0/1)
        is_symmetric: Whether alphabet has symmetry
    """
    letters: Tuple[Any, ...]
    size: int = field(init=False)  # Computed from letters
    is_binary: bool = field(init=False)
    is_symmetric: bool = False
    
    def __post_init__(self):
        """Initialize alphabet info."""
        self.size = len(self.letters)
        self.is_binary = set(self.letters) == {0, 1} or set(self.letters) == {'0', '1'}
    
    def contains(self, letter: Any) -> bool:
        """Check if letter is in alphabet."""
        return letter in self.letters
    
    def index(self, letter: Any) -> int:
        """Get index of letter in alphabet."""
        return self.letters.index(letter)


@dataclass
class ConstraintSystem:
    """
    System of linear constraints.

    Attributes:
        A: Coefficient matrix
        b: Right-hand side vector
        eq: Whether constraints are equalities (True) or inequalities (False)
        variables: Variable names
    """
    A: np.ndarray
    b: np.ndarray
    eq: bool = True
    variables: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate constraint system."""
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError(f"A rows {self.A.shape[0]} != b length {self.b.shape[0]}")
        
        if self.variables is None:
            self.variables = [f"x{i}" for i in range(self.A.shape[1])]
    
    @property
    def num_constraints(self) -> int:
        """Number of constraints."""
        return self.A.shape[0]
    
    @property
    def num_variables(self) -> int:
        """Number of variables."""
        return self.A.shape[1]
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate constraints at point x."""
        return self.A @ x - self.b


# ============================================================================
# Pipeline Configuration
# ============================================================================
@dataclass
class PipelineConfig:
    """
    Pipeline configuration.

    Attributes:
        max_dimension: Maximum dimension for computations
        integrality_tolerance: Tolerance for integrality checks
        random_seed: Random seed for reproducibility
        cache_results: Whether to cache results
        parallel: Whether to use parallel processing
        num_workers: Number of worker processes
    """
    max_dimension: int = 15
    integrality_tolerance: float = 1e-10
    random_seed: int = 42
    cache_results: bool = True
    parallel: bool = False
    num_workers: int = 4


# ============================================================================
# Enums
# ============================================================================
class OptimizationDirection(Enum):
    """Optimization direction."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ConstraintType(Enum):
    """Type of constraint."""
    EQUALITY = "eq"
    INEQUALITY_LE = "le"
    INEQUALITY_GE = "ge"


class Algorithm(Enum):
    """Available algorithms."""
    FOURIER_MOTZKIN = "fourier_motzkin"
    LATTICE_WALK = "lattice_walk"
    MARKOV_BASIS = "markov_basis"
    GIBBS_SAMPLER = "gibbs_sampler"
    METROPOLIS_HASTINGS = "metropolis_hastings"


class PhaseStatus(Enum):
    """Status of a pipeline phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class SymmetryType(Enum):
    """Type of symmetry."""
    NONE = "none"
    PERMUTATION = "permutation"
    REFLECTION = "reflection"
    ROTATION = "rotation"
    FULL = "full"


class ConeType(Enum):
    """Type of cone."""
    SIMPLICIAL = "simplicial"
    UNIMODULAR = "unimodular"
    NON_UNIMODULAR = "non_unimodular"
    GENERAL = "general"


# ============================================================================
# Exceptions
# ============================================================================
try:
    from .exceptions import ReverseStatsError
except ImportError:
    class ReverseStatsError(Exception):
        """Base exception for Reverse Statistics."""
        pass


# ============================================================================
# Testing
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Pipeline Types")
    print("=" * 60)
    
    # Test ObservedStatistics
    stats = ObservedStatistics(s1=10, s2=30, n=5)
    print(f"Statistics: mean={stats.mean}, variance={stats.variance}")
    
    # Test AlphabetInfo - size is now computed automatically
    alphabet = AlphabetInfo(letters=(0, 1, 2))
    print(f"Alphabet: size={alphabet.size}, binary={alphabet.is_binary}")
    
    # Test ConstraintSystem
    A = np.array([[1, 1], [1, -1]])
    b = np.array([1, 0])
    constraints = ConstraintSystem(A=A, b=b)
    print(f"Constraints: {constraints.num_constraints} x {constraints.num_variables}")
    
    # Test PipelineConfig
    config = PipelineConfig()
    print(f"PipelineConfig: max_dimension={config.max_dimension}")
    
    print("\n✅ Pipeline Types ready")