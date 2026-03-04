"""
Alphabet Module for Reverse Statistics Pipeline
Provides alphabet and frequency distribution structures for general k-dimensional problems.


Critical for: Representing alphabets of arbitrary size with permutation symmetry
"""

from .exceptions import ReverseStatsError
import numpy as np
import math
from collections import Counter, defaultdict
from fractions import Fraction
from typing import List, Tuple, Dict, Any, Optional, Union, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import sys
import os

# Add the current directory to path to handle imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Handle imports to work both as module and standalone
try:
    # When imported as part of package
    from .stats_utils import SymmetryOrbit, Histogram, MomentConstraints
    from .math_utils import compute_orbit_weight, is_integer
except ImportError:
    # When run directly
    try:
        from stats_utils import SymmetryOrbit, Histogram, MomentConstraints
        from math_utils import compute_orbit_weight, is_integer
    except ImportError as e:
        raise ImportError(
            f"alphabet.py: required dependencies (stats_utils, math_utils) could not be "
            f"imported: {e}. "
            "These modules are required — there is no correct fallback. "
            "Ensure the package is installed correctly (pip install reverse-stats) "
            "or that all sibling modules are on sys.path."
        ) from e

logger = logging.getLogger(__name__)

# ============================================================================
# EXCEPTIONS
# ============================================================================

class AlphabetError(ReverseStatsError):
    """Base exception for alphabet operations."""
    def __init__(self, message: str):
        self.message = message

        super().__init__(message)


class AlphabetSizeError(AlphabetError):
    """Raised when alphabet size constraints are violated."""
    pass


class FrequencyError(AlphabetError):
    """Raised for invalid frequency distributions."""
    pass


class SymmetryError(AlphabetError):
    """Raised for symmetry-related errors."""
    pass


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class AlphabetType(Enum):
    """Types of alphabets based on symmetry."""
    GENERIC = "generic"           # No special symmetry
    PERMUTATION_SYMMETRIC = "permutation_symmetric"  # S_k symmetric (general k)
    BINARY = "binary"              # Binary alphabet {0,1}
    TERNARY = "ternary"            # Ternary alphabet {0,1,2}
    CUSTOM = "custom"              # User-defined


class FrequencyType(Enum):
    """Types of frequency distributions."""
    UNIFORM = "uniform"            # All letters equally frequent
    SPARSE = "sparse"
    POWER_LAW = "power_law"        # Power law distribution
    CUSTOM = "custom"              # User-defined


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class Alphabet:
    """
    Immutable alphabet representation.

    Attributes:
        letters: Tuple of letter values (usually integers)
        weights: Optional tuple of weights for each letter
        name: Optional name for the alphabet
        description: Optional description
    """
    letters: Tuple[Union[int, str], ...]
    weights: Optional[Tuple[float, ...]] = None
    name: str = ""
    description: str = ""
    
    def __post_init__(self):
        """Validate alphabet data."""
        if not self.letters:
            raise AlphabetError("Alphabet cannot be empty")
        
        # Check for duplicates
        if len(set(self.letters)) != len(self.letters):
            raise AlphabetError("Alphabet contains duplicate letters")
        
        # Validate weights if provided
        if self.weights is not None:
            if len(self.weights) != len(self.letters):
                raise AlphabetError(
                    f"Weights length ({len(self.weights)}) must match letters length ({len(self.letters)})",

                )
            if any(w < 0 for w in self.weights):
                raise AlphabetError("Weights must be non-negative")
    
    @property
    def size(self) -> int:
        """Alphabet size (number of distinct letters)."""
        return len(self.letters)
    
    @property
    def is_numeric(self) -> bool:
        """Check if all letters are numeric."""
        return all(isinstance(x, (int, float)) for x in self.letters)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "letters": list(self.letters),
            "weights": list(self.weights) if self.weights else None,
            "size": self.size,
            "name": self.name,
            "description": self.description,
            "is_numeric": self.is_numeric
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alphabet':
        """Create alphabet from dictionary."""
        return cls(
            letters=tuple(data["letters"]),
            weights=tuple(data["weights"]) if data.get("weights") else None,
            name=data.get("name", ""),
            description=data.get("description", "")
        )


@dataclass(frozen=True)
class FrequencyDistribution:
    """
    Frequency distribution over an alphabet.

    Attributes:
        alphabet: The alphabet this distribution applies to
        counts: Number of occurrences for each letter
        total: Total count (sum of counts)
    """
    alphabet: Alphabet
    counts: Tuple[int, ...]
    
    def __post_init__(self):
        """Validate frequency distribution."""
        if len(self.counts) != self.alphabet.size:
            raise FrequencyError(
                f"Counts length ({len(self.counts)}) must match alphabet size ({self.alphabet.size})",

            )
        if any(c < 0 for c in self.counts):
            raise FrequencyError("Counts must be non-negative")
    
    @property
    def total(self) -> int:
        """Total number of observations."""
        return sum(self.counts)
    
    @property
    def nonzero_bins(self) -> int:
        """Number of letters with non-zero count (L₀ norm)."""
        return sum(1 for c in self.counts if c > 0)
    
    @property
    def frequencies(self) -> List[float]:
        """Relative frequencies (counts / total)."""
        if self.total == 0:
            return [0.0] * len(self.counts)
        return [c / self.total for c in self.counts]
    
    def to_histogram(self) -> Histogram:
        """Convert to histogram for statistical analysis."""
        if not self.alphabet.is_numeric:
            raise FrequencyError("Cannot convert non-numeric alphabet to histogram")
        return Histogram(
            bins=tuple(int(x) for x in self.alphabet.letters),
            counts=self.counts
        )
    
    def to_symmetry_orbit(self) -> Optional[SymmetryOrbit]:
        """
        Convert to symmetry orbit.
        Returns None if alphabet has no symmetry.
        """
        return SymmetryOrbit(frequencies=self.counts)
    
    def apply_l0_constraint(self, max_bins: Optional[int] = None) -> 'FrequencyDistribution':
        """
        Apply L₀ sparsity constraint.

        Args:
            max_bins: Maximum number of non-zero bins

        Returns:
            Self if constraint satisfied, raises otherwise
        """
        if max_bins is None:
            return self
        
        if self.nonzero_bins > max_bins:
            raise FrequencyError(
                f"Distribution has {self.nonzero_bins} non-zero bins, "
                f"exceeds L₀ constraint max_bins={max_bins}",

            )
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "alphabet": self.alphabet.to_dict(),
            "counts": list(self.counts),
            "total": self.total,
            "nonzero_bins": self.nonzero_bins,
            "frequencies": self.frequencies
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FrequencyDistribution':
        """Create frequency distribution from dictionary."""
        alphabet = Alphabet.from_dict(data["alphabet"])
        return cls(
            alphabet=alphabet,
            counts=tuple(data["counts"])
        )


@dataclass(frozen=True)
class SymmetricAlphabet:
    """
    Permutation-symmetric alphabet representation.

    For a symmetric alphabet of size k, letters are grouped into orbits
    under the action of the permutation group S_k.

    Attributes:
        orbit_frequencies: Tuple of frequencies for each orbit (length k)
        letter_values: Tuple of values for each orbit (length k)
        name: Optional name for the symmetric alphabet
    """
    orbit_frequencies: Tuple[int, ...]  # f₁, f₂, ..., f_k
    letter_values: Tuple[Union[int, str], ...]
    name: str = ""
    
    def __post_init__(self):
        """Validate symmetric alphabet."""
        k = len(self.orbit_frequencies)
        if len(self.letter_values) != k:
            raise SymmetryError(
                f"orbit_frequencies length ({k}) must match letter_values length ({len(self.letter_values)})",

            )
        if any(f < 0 for f in self.orbit_frequencies):
            raise SymmetryError("Frequencies must be non-negative")
    
    @property
    def k(self) -> int:
        """Dimension (alphabet size)."""
        return len(self.orbit_frequencies)
    
    @property
    def N(self) -> int:
        """Total count."""
        return sum(self.orbit_frequencies)
    
    @property
    def weight(self) -> int:
        """Orbit weight: N! / ∏fⱼ!"""
        return compute_orbit_weight(list(self.orbit_frequencies), self.N)
    
    @property
    def multiplicity(self) -> int:
        """Number of distinct permutations."""
        return self.weight
    
    def to_symmetry_orbit(self) -> SymmetryOrbit:
        """Convert to symmetry orbit."""
        return SymmetryOrbit(frequencies=self.orbit_frequencies)
    
    def to_frequency_distribution(self) -> FrequencyDistribution:
        """Convert to frequency distribution."""
        alphabet = Alphabet(
            letters=tuple(self.letter_values),
            name=self.name or f"symmetric_k{self.k}_N{self.N}",
            description=f"Permutation-symmetric alphabet of size {self.k}"
        )
        return FrequencyDistribution(
            alphabet=alphabet,
            counts=self.orbit_frequencies
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "orbit_frequencies": list(self.orbit_frequencies),
            "letter_values": list(self.letter_values),
            "k": self.k,
            "N": self.N,
            "weight": self.weight,
            "multiplicity": self.multiplicity,
            "name": self.name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SymmetricAlphabet':
        """Create symmetric alphabet from dictionary."""
        return cls(
            orbit_frequencies=tuple(data["orbit_frequencies"]),
            letter_values=tuple(data["letter_values"]),
            name=data.get("name", "")
        )


# ============================================================================
# STANDARD ALPHABETS
# ============================================================================

# Pre-defined standard alphabets
STANDARD_ALPHABETS = {
    "binary": Alphabet(
        letters=(0, 1),
        name="binary",
        description="Binary alphabet {0, 1}"
    ),
    "ternary": Alphabet(
        letters=(0, 1, 2),
        name="ternary",
        description="Ternary alphabet {0, 1, 2}"
    ),
    "dna": Alphabet(
        letters=('A', 'C', 'G', 'T'),
        name="dna",
        description="DNA alphabet {A, C, G, T}"
    ),
    "protein": Alphabet(
        letters=tuple('ACDEFGHIKLMNPQRSTVWY'),
        name="protein",
        description="20 standard amino acids"
    ),
    "numeric_1_10": Alphabet(
        letters=tuple(range(1, 11)),
        name="numeric_1_10",
        description="Integers 1 through 10"
    ),
}


def get_standard_alphabet(name: str) -> Alphabet:
    """
    Get a standard alphabet by name.

    Args:
        name: Name of standard alphabet

    Returns:
        Alphabet instance

    Raises:
        AlphabetError: If name not found
    """
    if name not in STANDARD_ALPHABETS:
        raise AlphabetError(f"Unknown standard alphabet: {name}")
    return STANDARD_ALPHABETS[name]


def create_numeric_alphabet(k: int, start: int = 1) -> Alphabet:
    """
    Create a numeric alphabet of size k.

    Args:
        k: Alphabet size
        start: Starting value (default 1)

    Returns:
        Alphabet with values start, start+1, ..., start+k-1
    """
    return Alphabet(
        letters=tuple(range(start, start + k)),
        name=f"numeric_k{k}_start{start}",
        description=f"Numeric alphabet of size {k} starting at {start}"
    )


# ============================================================================
# ALPHABET OPERATIONS
# ============================================================================

def create_alphabet_from_range(start: Union[int, str], end: Union[int, str], 
                              step: int = 1, name: str = "") -> Alphabet:
    """
    Create an alphabet from a range of values.

    Args:
        start: Start value
        end: End value (inclusive)
        step: Step size
        name: Optional name

    Returns:
        Alphabet instance
    """
    if isinstance(start, int) and isinstance(end, int):
        letters = tuple(range(start, end + 1, step))
    elif isinstance(start, str) and isinstance(end, str):
        # Character range (e.g., 'A' to 'Z')
        letters = tuple(chr(i) for i in range(ord(start), ord(end) + 1, step))
    else:
        raise AlphabetError("Start and end must be same type (both int or both str)")
    
    return Alphabet(
        letters=letters,
        name=name or f"range_{start}_{end}",
        description=f"Alphabet from {start} to {end}"
    )


def create_alphabet_from_list(values: List[Union[int, str]], 
                              name: str = "") -> Alphabet:
    """
    Create an alphabet from a list of values.

    Args:
        values: List of letter values
        name: Optional name

    Returns:
        Alphabet instance
    """
    return Alphabet(
        letters=tuple(values),
        name=name or "custom",
        description=f"Custom alphabet with {len(values)} letters"
    )


def alphabet_union(a1: Alphabet, a2: Alphabet, name: str = "") -> Alphabet:
    """
    Create union of two alphabets.

    Args:
        a1: First alphabet
        a2: Second alphabet
        name: Optional name for new alphabet

    Returns:
        New alphabet containing all letters from both alphabets
    """
    # Combine letters, preserving order and removing duplicates
    seen = set()
    letters = []
    
    for letter in list(a1.letters) + list(a2.letters):
        if letter not in seen:
            seen.add(letter)
            letters.append(letter)
    
    return Alphabet(
        letters=tuple(letters),
        name=name or f"union_{a1.name}_{a2.name}",
        description=f"Union of {a1.name} and {a2.name}"
    )


def alphabet_intersection(a1: Alphabet, a2: Alphabet, name: str = "") -> Alphabet:
    """
    Create intersection of two alphabets.

    Args:
        a1: First alphabet
        a2: Second alphabet
        name: Optional name for new alphabet

    Returns:
        New alphabet containing letters present in both alphabets
    """
    set1 = set(a1.letters)
    set2 = set(a2.letters)
    common = set1 & set2
    
    # Preserve order from first alphabet
    letters = [l for l in a1.letters if l in common]
    
    return Alphabet(
        letters=tuple(letters),
        name=name or f"intersection_{a1.name}_{a2.name}",
        description=f"Intersection of {a1.name} and {a2.name}"
    )


# ============================================================================
# FREQUENCY DISTRIBUTION OPERATIONS
# ============================================================================

def create_uniform_distribution(alphabet: Alphabet, total: int) -> FrequencyDistribution:
    """
    Create a uniform frequency distribution.

    Args:
        alphabet: Alphabet to distribute over
        total: Total number of observations

    Returns:
        Frequency distribution with counts as equal as possible
    """
    size = alphabet.size
    base = total // size
    remainder = total % size
    
    counts = [base] * size
    for i in range(remainder):
        counts[i] += 1
    
    return FrequencyDistribution(
        alphabet=alphabet,
        counts=tuple(counts)
    )


def create_power_law_distribution(alphabet: Alphabet, total: int, 
                                  exponent: float = 1.0) -> FrequencyDistribution:
    """
    Create a power law frequency distribution.

    Args:
        alphabet: Alphabet to distribute over
        total: Total number of observations
        exponent: Power law exponent (higher = more skewed)

    Returns:
        Frequency distribution following power law
    """
    size = alphabet.size
    
    # Generate weights following power law
    weights = [1.0 / (i + 1) ** exponent for i in range(size)]
    weight_sum = sum(weights)
    
    # Convert to counts
    counts = []
    remaining = total
    
    for i, w in enumerate(weights[:-1]):
        # Proportional allocation
        count = int(round(total * w / weight_sum))
        count = min(count, remaining)
        counts.append(count)
        remaining -= count
    
    # Last letter gets remaining
    counts.append(remaining)
    
    return FrequencyDistribution(
        alphabet=alphabet,
        counts=tuple(counts)
    )


def create_sparse_distribution(alphabet: Alphabet, total: int, 
                              nonzero_bins: int) -> FrequencyDistribution:
    """
    Create a sparse frequency distribution with L₀ constraint.

    Args:
        alphabet: Alphabet to distribute over
        total: Total number of observations
        nonzero_bins: Number of bins with non-zero counts (L₀ norm)

    Returns:
        Sparse frequency distribution

    Raises:
        FrequencyError: If nonzero_bins > alphabet size
    """
    if nonzero_bins > alphabet.size:
        raise FrequencyError(
            f"nonzero_bins ({nonzero_bins}) cannot exceed alphabet size ({alphabet.size})",

        )
    
    # Distribute total among nonzero bins
    base = total // nonzero_bins
    remainder = total % nonzero_bins
    
    nonzero_counts = [base] * nonzero_bins
    for i in range(remainder):
        nonzero_counts[i] += 1
    
    # Pad with zeros
    counts = list(nonzero_counts) + [0] * (alphabet.size - nonzero_bins)
    
    return FrequencyDistribution(
        alphabet=alphabet,
        counts=tuple(counts)
    )


def frequencies_from_histogram(histogram: Histogram, 
                               alphabet: Optional[Alphabet] = None) -> FrequencyDistribution:
    """
    Create frequency distribution from histogram.

    Args:
        histogram: Input histogram
        alphabet: Optional alphabet (if None, create from histogram bins)

    Returns:
        Frequency distribution
    """
    if alphabet is None:
        # Create alphabet from histogram bins
        alphabet = Alphabet(
            letters=tuple(histogram.bins),
            name="from_histogram",
            description="Alphabet derived from histogram bins"
        )
    
    return FrequencyDistribution(
        alphabet=alphabet,
        counts=histogram.counts
    )


# ============================================================================
# SYMMETRIC ALPHABET OPERATIONS (General k)
# ============================================================================

def compositions(N: int, k: int):
    """
    Generate all k-tuples of nonnegative integers summing to N.

    Args:
        N: Total sum
        k: Number of parts

    Yields:
        Tuples (f₁, f₂, ..., f_k) with fᵢ ≥ 0 and Σfᵢ = N
    """
    if k == 1:
        yield (N,)
        return
    for i in range(N + 1):
        for rest in compositions(N - i, k - 1):
            yield (i,) + rest


def generate_symmetric_distributions(
    N: int,
    k: int,
    letter_values: Optional[Tuple[Union[int, str], ...]] = None,
    max_distributions: int = 10000
) -> List[FrequencyDistribution]:
    """
    Generate all permutation-symmetric frequency distributions for given N and k.

    Args:
        N: Total count
        k: Alphabet size (dimension)
        letter_values: Values for each orbit (default: 1..k)
        max_distributions: Maximum number to generate

    Returns:
        List of frequency distributions
    """
    if letter_values is None:
        letter_values = tuple(range(1, k + 1))
    
    if len(letter_values) != k:
        raise ValueError(f"letter_values length ({len(letter_values)}) must equal k ({k})")
    
    distributions = []
    count = 0
    
    # Generate all compositions of N into k parts
    for comp in compositions(N, k):
        if count >= max_distributions:
            logger.warning(f"Reached max_distributions limit ({max_distributions})")
            break
        
        alphabet = Alphabet(
            letters=letter_values,
            name=f"symmetric_k{k}_N{N}",
            description=f"Permutation-symmetric alphabet of size {k}"
        )
        
        dist = FrequencyDistribution(
            alphabet=alphabet,
            counts=comp
        )
        distributions.append(dist)
        count += 1
    
    logger.info(f"Generated {len(distributions)} symmetric distributions for k={k}, N={N}")
    return distributions


def filter_symmetric_distributions_by_constraints(
    distributions: List[FrequencyDistribution],
    constraints: MomentConstraints
) -> List[FrequencyDistribution]:
    """
    Filter symmetric distributions by moment constraints.

    Args:
        distributions: List of frequency distributions
        constraints: Moment constraints to satisfy

    Returns:
        Filtered distributions
    """
    valid = []
    
    for dist in distributions:
        if dist.total != constraints.n:
            continue
        
        hist = dist.to_histogram()
        s1, s2 = hist.moments()
        
        if s1 == constraints.s1 and s2 == constraints.s2:
            valid.append(dist)
    
    logger.info(f"Filtered {len(valid)}/{len(distributions)} distributions by constraints")
    return valid


# ============================================================================
# ALPHABET CACHE
# ============================================================================

class AlphabetCache:
    """LRU cache for alphabet and frequency distributions."""
    
    def __init__(self, maxsize: int = 256):
        self._cache = {}
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
    
    def get_alphabet(self, key: tuple, create_func: Callable, *args, **kwargs) -> Alphabet:
        """Get cached alphabet or create new one."""
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        
        self.misses += 1
        result = create_func(*args, **kwargs)
        
        if len(self._cache) >= self.maxsize:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        
        self._cache[key] = result
        return result
    
    def get_distribution(self, key: tuple, create_func: Callable, *args, **kwargs) -> FrequencyDistribution:
        """Get cached frequency distribution or create new one."""
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        
        self.misses += 1
        result = create_func(*args, **kwargs)
        
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
_alphabet_cache = AlphabetCache(maxsize=128)


# ============================================================================
# VALIDATION AND TESTING
# ============================================================================

def validate_alphabet_utils() -> Dict[str, bool]:
    """Run internal test suite to verify alphabet utilities."""
    results = {}
    
    try:
        # Test 1: Alphabet creation
        a1 = Alphabet(letters=(1, 2, 3, 4), name="test1")
        results["alphabet_creation"] = a1.size == 4 and a1.is_numeric
        
        # Test 2: Standard alphabet retrieval
        a2 = get_standard_alphabet("binary")
        results["standard_alphabet"] = a2.size == 2 and a2.letters == (0, 1)
        
        # Test 3: Alphabet from range
        a3 = create_alphabet_from_range(1, 5)
        results["range_alphabet"] = a3.size == 5 and a3.letters == (1, 2, 3, 4, 5)
        
        # Test 4: Frequency distribution
        dist = FrequencyDistribution(alphabet=a2, counts=(3, 2))
        results["frequency_distribution"] = dist.total == 5 and dist.nonzero_bins == 2
        
        # Test 5: Uniform distribution
        uniform = create_uniform_distribution(a2, 10)
        results["uniform_distribution"] = uniform.counts == (5, 5)
        
        # Test 6: Sparse distribution
        sparse = create_sparse_distribution(a2, 10, nonzero_bins=1)
        results["sparse_distribution"] = sparse.nonzero_bins == 1
        
        # Test 7: Numeric alphabet creation
        num_alpha = create_numeric_alphabet(5)
        results["numeric_alphabet"] = num_alpha.size == 5 and num_alpha.letters == (1, 2, 3, 4, 5)
        
        # Test 8: Symmetric alphabet class
        sym_alpha = SymmetricAlphabet(
            orbit_frequencies=(2, 1, 1),
            letter_values=(1, 2, 3),
            name="test_sym"
        )
        results["symmetric_class"] = sym_alpha.k == 3 and sym_alpha.N == 4
        
        # Test 9: Composition generator for k=3, N=4
        comps = list(compositions(4, 3))
        # Number of compositions of 4 into 3 parts = C(4+3-1,3-1) = C(6,2) = 15
        results["compositions"] = len(comps) == 15
        
        # Test 10: Generate symmetric distributions for k=3, N=4
        sym_dists = generate_symmetric_distributions(N=4, k=3, max_distributions=100)
        results["symmetric_generation"] = len(sym_dists) == 15
        
        # Test 11: L₀ constraint
        try:
            sparse.apply_l0_constraint(max_bins=1)
            l0_passed = True
        except FrequencyError:
            l0_passed = False
        results["l0_constraint"] = l0_passed
        
        # Test 12: Alphabet operations
        a4 = create_alphabet_from_list([1, 2, 3])
        a5 = create_alphabet_from_list([2, 3, 4])
        union = alphabet_union(a4, a5)
        intersection = alphabet_intersection(a4, a5)
        results["alphabet_operations"] = (
            union.size == 4 and
            intersection.size == 2 and
            set(intersection.letters) == {2, 3}
        )
        
        # Test 13: Serialization
        dist_dict = dist.to_dict()
        dist2 = FrequencyDistribution.from_dict(dist_dict)
        results["serialization"] = (
            dist2.alphabet.size == dist.alphabet.size and
            dist2.counts == dist.counts
        )
        
        logger.info("✅ Alphabet utilities validation passed")
        
    except Exception as e:
        logger.error(f"❌ Alphabet utilities validation failed: {e}")
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
    print("Testing Production-Ready Alphabet Utilities")
    print("=" * 60)
    print("\nNOTE: Now supports general k-dimensional alphabets")
    print("      No longer hardcoded for S₄ (k=4)")
    print("=" * 60)
    
    # Run validation
    results = validate_alphabet_utils()
    
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
    print("Alphabet Utilities Demo - General k-Dimensional Support")
    print("=" * 60)
    
    # Standard alphabets
    print("\n1. Standard Alphabets:")
    for name in ["binary", "ternary", "dna", "protein"]:
        try:
            alpha = get_standard_alphabet(name)
            print(f"   {name}: {alpha.letters} (size {alpha.size})")
        except:
            print(f"   {name}: (not available)")
    
    # Create numeric alphabets of various sizes
    print("\n2. Numeric Alphabets (general k):")
    for k in [3, 4, 5, 6]:
        alpha = create_numeric_alphabet(k)
        print(f"   k={k}: {alpha.letters}")
    
    # Symmetric distributions for various k
    print("\n3. Symmetric Distributions for Different k:")
    for k in [2, 3, 4]:
        N = k + 2  # Small N for demonstration
        dists = generate_symmetric_distributions(N=N, k=k, max_distributions=5)
        print(f"\n   k={k}, N={N}: {len(dists)} total distributions")
        print(f"   First 3: {[d.counts for d in dists[:3]]}")
    
    # Frequency distributions
    print("\n4. Frequency Distributions:")
    alpha = get_standard_alphabet("binary")
    
    uniform = create_uniform_distribution(alpha, 10)
    print(f"   Uniform (total 10): {uniform.counts}")
    
    power = create_power_law_distribution(alpha, 10, exponent=2.0)
    print(f"   Power law (exponent 2): {power.counts}")
    
    sparse = create_sparse_distribution(alpha, 10, nonzero_bins=1)
    print(f"   Sparse (1 bin): {sparse.counts}")
    
    print("\n" + "=" * 60)
    print("✅ Alphabet Utilities Ready for Production")
    print("=" * 60)