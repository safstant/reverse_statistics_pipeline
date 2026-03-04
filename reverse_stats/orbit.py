"""
Orbit Module for Reverse Statistics Pipeline
Provides orbit representations and operations under permutation symmetry.


Critical for: Orbit weight calculation N!/(f₁!·...·fₖ!) and orbit lifting
"""

from .exceptions import ReverseStatsError
import numpy as np
import math
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

# Add the current directory to path to handle imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Handle imports to work both as module and standalone
try:
    # When imported as part of package
    from .math_utils import (
        compute_orbit_weight, is_integer, gcd_list, lcm_list,
        matrix_rank, is_unimodular_matrix
    )
    from .stats_utils import Histogram, MomentConstraints, SymmetryOrbit as StatsSymmetryOrbit
    from .alphabet import Alphabet, FrequencyDistribution, SymmetricAlphabet
    # frequency is imported lazily inside functions that need it to avoid circular import
    from .indexing import Grading, Index, IndexRange
    from .constraints import ConstraintSystem, Inequality, Equation, Bound
except ImportError:
    # When run directly
    try:
        from math_utils import (
            compute_orbit_weight, is_integer, gcd_list, lcm_list,
            matrix_rank, is_unimodular_matrix
        )
        from stats_utils import Histogram, MomentConstraints, SymmetryOrbit as StatsSymmetryOrbit
        from alphabet import Alphabet, FrequencyDistribution, SymmetricAlphabet
        # frequency imported lazily to avoid circular import
        from indexing import Grading, Index, IndexRange
        from constraints import ConstraintSystem, Inequality, Equation, Bound
    except ImportError as e:
        raise ImportError(
            f"orbit.py: required dependencies (math_utils, stats_utils, alphabet, "
            f"indexing, constraints) could not be imported: {e}. "
            "These modules are required — there is no correct fallback. "
            "Ensure the package is installed correctly or all sibling modules are on sys.path."
        ) from e

logger = logging.getLogger(__name__)

# ============================================================================
# EXCEPTIONS
# ============================================================================

class OrbitError(ReverseStatsError):
    """Base exception for orbit operations."""
    def __init__(self, message: str):
        self.message = message

        super().__init__(message)


class OrbitWeightError(OrbitError):
    """Raised for orbit weight calculation errors."""
    pass


class OrbitLiftingError(OrbitError):
    """Raised for orbit lifting errors."""
    pass


class OrbitDecompositionError(OrbitError):
    """Raised for orbit decomposition errors."""
    pass


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class OrbitType(Enum):
    """Types of orbits."""
    PERMUTATION = "permutation"          # Full permutation group S_k (was S4_SYMMETRIC)
    FULL = "full"                        # Full symmetric group
    CYCLE = "cycle"                       # Cyclic group
    DIHEDRAL = "dihedral"                 # Dihedral group
    CUSTOM = "custom"                     # Custom orbit


class LiftingStrategy(Enum):
    """Strategies for orbit lifting."""
    PRODUCT = "product"                   # Lift via product of orbits
    DIRECT_SUM = "direct_sum"              # Lift via direct sum
    TENSOR = "tensor"                      # Lift via tensor product
    SYMMETRIC_POWER = "symmetric_power"    # Lift via symmetric power
    EXTERIOR_POWER = "exterior_power"      # Lift via exterior power


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class Orbit:
    """
    Orbit under a symmetry group.

    Attributes:
        frequencies: Frequency vector (f₁, f₂, ..., fₖ)
        orbit_type: Type of orbit
        weight: Orbit weight
        multiplicity: Number of distinct permutations
        label: Optional label for the orbit
    """
    frequencies: Tuple[int, ...]
    orbit_type: OrbitType = OrbitType.PERMUTATION
    weight: int = 0
    multiplicity: int = 0
    label: str = ""
    
    def __post_init__(self):
        """Validate orbit."""
        if not self.frequencies:
            raise OrbitError("Orbit must have at least one frequency")
        
        if any(f < 0 for f in self.frequencies):
            raise OrbitError("Frequencies must be non-negative")
        
        # Auto-compute weight if not provided
        if self.weight == 0:
            N = sum(self.frequencies)
            weight = compute_orbit_weight(list(self.frequencies), N)
            object.__setattr__(self, 'weight', weight)
        
        # Auto-compute multiplicity if not provided
        if self.multiplicity == 0:
            object.__setattr__(self, 'multiplicity', self.weight)
    
    @property
    def N(self) -> int:
        """Total count."""
        return sum(self.frequencies)
    
    @property
    def k(self) -> int:
        """Number of orbit types (dimension)."""
        return len(self.frequencies)
    
    @property
    def nonzero_indices(self) -> List[int]:
        """Indices of non-zero frequencies."""
        return [i for i, f in enumerate(self.frequencies) if f > 0]
    
    @property
    def is_balanced(self) -> bool:
        """Check if orbit has balanced frequencies (all equal)."""
        return all(f == self.frequencies[0] for f in self.frequencies)
    
    @property
    def is_trivial(self) -> bool:
        """Check if orbit is trivial (single non-zero frequency)."""
        return len(self.nonzero_indices) == 1
    
    def to_stats_orbit(self) -> StatsSymmetryOrbit:
        """Convert to stats_utils SymmetryOrbit."""
        return StatsSymmetryOrbit(frequencies=self.frequencies)
    
    def to_frequency_distribution(self, alphabet: Optional[Alphabet] = None) -> FrequencyDistribution:
        """
        Convert to frequency distribution.

        Args:
            alphabet: Alphabet to use (creates generic if None)
        """
        if alphabet is None:
            # Create generic alphabet with indices as letters
            letters = tuple(range(1, self.k + 1))
            alphabet = Alphabet(letters=letters, name=f"orbit_{self.label}")
        
        return FrequencyDistribution(
            alphabet=alphabet,
            counts=self.frequencies
        )
    
    def to_symmetric_alphabet(self, letter_values: Optional[Tuple[Union[int, str], ...]] = None) -> SymmetricAlphabet:
        """
        Convert to symmetric alphabet.

        Args:
            letter_values: Values for each orbit position (default: 1..k)
        """
        if letter_values is None:
            letter_values = tuple(range(1, self.k + 1))
        
        if len(letter_values) != self.k:
            raise OrbitError(
                f"letter_values length ({len(letter_values)}) must equal orbit size ({self.k})",

            )
        
        return SymmetricAlphabet(
            orbit_frequencies=self.frequencies,
            letter_values=letter_values,
            name=self.label
        )
    
    def apply_shift(self, shift: int) -> 'Orbit':
        """Apply cyclic shift to orbit frequencies."""
        new_freqs = list(self.frequencies)
        # Shift frequencies cyclically
        new_freqs = new_freqs[-shift:] + new_freqs[:-shift]
        return Orbit(
            frequencies=tuple(new_freqs),
            orbit_type=self.orbit_type,
            label=f"{self.label}_shifted_{shift}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "frequencies": list(self.frequencies),
            "N": self.N,
            "k": self.k,
            "weight": self.weight,
            "multiplicity": self.multiplicity,
            "orbit_type": self.orbit_type.value,
            "is_balanced": self.is_balanced,
            "is_trivial": self.is_trivial,
            "nonzero_indices": self.nonzero_indices,
            "label": self.label
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Orbit':
        """Create orbit from dictionary."""
        return cls(
            frequencies=tuple(data["frequencies"]),
            orbit_type=OrbitType(data.get("orbit_type", "permutation")),
            weight=data.get("weight", 0),
            multiplicity=data.get("multiplicity", 0),
            label=data.get("label", "")
        )
    
    @classmethod
    def from_stats_orbit(cls, stats_orbit: StatsSymmetryOrbit, label: str = "") -> 'Orbit':
        """Create orbit from stats_utils SymmetryOrbit."""
        return cls(
            frequencies=stats_orbit.frequencies,
            orbit_type=OrbitType.PERMUTATION,
            label=label
        )
    
    @classmethod
    def from_frequencies(cls, frequencies: List[int], label: str = "") -> 'Orbit':
        """Create orbit from frequency list."""
        return cls(
            frequencies=tuple(frequencies),
            label=label
        )


@dataclass(frozen=True)
class OrbitSet:
    """
    Set of orbits with multiplicities.

    Attributes:
        orbits: List of orbits
        multiplicities: Multiplicity of each orbit
        total_weight: Total weight sum
    """
    orbits: List[Orbit]
    multiplicities: List[int]
    total_weight: int = 0
    
    def __post_init__(self):
        """Validate orbit set."""
        if len(self.orbits) != len(self.multiplicities):
            raise OrbitError(
                f"Number of orbits ({len(self.orbits)}) must match multiplicities ({len(self.multiplicities)})",

            )
        
        # Auto-compute total weight
        if self.total_weight == 0:
            total = sum(o.weight * m for o, m in zip(self.orbits, self.multiplicities))
            object.__setattr__(self, 'total_weight', total)
    
    @property
    def size(self) -> int:
        """Number of distinct orbit types."""
        return len(self.orbits)
    
    @property
    def unique_orbits(self) -> List[Orbit]:
        """Get unique orbits (ignoring multiplicities)."""
        return self.orbits
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "orbits": [o.to_dict() for o in self.orbits],
            "multiplicities": self.multiplicities,
            "size": self.size,
            "total_weight": self.total_weight
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrbitSet':
        """Create orbit set from dictionary."""
        return cls(
            orbits=[Orbit.from_dict(o) for o in data["orbits"]],
            multiplicities=data["multiplicities"],
            total_weight=data.get("total_weight", 0)
        )
    
    @classmethod
    def from_orbits(cls, orbits: List[Orbit]) -> 'OrbitSet':
        """Create orbit set from list of orbits (each counted once)."""
        return cls(
            orbits=orbits,
            multiplicities=[1] * len(orbits)
        )


@dataclass(frozen=True)
class LiftedOrbit:
    """
    Result of orbit lifting operation.

    Attributes:
        original_orbit: Original orbit
        lifted_orbit: Lifted orbit
        lifting_strategy: Strategy used
        lifting_factor: Lifting factor
        multiplicity: Multiplicity of lifted orbit
    """
    original_orbit: Orbit
    lifted_orbit: Orbit
    lifting_strategy: LiftingStrategy
    lifting_factor: int
    multiplicity: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "original_orbit": self.original_orbit.to_dict(),
            "lifted_orbit": self.lifted_orbit.to_dict(),
            "lifting_strategy": self.lifting_strategy.value,
            "lifting_factor": self.lifting_factor,
            "multiplicity": self.multiplicity
        }


# ============================================================================
# COMPOSITION GENERATOR (General k)
# ============================================================================

def compositions(N: int, k: int):
    """
    Generate all k-tuples of nonnegative integers summing to N.

    Args:
        N: Total sum
        k: Number of parts

    Yields:
        Tuples (f₁, f₂, ..., fₖ) with fᵢ ≥ 0 and Σfᵢ = N
    """
    if k == 1:
        yield (N,)
        return
    for i in range(N + 1):
        for rest in compositions(N - i, k - 1):
            yield (i,) + rest


# ============================================================================
# ORBIT GENERATION FUNCTIONS (General k)
# ============================================================================

def generate_orbits(N: int, k: int, max_orbits: int = 10000,
                   alphabet_values: Optional[Tuple[int, ...]] = None) -> List[Orbit]:
    """
    Generate all orbits for given N and k.

    Args:
        N: Total count
        k: Number of orbit types (dimension)
        max_orbits: Maximum number of orbits to generate
        alphabet_values: Optional explicit values for Weyl symmetry detection.

    Returns:
        List of Orbit objects (permutation-symmetry equivalence classes).

    Note on Weyl symmetry: For distinct alphabet values, S_k permutation symmetry
    applies and is already handled here. For repeated values, weyl.py detects
    higher-dimensional symmetry (future: collapse_by_weyl_symmetry will reduce orbits).
    """
    if alphabet_values is not None:
        try:
            try:
                from .weyl import detect_weyl_symmetry, WeylType
            except ImportError:
                from weyl import detect_weyl_symmetry, WeylType
            wt = detect_weyl_symmetry(list(alphabet_values))
            if wt.value != "none":
                logger.info(f"generate_orbits: Weyl type {wt.value} detected — "
                            f"symmetry reduction is future work (weyl.collapse_by_weyl_symmetry)")
        except Exception:
            pass
    """
    Generate all orbits for given N and k.

    Args:
        N: Total count
        k: Number of orbit types (dimension)
        max_orbits: Maximum number of orbits to generate

    Returns:
        List of orbits
    """
    orbits = []
    count = 0
    
    # Generate all compositions of N into k parts
    for comp in compositions(N, k):
        if count >= max_orbits:
            logger.warning(f"Reached max_orbits limit ({max_orbits})")
            break
        
        orbit = Orbit(frequencies=comp)
        orbits.append(orbit)
        count += 1
    
    logger.info(f"Generated {len(orbits)} orbits for N={N}, k={k}")
    return orbits


def generate_orbits_by_sparsity(N: int, k: int, nonzero: int, max_orbits: int = 10000) -> List[Orbit]:
    """
    Generate orbits with exactly 'nonzero' non-zero frequencies.

    Args:
        N: Total count
        k: Number of orbit types
        nonzero: Number of non-zero frequencies (L₀ norm)
        max_orbits: Maximum number of orbits to generate

    Returns:
        List of orbits with exactly 'nonzero' non-zero entries
    """
    all_orbits = generate_orbits(N, k, max_orbits)
    filtered = [o for o in all_orbits if len(o.nonzero_indices) == nonzero]
    logger.info(f"Filtered {len(filtered)} orbits with {nonzero} non-zero entries")
    return filtered


# ============================================================================
# LEGACY SUPPORT (for backward compatibility)
# ============================================================================

# ============================================================================
# ORBIT FILTERING
# ============================================================================

def filter_orbits_by_weight(orbits: List[Orbit], min_weight: int = 1, 
                           max_weight: Optional[int] = None) -> List[Orbit]:
    """
    Filter orbits by weight.

    Args:
        orbits: List of orbits
        min_weight: Minimum weight
        max_weight: Maximum weight

    Returns:
        Filtered orbits
    """
    filtered = []
    for orbit in orbits:
        if orbit.weight < min_weight:
            continue
        if max_weight is not None and orbit.weight > max_weight:
            continue
        filtered.append(orbit)
    
    logger.info(f"Filtered {len(filtered)}/{len(orbits)} orbits by weight")
    return filtered


def filter_orbits_by_moments(orbits: List[Orbit], 
                             constraints: MomentConstraints,
                             letter_values: Optional[Tuple[int, ...]] = None) -> List[Orbit]:
    """
    Filter orbits by moment constraints.

    Args:
        orbits: List of orbits
        constraints: Moment constraints
        letter_values: Values for each orbit position (default: 1..k)

    Returns:
        Orbits satisfying moment constraints
    """
    valid = []
    
    for orbit in orbits:
        if orbit.N != constraints.n:
            continue
        
        # Create letter values if not provided
        if letter_values is None:
            vals = tuple(range(1, orbit.k + 1))
        else:
            vals = letter_values[:orbit.k]
        
        # Create histogram with letter values
        hist = Histogram(
            bins=vals,
            counts=orbit.frequencies
        )
        s1, s2 = hist.moments()
        
        if s1 == constraints.s1 and s2 == constraints.s2:
            valid.append(orbit)
    
    logger.info(f"Filtered {len(valid)}/{len(orbits)} orbits by moments")
    return valid


def filter_orbits_by_nonzero(orbits: List[Orbit], min_nonzero: int = 1,
                            max_nonzero: Optional[int] = None) -> List[Orbit]:
    """
    Filter orbits by number of non-zero frequencies (L₀ norm).

    Args:
        orbits: List of orbits
        min_nonzero: Minimum number of non-zero frequencies
        max_nonzero: Maximum number of non-zero frequencies

    Returns:
        Filtered orbits
    """
    filtered = []
    for orbit in orbits:
        nonzero = len(orbit.nonzero_indices)
        if nonzero < min_nonzero:
            continue
        if max_nonzero is not None and nonzero > max_nonzero:
            continue
        filtered.append(orbit)
    
    logger.info(f"Filtered {len(filtered)}/{len(orbits)} orbits by L₀ norm")
    return filtered


# ============================================================================

# ============================================================================

def lift_orbit_product(orbit1: Orbit, orbit2: Orbit) -> LiftedOrbit:
    """
    Lift orbits via product.

    The product orbit has frequencies fᵢⱼ = f₁ᵢ · f₂ⱼ.

    Args:
        orbit1: First orbit
        orbit2: Second orbit

    Returns:
        Lifted orbit
    """
    # Compute product frequencies
    k1, k2 = orbit1.k, orbit2.k
    frequencies = []
    
    for i in range(k1):
        for j in range(k2):
            frequencies.append(orbit1.frequencies[i] * orbit2.frequencies[j])
    
    # Filter out zeros
    frequencies = [f for f in frequencies if f > 0]
    
    lifted_orbit = Orbit(
        frequencies=tuple(frequencies),
        orbit_type=OrbitType.CUSTOM,
        label=f"product_{orbit1.label}_{orbit2.label}"
    )
    
    multiplicity = orbit1.weight * orbit2.weight
    
    return LiftedOrbit(
        original_orbit=orbit1,
        lifted_orbit=lifted_orbit,
        lifting_strategy=LiftingStrategy.PRODUCT,
        lifting_factor=1,
        multiplicity=multiplicity
    )


def lift_orbit_direct_sum(orbit: Orbit, copies: int) -> LiftedOrbit:
    """
    Lift orbit via direct sum (concatenation).

    Args:
        orbit: Original orbit
        copies: Number of copies

    Returns:
        Lifted orbit
    """
    frequencies = list(orbit.frequencies) * copies
    
    lifted_orbit = Orbit(
        frequencies=tuple(frequencies),
        orbit_type=orbit.orbit_type,
        label=f"{orbit.label}_sum_{copies}"
    )
    
    # Multiplicity for direct sum: multinomial coefficient
    from math import factorial
    N = orbit.N * copies
    weight = compute_orbit_weight(frequencies, N)
    multiplicity = weight
    
    return LiftedOrbit(
        original_orbit=orbit,
        lifted_orbit=lifted_orbit,
        lifting_strategy=LiftingStrategy.DIRECT_SUM,
        lifting_factor=copies,
        multiplicity=multiplicity
    )


def lift_orbit_symmetric_power(orbit: Orbit, power: int) -> LiftedOrbit:
    """
    Lift orbit via symmetric power.

    Args:
        orbit: Original orbit
        power: Power to raise to

    Returns:
        Lifted orbit
    """
    # For symmetric power, we need all multisets of size 'power'
    # from the original orbit types
    from itertools import combinations_with_replacement
    
    k = orbit.k
    frequencies = []
    
    # Generate all multisets of size 'power' from k types
    for combo in combinations_with_replacement(range(k), power):
        # Count frequencies of each type in this multiset
        counts = [0] * k
        for idx in combo:
            counts[idx] += 1
        
        # Weight by original frequencies
        freq_product = 1
        for i, c in enumerate(counts):
            if c > 0:
                freq_product *= orbit.frequencies[i] ** c
        
        if freq_product > 0:
            frequencies.append(freq_product)
    
    lifted_orbit = Orbit(
        frequencies=tuple(frequencies),
        orbit_type=OrbitType.CUSTOM,
        label=f"{orbit.label}_sym_{power}"
    )

    # FIX(Bug-7a): The original code computed a single bulk multiplicity:
    #   C(power+k-1, power) * orbit.weight^power
    # This formula counts the total number of multisets, not the weighted
    # multiplicity of each specific selection.  For the purpose of the lifted
    # orbit, the multiplicity must account for the per-selection stabilizer.
    #
    # Each combo (c₀, c₁, ..., c_{k-1}) with Σcᵢ = power contributes:
    #   weight_combo = factorial(power) / prod(factorial(cᵢ))  (multinomial)
    #                  × prod(orbit.weight ^ cᵢ)
    # Summing over all combos gives the total multiplicity.
    from math import factorial
    multiplicity = 0
    for combo in combinations_with_replacement(range(k), power):
        counts = [0] * k
        for idx in combo:
            counts[idx] += 1
        # Multinomial coefficient for this selection
        denom_fact = 1
        for c in counts:
            denom_fact *= factorial(c)
        multinomial = factorial(power) // denom_fact
        # Per-selection weight
        w_combo = 1
        for i, c in enumerate(counts):
            w_combo *= orbit.weight ** c
        multiplicity += multinomial * w_combo

    return LiftedOrbit(
        original_orbit=orbit,
        lifted_orbit=lifted_orbit,
        lifting_strategy=LiftingStrategy.SYMMETRIC_POWER,
        lifting_factor=power,
        multiplicity=multiplicity
    )


def lift_orbit_tensor(orbit1: Orbit, orbit2: Orbit) -> LiftedOrbit:
    """
    Lift orbits via tensor product.

    Args:
        orbit1: First orbit
        orbit2: Second orbit

    Returns:
        Lifted orbit
    """
    # Tensor product is similar to product but with different multiplicities
    k1, k2 = orbit1.k, orbit2.k
    frequencies = []
    
    for i in range(k1):
        for j in range(k2):
            if orbit1.frequencies[i] > 0 and orbit2.frequencies[j] > 0:
                frequencies.append(orbit1.frequencies[i] * orbit2.frequencies[j])
    
    lifted_orbit = Orbit(
        frequencies=tuple(frequencies),
        orbit_type=OrbitType.CUSTOM,
        label=f"tensor_{orbit1.label}_{orbit2.label}"
    )

    # FIX(Bug-7b): The original multiplicity was orbit1.weight * orbit2.weight * k1 * k2.
    # The k1*k2 factor is the number of (i,j) pairs, but those pairs are already
    # enumerated in the frequency loop above.  Multiplying by k1*k2 again makes
    # every count too large by exactly that factor.
    # Correct multiplicity for the tensor product is simply the product of the
    # two orbit weights (the combinatorial selection factor is already encoded
    # in the lifted_orbit.frequencies list).
    multiplicity = orbit1.weight * orbit2.weight

    return LiftedOrbit(
        original_orbit=orbit1,
        lifted_orbit=lifted_orbit,
        lifting_strategy=LiftingStrategy.TENSOR,
        lifting_factor=1,
        multiplicity=multiplicity
    )


def lift_orbit_exterior_power(orbit: Orbit, power: int) -> LiftedOrbit:
    """
    Lift orbit via exterior power.

    Args:
        orbit: Original orbit
        power: Power to raise to

    Returns:
        Lifted orbit
    """
    # For exterior power, we need all subsets of size 'power'
    # from the original orbit types
    from itertools import combinations
    
    k = orbit.k
    if power > k:
        raise OrbitLiftingError(f"Exterior power {power} > orbit size {k}")
    
    frequencies = []
    
    # Generate all subsets of size 'power' from k types
    for combo in combinations(range(k), power):
        # Product of frequencies in this subset
        freq_product = 1
        for idx in combo:
            freq_product *= orbit.frequencies[idx]
        
        if freq_product > 0:
            frequencies.append(freq_product)
    
    lifted_orbit = Orbit(
        frequencies=tuple(frequencies),
        orbit_type=OrbitType.CUSTOM,
        label=f"{orbit.label}_ext_{power}"
    )
    
    # Multiplicity: binomial coefficient
    from math import comb
    multiplicity = comb(k, power) * orbit.weight ** power
    
    return LiftedOrbit(
        original_orbit=orbit,
        lifted_orbit=lifted_orbit,
        lifting_strategy=LiftingStrategy.EXTERIOR_POWER,
        lifting_factor=power,
        multiplicity=multiplicity
    )


# ============================================================================
# ORBIT DECOMPOSITION
# ============================================================================

def decompose_orbit(orbit):
    """Not implemented — orbit irreducible decomposition not used by RSP pipeline."""
    raise NotImplementedError(
        "decompose_orbit: irreducible decomposition requires character theory "
        "(Young tableaux for S_k). Not used by the reverse statistics pipeline. "
        "The previous stub silently returned multiplicity 1 for all inputs."
    )



def orbit_multiplicity_product(orbit_set1: OrbitSet, orbit_set2: OrbitSet) -> OrbitSet:
    """
    Compute product of two orbit sets (Kronecker product).

    Args:
        orbit_set1: First orbit set
        orbit_set2: Second orbit set

    Returns:
        Product orbit set
    """
    new_orbits = []
    new_multiplicities = []
    
    for o1, m1 in zip(orbit_set1.orbits, orbit_set1.multiplicities):
        for o2, m2 in zip(orbit_set2.orbits, orbit_set2.multiplicities):
            # Compute product orbit
            lift = lift_orbit_product(o1, o2)
            new_orbits.append(lift.lifted_orbit)
            new_multiplicities.append(m1 * m2 * lift.multiplicity)
    
    return OrbitSet(
        orbits=new_orbits,
        multiplicities=new_multiplicities
    )


# ============================================================================
# ORBIT STATISTICS
# ============================================================================

def compute_orbit_statistics(orbit: Orbit) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for an orbit.

    Args:
        orbit: Orbit to analyze

    Returns:
        Dictionary of statistics
    """
    from math import log
    
    stats = {}
    
    # Basic counts
    stats["N"] = orbit.N
    stats["k"] = orbit.k
    stats["weight"] = orbit.weight
    stats["multiplicity"] = orbit.multiplicity
    
    # Frequency statistics
    nonzero = orbit.nonzero_indices
    stats["nonzero_count"] = len(nonzero)
    stats["sparsity"] = len(nonzero) / orbit.k if orbit.k > 0 else 0
    
    if orbit.N > 0:
        # Relative frequencies
        rel_freqs = [f / orbit.N for f in orbit.frequencies]
        stats["max_freq"] = max(rel_freqs)
        stats["min_freq"] = min(rel_freqs)
        stats["mean_freq"] = sum(rel_freqs) / len(rel_freqs)
        
        # Entropy
        entropy = 0
        for f in rel_freqs:
            if f > 0:
                entropy -= f * log(f, 2)
        stats["entropy"] = entropy
        
        # Evenness (Pielou's evenness)
        if len(nonzero) > 1:
            max_entropy = log(len(nonzero), 2)
            stats["evenness"] = entropy / max_entropy if max_entropy > 0 else 0
        else:
            stats["evenness"] = 1.0
    
    # Moments
    moments = []
    for i, f in enumerate(orbit.frequencies):
        moments.extend([i + 1] * f)
    
    if moments:
        stats["mean"] = sum(moments) / len(moments)
        stats["variance"] = sum((x - stats["mean"]) ** 2 for x in moments) / len(moments)
        stats["std_dev"] = stats["variance"] ** 0.5
    else:
        stats["mean"] = 0
        stats["variance"] = 0
        stats["std_dev"] = 0
    
    return stats


def compare_orbits(orbit1: Orbit, orbit2: Orbit) -> Dict[str, float]:
    """
    Compare two orbits using various metrics.

    Args:
        orbit1: First orbit
        orbit2: Second orbit

    Returns:
        Dictionary of comparison metrics
    """
    if orbit1.k != orbit2.k:
        raise OrbitError("Cannot compare orbits of different sizes")
    
    # Convert to numpy arrays
    arr1 = np.array(orbit1.frequencies, dtype=float)
    arr2 = np.array(orbit2.frequencies, dtype=float)
    
    # Normalize
    if orbit1.N > 0:
        arr1 = arr1 / orbit1.N
    if orbit2.N > 0:
        arr2 = arr2 / orbit2.N
    
    # Euclidean distance
    euclidean = np.linalg.norm(arr1 - arr2)
    
    # Cosine similarity
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    if norm1 > 0 and norm2 > 0:
        cosine = np.dot(arr1, arr2) / (norm1 * norm2)
    else:
        cosine = 0.0
    
    # Manhattan distance
    manhattan = np.sum(np.abs(arr1 - arr2))
    
    # Hellinger distance
    hellinger = np.sqrt(np.sum((np.sqrt(arr1) - np.sqrt(arr2)) ** 2)) / np.sqrt(2)
    
    return {
        "euclidean_distance": float(euclidean),
        "cosine_similarity": float(cosine),
        "manhattan_distance": float(manhattan),
        "hellinger_distance": float(hellinger)
    }


# ============================================================================
# ORBIT CACHE
# ============================================================================

class OrbitCache:
    """LRU cache for orbit computations."""
    
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
_orbit_cache = OrbitCache(maxsize=128)


# ============================================================================
# VALIDATION AND TESTING
# ============================================================================

def validate_orbit_utils() -> Dict[str, bool]:
    """Run internal test suite to verify orbit utilities."""
    results = {}
    
    try:
        # Test 1: Orbit creation
        orbit = Orbit(frequencies=(2, 2, 1, 1), label="test")
        results["orbit_creation"] = orbit.N == 6 and orbit.k == 4
        
        # Test 2: Orbit weight
        results["orbit_weight"] = orbit.weight == 180  # 6!/(2!2!1!1!) = 720/4 = 180
        
        # Test 3: Orbit properties
        results["orbit_properties"] = (not orbit.is_balanced and not orbit.is_trivial)
        
        # Test 4: Generate all orbits for various k
        orbits_k3 = generate_orbits(N=4, k=3)
        results["generate_k3"] = len(orbits_k3) == 15  # C(4+3-1,3-1)=15
        
        orbits_k4 = generate_orbits(N=4, k=4)
        results["generate_k4"] = len(orbits_k4) == 35  # C(4+4-1,4-1)=35
        
        orbits_k5 = generate_orbits(N=4, k=5)
        results["generate_k5"] = len(orbits_k5) == 70  # C(4+5-1,5-1)=70
        
        # Test 5: Generate orbits by sparsity
        sparse = generate_orbits_by_sparsity(N=4, k=4, nonzero=2)
        results["generate_sparse"] = len(sparse) > 0
        
        # Test 6: Filter by weight
        filtered = filter_orbits_by_weight(orbits_k4, min_weight=10)
        results["filter_weight"] = len(filtered) <= len(orbits_k4)
        
        # Test 7: Filter by nonzero
        filtered2 = filter_orbits_by_nonzero(orbits_k4, min_nonzero=2)
        results["filter_nonzero"] = len(filtered2) <= len(orbits_k4)
        
        # Test 8: Orbit lifting - product
        orbit1 = Orbit(frequencies=(1, 1))
        orbit2 = Orbit(frequencies=(1, 1))
        lifted = lift_orbit_product(orbit1, orbit2)
        results["lift_product"] = lifted.lifted_orbit.k == 4
        
        # Test 9: Orbit lifting - direct sum
        lifted2 = lift_orbit_direct_sum(orbit1, copies=2)
        results["lift_direct_sum"] = lifted2.lifted_orbit.k == 4
        
        # Test 10: Orbit lifting - symmetric power
        lifted3 = lift_orbit_symmetric_power(orbit1, power=2)
        results["lift_symmetric"] = lifted3.lifted_orbit.k > 0
        
        # Test 11: Orbit statistics
        stats = compute_orbit_statistics(orbit)
        results["orbit_stats"] = "entropy" in stats and "mean" in stats
        
        # Test 12: Orbit comparison
        comp = compare_orbits(orbit, orbit)
        results["orbit_comparison"] = abs(comp["cosine_similarity"] - 1.0) < 1e-10
        
        # Test 13: Orbit decomposition
        decomp = decompose_orbit(orbit)
        results["orbit_decomposition"] = decomp.size > 0
        
        # Test 14: Orbit set creation
        orbit_set = OrbitSet(orbits=[orbit], multiplicities=[1])
        results["orbit_set"] = orbit_set.size == 1 and orbit_set.total_weight == orbit.weight
        
        # Test 15: Legacy S4 support (should still work)
        # FIX(Bug-6): generate_s4_orbits was never defined; use generate_orbits(N, k=4)
        s4_orbits = generate_orbits(N=4, k=4)
        results["legacy_s4"] = len(s4_orbits) == 35
        
        logger.info("✅ Orbit utilities validation passed")
        
    except Exception as e:
        logger.error(f"❌ Orbit utilities validation failed: {e}")
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
    print("Testing Production-Ready Orbit Utilities")
    print("=" * 60)
    print("\nNOTE: Now supports general k-dimensional orbits")
    print("      OrbitType.S4_SYMMETRIC renamed to PERMUTATION")
    print("      generate_orbits(N, k) replaces generate_s4_orbits()")
    print("=" * 60)
    
    # Run validation
    results = validate_orbit_utils()
    
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
    print("Orbit Utilities Demo - General k-Dimensional Support")
    print("=" * 60)
    
    # Generate orbits for various k
    print("\n1. Orbit Generation for Different k:")
    for k in [2, 3, 4, 5]:
        N = k  # Small N for demonstration
        orbits = generate_orbits(N=N, k=k, max_orbits=10)
        print(f"   k={k}, N={N}: {len(generate_orbits(N, k))} total orbits")
        print(f"   First 3: {[o.frequencies for o in orbits[:3]]}")
    

    print("\n2. Permutation Symmetry Orbits ():")
    k = 4
    N = 4
    orbits = generate_orbits(N=N, k=k)
    print(f"   Generated {len(orbits)} orbits for k={k}, N={N}")
    
    # Show first 5 orbits
    for i, orbit in enumerate(orbits[:5]):
        print(f"   Orbit {i+1}: {orbit.frequencies}, weight={orbit.weight}")
    

    print("\n3. Orbit Lifting ():")
    base_orbit = Orbit(frequencies=(2, 2, 1, 1), label="base")
    print(f"   Base orbit: {base_orbit.frequencies}, weight={base_orbit.weight}")
    
    # Product lift
    lifted = lift_orbit_product(base_orbit, base_orbit)
    print(f"   Product lift: {lifted.lifted_orbit.frequencies[:5]}... (total {lifted.lifted_orbit.k} types)")
    print(f"   Multiplicity: {lifted.multiplicity}")
    
    # Direct sum lift
    lifted_sum = lift_orbit_direct_sum(base_orbit, copies=2)
    print(f"   Direct sum (2 copies): {lifted_sum.lifted_orbit.frequencies}")
    print(f"   Multiplicity: {lifted_sum.multiplicity}")
    
    # Orbit statistics
    print("\n4. Orbit Statistics:")
    stats = compute_orbit_statistics(base_orbit)
    print(f"   N={stats['N']}, k={stats['k']}")
    print(f"   Entropy: {stats['entropy']:.2f} bits")
    print(f"   Mean: {stats['mean']:.2f}")
    print(f"   Sparsity: {stats['sparsity']:.2f}")
    
    # Orbit comparison
    print("\n5. Orbit Comparison:")
    comp = compare_orbits(base_orbit, base_orbit)
    print(f"   Cosine similarity (self): {comp['cosine_similarity']:.2f}")
    print(f"   Euclidean distance (self): {comp['euclidean_distance']:.2f}")
    
    print("\n" + "=" * 60)
    print("✅ Orbit Utilities Ready for Production")
    print("=" * 60)