"""
Enumerations Module for Reverse Statistics Pipeline
Provides state space enumeration, combinatorial analysis, and pattern discovery.

Phases: 28 (State Space), 29 (Combinatorial), 30 (Pattern Discovery)
"""

from .exceptions import ReverseStatsError
import numpy as np
from numpy.random import default_rng
import math
from fractions import Fraction
from typing import List, Tuple, Dict, Any, Optional, Union, Set, Callable, Iterator, Generator
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from functools import lru_cache
import itertools
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

# ============================================================================
# EXCEPTIONS
# ============================================================================

class EnumerationError(ReverseStatsError):
    """Base exception for enumeration operations."""
    def __init__(self, message: str):
        self.message = message

        super().__init__(message)


class StateSpaceError(EnumerationError):
    """Raised when state space enumeration fails."""
    def __init__(self, message: str):
        super().__init__(message)


class CombinatorialError(EnumerationError):
    """Raised when combinatorial analysis fails."""
    def __init__(self, message: str):
        super().__init__(message)


class PatternError(EnumerationError):
    """Raised when pattern discovery fails."""
    def __init__(self, message: str):
        super().__init__(message)


# ============================================================================
# ENUMS
# ============================================================================

class EnumerationMethod(Enum):
    """Methods for state space enumeration."""
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    DIAGONAL = "diagonal"
    LEXICOGRAPHIC = "lexicographic"
    GREY_CODE = "grey_code"
    RANDOM = "random"


class CombinationType(Enum):
    """Types of combinatorial structures."""
    PERMUTATIONS = "permutations"
    COMBINATIONS = "combinations"
    POWER_SET = "power_set"
    CARTESIAN_PRODUCT = "cartesian_product"
    PARTITIONS = "partitions"
    COMPOSITIONS = "compositions"


class PatternType(Enum):
    """Types of patterns to discover."""
    FREQUENT = "frequent"
    SEQUENTIAL = "sequential"
    PERIODIC = "periodic"
    SYMMETRIC = "symmetric"
    RECURRENT = "recurrent"
    ANOMALOUS = "anomalous"


# ============================================================================
# CONFIGURATION
# ============================================================================

def get_enumeration_config() -> Dict[str, Any]:
    """Get enumeration-specific configuration with sane defaults."""
    config = {
        "max_state_space_size": 1_000_000,      # Maximum states to enumerate
        "max_combination_length": 20,            # Maximum length for combinations
        "enable_bit_optimization": True,         # Use bit operations for performance
        "enable_pruning": True,                   # Enable search space pruning
        "pattern_min_support": 0.01,              # Minimum support for frequent patterns
        "pattern_max_length": 10,                  # Maximum pattern length
        "parallel_enumerate": False,               # Use parallel enumeration
        "cache_results": True,                      # Cache enumeration results
    }
    
    # Try to integrate with global config
    try:
        from .config import get_config
        global_config = get_config()
        pipeline_config = getattr(global_config, 'pipeline_config', global_config)
        config["max_state_space_size"] = getattr(pipeline_config, "max_state_space_size", 
                                                 config["max_state_space_size"])
        config["max_combination_length"] = getattr(pipeline_config, "max_combination_length", 
                                                   config["max_combination_length"])
    except (ImportError, AttributeError):
        pass
    
    return config


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class State:
    """
    Representation of a state in the state space.

    Attributes:
        values: Tuple of values representing the state
        metadata: Additional state metadata
    """
    values: Tuple[Any, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)
    _hash: int = field(init=False, repr=False)
    
    def __post_init__(self):
        """Compute hash value."""
        self._hash = hash(self.values)
    
    def __hash__(self) -> int:
        return self._hash
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return False
        return self.values == other.values
    
    def __lt__(self, other: 'State') -> bool:
        """Lexicographic ordering."""
        return self.values < other.values
    
    def __str__(self) -> str:
        return f"State{self.values}"
    
    def __repr__(self) -> str:
        return f"State({self.values})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "values": list(self.values),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'State':
        """Create State from dictionary."""
        return cls(
            values=tuple(data["values"]),
            metadata=data.get("metadata", {})
        )


@dataclass
class StateSpace:
    """
    State space representation.

    Attributes:
        dimensions: Dimensions of each variable
        states: Set of all states
        variable_names: Names of variables
        transitions: Optional transition graph
        metadata: Additional metadata
    """
    dimensions: List[int]
    states: Set[State] = field(default_factory=set)
    variable_names: Optional[List[str]] = None
    transitions: Optional[Dict[State, List[State]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize state space."""
        if self.variable_names is None:
            self.variable_names = [f"X{i}" for i in range(len(self.dimensions))]
        
        if self.transitions is None:
            self.transitions = {}
    
    @property
    def size(self) -> int:
        """Number of states in the state space."""
        return len(self.states)
    
    @property
    def dimension(self) -> int:
        """Number of variables."""
        return len(self.dimensions)
    
    @property
    def total_possible_states(self) -> int:
        """Total number of possible states (product of dimensions)."""
        return int(np.prod(self.dimensions))
    
    def contains(self, state: Union[State, Tuple[Any, ...]]) -> bool:
        """Check if state is in the state space."""
        if isinstance(state, tuple):
            state = State(state)
        return state in self.states
    
    def add_state(self, state: Union[State, Tuple[Any, ...]]) -> None:
        """Add a state to the state space."""
        if isinstance(state, tuple):
            state = State(state)
        self.states.add(state)
    
    def add_transition(self, from_state: State, to_state: State) -> None:
        """Add a transition between states."""
        if from_state not in self.states:
            self.add_state(from_state)
        if to_state not in self.states:
            self.add_state(to_state)
        
        if from_state not in self.transitions:
            self.transitions[from_state] = []
        self.transitions[from_state].append(to_state)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "states": [list(s.values) for s in sorted(self.states)],
            "dimensions": self.dimensions,
            "variable_names": self.variable_names,
            "size": self.size,
            "total_possible": self.total_possible_states,
            "coverage": self.size / self.total_possible_states if self.total_possible_states > 0 else 0,
            "metadata": self.metadata
        }
    
    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "State Space Summary",
            "=" * 40,
            f"Dimensions: {self.dimensions}",
            f"Variables: {self.variable_names}",
            f"States: {self.size} / {self.total_possible_states}",
        ]
        if self.total_possible_states > 0:
            lines.append(f"Coverage: {self.size/self.total_possible_states:.2%}")
        lines.append(f"Transitions: {sum(len(v) for v in self.transitions.values()) if self.transitions else 0}")
        return "\n".join(lines)


@dataclass
class Combination:
    """
    Combinatorial structure.

    Attributes:
        elements: Elements in the combination
        type: Type of combination
        weight: Optional weight value
        metadata: Additional metadata
    """
    elements: Tuple[Any, ...]
    type: CombinationType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize combination."""
        if self.type == CombinationType.COMBINATIONS:
            self.elements = tuple(sorted(self.elements))
    
    def __hash__(self) -> int:
        return hash((self.elements, self.type.value))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Combination):
            return False
        return self.elements == other.elements and self.type == other.type
    
    def __len__(self) -> int:
        return len(self.elements)
    
    def __iter__(self) -> Iterator[Any]:
        return iter(self.elements)
    
    def __contains__(self, item: Any) -> bool:
        return item in self.elements
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "elements": list(self.elements),
            "type": self.type.value,
            "weight": self.weight,
            "metadata": self.metadata
        }


@dataclass
class Pattern:
    """
    Discovered pattern.

    Attributes:
        pattern: The pattern itself
        type: Type of pattern
        support: Support count/frequency
        confidence: Confidence measure
        lift: Lift value (for association rules)
        locations: Where pattern occurs
        metadata: Additional metadata
    """
    pattern: Tuple[Any, ...]
    type: PatternType
    support: float
    confidence: float = 1.0
    lift: float = 1.0
    locations: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash((self.pattern, self.type.value))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pattern):
            return False
        return self.pattern == other.pattern and self.type == other.type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern": list(self.pattern),
            "type": self.type.value,
            "support": self.support,
            "confidence": self.confidence,
            "lift": self.lift,
            "locations": self.locations,
            "metadata": self.metadata
        }


@dataclass
class EnumerationResult:
    """
    Result of enumeration operation.

    Attributes:
        method: Enumeration method used
        items: List of enumerated items
        time_taken: Time taken in seconds
        memory_used: Approximate memory used
        metadata: Additional metadata
    """
    method: EnumerationMethod
    items: List[Any]
    time_taken: float
    memory_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def count(self) -> int:
        """Number of items enumerated."""
        return len(self.items)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method.value,
            "count": self.count,
            "items_sample": self.items[:10] if self.count > 10 else self.items,
            "total_items": self.count,
            "time_taken": self.time_taken,
            "memory_used": self.memory_used,
            "metadata": self.metadata
        }


# ============================================================================
# UTILITY FUNCTIONS (from math_utils)
# ============================================================================

def binomial_coefficient(n: int, k: int) -> int:
    """Compute binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for i in range(1, k + 1):
        result = result * (n - k + i) // i
    return result


# ============================================================================

# ============================================================================

def enumerate_state_space(dimensions: List[int],
                         method: EnumerationMethod = EnumerationMethod.LEXICOGRAPHIC,
                         max_states: Optional[int] = None,
                         **kwargs) -> StateSpace:
    """
    Enumerate all possible states in a state space.

    Args:
        dimensions: List of dimensions for each variable
        method: Enumeration method to use
        max_states: Maximum number of states to enumerate
        **kwargs: Additional method-specific parameters

    Returns:
        State space containing all enumerated states
    """
    config = get_enumeration_config()
    max_states = max_states or config.get("max_state_space_size", 1_000_000)
    
    # Calculate total possible states
    total_possible = int(np.prod(dimensions))
    if total_possible > max_states:
        logger.warning(f"State space size {total_possible} exceeds max_states {max_states}")
    
    state_space = StateSpace(dimensions=dimensions)
    
    start_time = time.time()
    
    # Dispatch to appropriate enumeration method
    enumerators = {
        EnumerationMethod.LEXICOGRAPHIC: _enumerate_lexicographic,
        EnumerationMethod.BREADTH_FIRST: _enumerate_breadth_first,
        EnumerationMethod.DEPTH_FIRST: _enumerate_depth_first,
        EnumerationMethod.DIAGONAL: _enumerate_diagonal,
        EnumerationMethod.GREY_CODE: _enumerate_grey_code,
        EnumerationMethod.RANDOM: _enumerate_random,
    }
    
    enumerator = enumerators.get(method)
    if enumerator is None:
        raise StateSpaceError(f"Unsupported enumeration method: {method}")
    
    enumerator(state_space, dimensions, max_states, **kwargs)
    
    elapsed = time.time() - start_time
    logger.info(f"Enumerated {state_space.size} states in {elapsed:.3f}s using {method.value}")
    
    state_space.metadata.update({
        "enumeration_method": method.value,
        "enumeration_time": elapsed
    })
    
    return state_space


def _enumerate_lexicographic(state_space: StateSpace,
                            dimensions: List[int],
                            max_states: int,
                            **kwargs) -> None:
    """Enumerate states in lexicographic order."""
    ranges = [range(dim) for dim in dimensions]
    
    count = 0
    for values in itertools.product(*ranges):
        if count >= max_states:
            break
        state_space.add_state(State(values))
        count += 1


def _enumerate_breadth_first(state_space: StateSpace,
                            dimensions: List[int],
                            max_states: int,
                            start_state: Optional[Tuple[int, ...]] = None,
                            **kwargs) -> None:
    """Enumerate states using breadth-first search."""
    from collections import deque
    
    if start_state is None:
        start_state = tuple([0] * len(dimensions))
    
    start = State(start_state)
    queue = deque([start])
    visited = {start}
    
    while queue and len(visited) < max_states:
        current = queue.popleft()
        state_space.add_state(current)
        
        # Generate neighbors
        for i in range(len(dimensions)):
            for delta in [-1, 1]:
                new_values = list(current.values)
                new_values[i] += delta
                if 0 <= new_values[i] < dimensions[i]:
                    new_state = State(tuple(new_values))
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append(new_state)


def _enumerate_depth_first(state_space: StateSpace,
                          dimensions: List[int],
                          max_states: int,
                          start_state: Optional[Tuple[int, ...]] = None,
                          **kwargs) -> None:
    """
    Enumerate states using depth-first search.


    on large state spaces (>1000 states would exceed Python's default call stack).
    """
    if start_state is None:
        start_state = tuple([0] * len(dimensions))

    visited: Set[State] = set()
    stack = [State(start_state)]

    while stack and len(visited) < max_states:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        state_space.add_state(current)

        # Push unvisited neighbours
        for i in range(len(dimensions)):
            for delta in (-1, 1):
                new_values = list(current.values)
                new_values[i] += delta
                if 0 <= new_values[i] < dimensions[i]:
                    neighbor = State(tuple(new_values))
                    if neighbor not in visited:
                        stack.append(neighbor)


def _enumerate_diagonal(state_space: StateSpace,
                       dimensions: List[int],
                       max_states: int,
                       **kwargs) -> None:
    """Enumerate states along diagonals (optimized for 2D spaces)."""
    if len(dimensions) != 2:
        # Fallback to lexicographic for non-2D
        _enumerate_lexicographic(state_space, dimensions, max_states, **kwargs)
        return
    
    n, m = dimensions
    count = 0
    
    # Enumerate by sum of coordinates
    for s in range(n + m - 1):
        for i in range(max(0, s - m + 1), min(n, s + 1)):
            if count >= max_states:
                return
            state_space.add_state(State((i, s - i)))
            count += 1


def _enumerate_grey_code(state_space: StateSpace,
                        dimensions: List[int],
                        max_states: int,
                        **kwargs) -> None:
    """Enumerate using Gray code."""
    if len(dimensions) == 1:
        # Single dimension Gray code
        n = dimensions[0]
        for i in range(min(n, max_states)):
            grey = i ^ (i >> 1)
            if grey < n:
                state_space.add_state(State((grey,)))
    
    elif len(dimensions) == 2:
        # 2D Gray code product
        n, m = dimensions
        count = 0
        for i in range(n):
            grey_i = i ^ (i >> 1)
            for j in range(m):
                if count >= max_states:
                    return
                grey_j = j ^ (j >> 1)
                state_space.add_state(State((grey_i % n, grey_j % m)))
                count += 1
    else:

        # Callers relying on Hamiltonian-path ordering will now get a visible warning.
        logger.warning(
            f"_enumerate_grey_code: Gray code not implemented for {len(dimensions)}D "
            f"(only 1D and 2D supported). Falling back to lexicographic order."
        )
        _enumerate_lexicographic(state_space, dimensions, max_states, **kwargs)


def _enumerate_random(state_space: StateSpace,
                     dimensions: List[int],
                     max_states: int,
                     random_seed: Optional[int] = None,
                     **kwargs) -> None:
    """Enumerate states randomly with replacement."""
    if random_seed is not None:
        rng = default_rng(random_seed)
    else:
        rng = default_rng()
    
    n_states = min(int(np.prod(dimensions)), max_states)
    for _ in range(n_states):
        values = tuple(rng.integers(0, dim) for dim in dimensions)
        state_space.add_state(State(values))


def sample_state_space(dimensions: List[int],
                      n_samples: int,
                      method: str = "uniform",
                      random_seed: Optional[int] = None) -> List[State]:
    """
    Sample states from the state space.

    Args:
        dimensions: Dimensions of each variable
        n_samples: Number of samples
        method: Sampling method ('uniform', 'weighted')
        random_seed: Random seed

    Returns:
        List of sampled states
    """
    if random_seed is not None:
        rng = default_rng(random_seed)
    else:
        rng = default_rng()
    
    samples = []
    
    if method == "uniform":
        for _ in range(n_samples):
            values = tuple(rng.integers(0, dim) for dim in dimensions)
            samples.append(State(values))
    
    elif method == "weighted":

        # materialises the entire state space in memory — up to 10^10 tuples for
        # large problems.  Instead use reservoir sampling with a single generator
        # pass: each state is seen once, edge states get 2x acceptance weight.
        # This is O(n_samples) memory regardless of state-space size.
        total_dims = dimensions  # alias for clarity

        # Two-pass reservoir sampling weighted by is_edge factor (weight 2 vs 1).
        # We use Algorithm A-Chao (weighted reservoir sampling).
        reservoir = []
        reservoir_weights = []
        W = 0.0  # running total weight
        gen = itertools.product(*[range(d) for d in total_dims])
        for values in gen:
            is_edge = any(v == 0 or v == d - 1 for v, d in zip(values, total_dims))
            w = 2.0 if is_edge else 1.0
            W += w
            if len(reservoir) < n_samples:
                reservoir.append(values)
                reservoir_weights.append(w)
            else:
                p = w / W
                if rng.random() < p:
                    # FIX(Bug-6): The original code evicted a uniformly random
                    # slot (rng.integers(0, len(reservoir))), destroying the
                    # intended weighted distribution.  Algorithm A-Chao requires
                    # evicting item j with probability proportional to *its own
                    # weight* (so lighter existing items are displaced first).
                    # We select j ∝ (1/reservoir_weights[j]) = ∝ inverse weight.
                    inv_weights = [1.0 / rw for rw in reservoir_weights]
                    total_inv = sum(inv_weights)
                    probs = [iw / total_inv for iw in inv_weights]
                    cumulative = 0.0
                    r = rng.random()
                    j = len(reservoir) - 1  # default to last slot
                    for idx, prob in enumerate(probs):
                        cumulative += prob
                        if r < cumulative:
                            j = idx
                            break
                    reservoir[j] = values
                    reservoir_weights[j] = w
        for values in reservoir:
            samples.append(State(tuple(values)))
    
    else:
        # Default to uniform
        return sample_state_space(dimensions, n_samples, "uniform", random_seed)
    
    return samples


# ============================================================================

# ============================================================================

def generate_combinations(elements: List[Any],
                         k: int,
                         with_replacement: bool = False,
                         max_results: Optional[int] = None) -> Generator[Combination, None, None]:
    """
    Generate all combinations of k elements.

    Args:
        elements: List of elements to combine
        k: Size of combinations
        with_replacement: Whether to allow replacement
        max_results: Maximum number of results to yield

    Yields:
        Combinations
    """
    config = get_enumeration_config()
    max_results = max_results or config.get("max_state_space_size", 1_000_000)
    
    count = 0
    iterator = (itertools.combinations_with_replacement(elements, k) if with_replacement
                else itertools.combinations(elements, k))
    
    for combo in iterator:
        if count >= max_results:
            break
        yield Combination(elements=combo, type=CombinationType.COMBINATIONS)
        count += 1


def generate_permutations(elements: List[Any],
                         k: Optional[int] = None,
                         max_results: Optional[int] = None) -> Generator[Combination, None, None]:
    """
    Generate all permutations of k elements.

    Args:
        elements: List of elements to permute
        k: Size of permutations (None for all elements)
        max_results: Maximum number of results to yield

    Yields:
        Permutations
    """
    config = get_enumeration_config()
    max_results = max_results or config.get("max_state_space_size", 1_000_000)
    
    if k is None:
        k = len(elements)
    
    count = 0
    for perm in itertools.permutations(elements, k):
        if count >= max_results:
            break
        yield Combination(elements=perm, type=CombinationType.PERMUTATIONS)
        count += 1


def generate_power_set(elements: List[Any],
                      min_size: int = 0,
                      max_size: Optional[int] = None,
                      max_results: Optional[int] = None) -> Generator[Combination, None, None]:
    """
    Generate all subsets (power set) of elements.

    Args:
        elements: List of elements
        min_size: Minimum subset size
        max_size: Maximum subset size
        max_results: Maximum number of results to yield

    Yields:
        Subsets
    """
    config = get_enumeration_config()
    max_results = max_results or config.get("max_state_space_size", 1_000_000)
    
    if max_size is None:
        max_size = len(elements)
    
    count = 0
    n = len(elements)
    
    # Use binary representation for power set
    for i in range(1 << n):
        subset = [elements[j] for j in range(n) if i & (1 << j)]
        
        if min_size <= len(subset) <= max_size:
            if count >= max_results:
                break
            yield Combination(elements=tuple(subset), type=CombinationType.POWER_SET)
            count += 1


def generate_cartesian_product(sets: List[List[Any]],
                              max_results: Optional[int] = None) -> Generator[Combination, None, None]:
    """
    Generate Cartesian product of multiple sets.

    Args:
        sets: List of sets to take product of
        max_results: Maximum number of results to yield

    Yields:
        Cartesian product elements
    """
    config = get_enumeration_config()
    max_results = max_results or config.get("max_state_space_size", 1_000_000)
    
    count = 0
    for product in itertools.product(*sets):
        if count >= max_results:
            break
        yield Combination(elements=product, type=CombinationType.CARTESIAN_PRODUCT)
        count += 1


def generate_partitions(n: int,
                       max_parts: Optional[int] = None,
                       max_results: Optional[int] = None) -> Generator[Combination, None, None]:
    """
    Generate integer partitions of n.

    Args:
        n: Integer to partition
        max_parts: Maximum number of parts
        max_results: Maximum number of results to yield

    Yields:
        Partitions as combinations
    """
    config = get_enumeration_config()
    max_results = max_results or config.get("max_state_space_size", 1_000_000)
    
    def _partitions(n: int, max_part: int, current: List[int]):
        if n == 0:
            yield tuple(current)
            return
        
        for i in range(min(max_part, n), 0, -1):
            if max_parts is None or len(current) < max_parts:
                yield from _partitions(n - i, i, current + [i])
    
    count = 0
    for partition in _partitions(n, n, []):
        if count >= max_results:
            break
        yield Combination(elements=partition, type=CombinationType.PARTITIONS)
        count += 1


def count_combinations(n: int, k: int, with_replacement: bool = False) -> int:
    """
    Count number of combinations.

    Args:
        n: Number of elements
        k: Combination size
        with_replacement: Whether to allow replacement

    Returns:
        Number of combinations
    """
    if with_replacement:
        return binomial_coefficient(n + k - 1, k)
    return binomial_coefficient(n, k)


def combinatorial_explosion(n: int, k: int, threshold: int = 1_000_000) -> bool:
    """
    Check if combinatorial explosion is likely.

    Args:
        n: Number of elements
        k: Combination size
        threshold: Threshold for explosion

    Returns:
        True if likely to explode, False otherwise
    """
    return count_combinations(n, k) > threshold


# ============================================================================

# ============================================================================

def find_frequent_patterns(sequences: List[List[Any]],
                          min_support: float = 0.01,
                          max_length: int = 5,
                          **kwargs) -> List[Pattern]:
    """

    Performance note (Bug 17): Naive O(C·S·L) Apriori implementation.
    For >10,000 sequences, replace with FP-Tree or Trie for practical speed.
    Find frequent patterns in sequences using Apriori algorithm.

    Args:
        sequences: List of sequences
        min_support: Minimum support threshold
        max_length: Maximum pattern length
        **kwargs: Additional parameters

    Returns:
        List of frequent patterns
    """
    config = get_enumeration_config()
    min_support = min_support or config.get("pattern_min_support", 0.01)
    max_length = max_length or config.get("pattern_max_length", 10)
    
    n_sequences = len(sequences)
    min_count = max(1, int(min_support * n_sequences))
    
    # Count individual items
    item_counts = Counter()
    for seq in sequences:
        item_counts.update(set(seq))
    
    # Filter frequent items
    frequent_items = {item for item, count in item_counts.items() if count >= min_count}
    
    if not frequent_items:
        return []
    
    patterns = []
    
    # Level 1: Single items
    level_patterns = []
    for item in frequent_items:
        pattern = Pattern(
            pattern=(item,),
            type=PatternType.FREQUENT,
            support=item_counts[item] / n_sequences
        )
        patterns.append(pattern)
        level_patterns.append(pattern)
    
    # Higher levels
    k = 2
    while level_patterns and k <= max_length:
        # Generate candidates
        candidates = _generate_candidates(level_patterns, k)
        
        # Count candidates
        candidate_counts = Counter()
        for seq in sequences:
            seq_set = set(seq)
            for candidate in candidates:
                if all(item in seq_set for item in candidate):
                    candidate_counts[candidate] += 1
        
        # Filter candidates
        level_patterns = []
        for candidate, count in candidate_counts.items():
            if count >= min_count:
                pattern = Pattern(
                    pattern=candidate,
                    type=PatternType.FREQUENT,
                    support=count / n_sequences
                )
                patterns.append(pattern)
                level_patterns.append(pattern)
        
        k += 1
    
    return patterns


def _generate_candidates(prev_patterns: List[Pattern], k: int) -> List[Tuple[Any, ...]]:
    """Generate candidate patterns of length k."""
    candidates = []
    
    # Get patterns of length k-1
    prev_patterns = [p for p in prev_patterns if len(p.pattern) == k - 1]
    
    for i in range(len(prev_patterns)):
        for j in range(i + 1, len(prev_patterns)):
            p1 = prev_patterns[i].pattern
            p2 = prev_patterns[j].pattern
            
            if p1[:-1] == p2[:-1]:
                candidates.append(p1 + (p2[-1],))
    
    return candidates


def find_sequential_patterns(sequences: List[List[Any]],
                            min_support: float = 0.01,
                            max_gap: int = 2,
                            **kwargs) -> List[Pattern]:
    """
    Find sequential patterns (order matters).

    Args:
        sequences: List of sequences
        min_support: Minimum support threshold
        max_gap: Maximum gap between elements
        **kwargs: Additional parameters

    Returns:
        List of sequential patterns
    """
    config = get_enumeration_config()
    min_support = min_support or config.get("pattern_min_support", 0.01)
    
    n_sequences = len(sequences)
    min_count = max(1, int(min_support * n_sequences))
    
    patterns = []
    pattern_dict = {}  # Deduplicate patterns
    
    for seq_idx, seq in enumerate(sequences):
        seq_len = len(seq)
        
        for start in range(seq_len):
            for length in range(1, min(max_gap + 2, seq_len - start + 1)):
                pattern_seq = tuple(seq[start:start + length])
                
                # Count occurrences
                support_count = 0
                for other_seq in sequences:
                    if _is_subsequence(pattern_seq, other_seq):
                        support_count += 1
                
                if support_count >= min_count:
                    if pattern_seq not in pattern_dict:
                        pattern_dict[pattern_seq] = Pattern(
                            pattern=pattern_seq,
                            type=PatternType.SEQUENTIAL,
                            support=support_count / n_sequences,
                            locations=[]
                        )
                    
                    if seq_idx not in pattern_dict[pattern_seq].locations:
                        pattern_dict[pattern_seq].locations.append(seq_idx)
    
    return list(pattern_dict.values())


def _is_subsequence(pattern: Tuple[Any, ...], sequence: List[Any]) -> bool:
    """Check if pattern is a subsequence of sequence."""
    if not pattern:
        return True
    
    pattern_idx = 0
    for item in sequence:
        if item == pattern[pattern_idx]:
            pattern_idx += 1
            if pattern_idx == len(pattern):
                return True
    
    return False


def find_periodic_patterns(sequence: List[Any],
                          min_period: int = 1,
                          max_period: int = 10,
                          **kwargs) -> List[Pattern]:
    """
    Find periodic patterns in a sequence.

    Args:
        sequence: Input sequence
        min_period: Minimum period to consider
        max_period: Maximum period to consider
        **kwargs: Additional parameters

    Returns:
        List of periodic patterns
    """
    n = len(sequence)
    patterns = []
    pattern_dict = {}
    
    for period in range(min_period, min(max_period + 1, n // 2 + 1)):
        # Check if sequence is fully periodic
        is_periodic = all(sequence[i] == sequence[i % period] for i in range(period, n))
        
        if is_periodic:
            pattern_tuple = tuple(sequence[:period])
            if pattern_tuple not in pattern_dict:
                pattern_dict[pattern_tuple] = Pattern(
                    pattern=pattern_tuple,
                    type=PatternType.PERIODIC,
                    support=n / period,
                    metadata={"period": period}
                )
        else:
            # Find partial periodic patterns
            for start in range(period):
                pattern = [sequence[i] for i in range(start, n, period) if i < n]
                if len(pattern) >= 2:
                    pattern_tuple = tuple(pattern)
                    if pattern_tuple not in pattern_dict:
                        pattern_dict[pattern_tuple] = Pattern(
                            pattern=pattern_tuple,
                            type=PatternType.PERIODIC,
                            support=len(pattern),
                            metadata={"period": period, "phase": start}
                        )
    
    return list(pattern_dict.values())


def find_symmetric_patterns(sequence: List[Any], **kwargs) -> List[Pattern]:
    """
    Find symmetric patterns in a sequence.

    Args:
        sequence: Input sequence
        **kwargs: Additional parameters

    Returns:
        List of symmetric patterns
    """
    n = len(sequence)
    patterns = []
    pattern_dict = {}
    
    # Palindrome patterns
    for length in range(2, min(n, 10)):
        for start in range(n - length + 1):
            subseq = sequence[start:start + length]
            if subseq == subseq[::-1]:
                pattern_tuple = tuple(subseq)
                if pattern_tuple not in pattern_dict:
                    pattern_dict[pattern_tuple] = Pattern(
                        pattern=pattern_tuple,
                        type=PatternType.SYMMETRIC,
                        support=1.0,
                        locations=[start],
                        metadata={"length": length, "type": "palindrome"}
                    )
    
    # Reflective symmetry
    for center in range(n):
        radius = 0
        while (center - radius >= 0 and center + radius < n and 
               sequence[center - radius] == sequence[center + radius]):
            radius += 1
        
        if radius > 1:
            pattern_tuple = tuple(sequence[center - radius + 1:center + radius])
            if pattern_tuple and pattern_tuple not in pattern_dict:
                pattern_dict[pattern_tuple] = Pattern(
                    pattern=pattern_tuple,
                    type=PatternType.SYMMETRIC,
                    support=radius,
                    locations=[center],
                    metadata={"center": center, "radius": radius, "type": "reflective"}
                )
    
    return list(pattern_dict.values())


def find_anomalous_patterns(sequence: List[Any],
                           baseline: Optional[List[Any]] = None,
                           threshold: float = 2.0,
                           **kwargs) -> List[Pattern]:
    """
    Find anomalous patterns in a sequence.

    Args:
        sequence: Input sequence
        baseline: Baseline sequence for comparison
        threshold: Anomaly threshold (multiples of expected probability)
        **kwargs: Additional parameters

    Returns:
        List of anomalous patterns
    """
    if baseline is None:
        baseline = sequence
    
    # Compute frequency distribution
    freq_dist = Counter(baseline)
    total = len(baseline)
    
    if total == 0:
        return []
    
    expected_probs = {item: count / total for item, count in freq_dist.items()}
    
    patterns = []
    window_sizes = [1, 2, 3]
    
    for window_size in window_sizes:
        for i in range(len(sequence) - window_size + 1):
            window = tuple(sequence[i:i + window_size])
            
            # Expected probability (product of individual probabilities)
            expected_prob = 1.0
            for item in window:
                expected_prob *= expected_probs.get(item, 1e-10)
            
            # Observed frequency
            observed_count = 0
            for j in range(len(sequence) - window_size + 1):
                if tuple(sequence[j:j + window_size]) == window:
                    observed_count += 1
            
            observed_prob = observed_count / (len(sequence) - window_size + 1)
            
            # Check if anomalous
            if expected_prob > 0 and observed_prob > threshold * expected_prob:
                pattern = Pattern(
                    pattern=window,
                    type=PatternType.ANOMALOUS,
                    support=observed_count,
                    metadata={
                        "expected_prob": expected_prob,
                        "observed_prob": observed_prob,
                        "ratio": observed_prob / expected_prob
                    }
                )
                patterns.append(pattern)
    
    return patterns


# ============================================================================
# STATE SPACE UTILITIES
# ============================================================================

def state_distance(state1: State, state2: State, metric: str = "euclidean") -> float:
    """
    Compute distance between two states.

    Args:
        state1: First state
        state2: Second state
        metric: Distance metric ('euclidean', 'manhattan', 'chebyshev', 'hamming')

    Returns:
        Distance value
    """
    if len(state1.values) != len(state2.values):
        raise StateSpaceError("States must have same dimension")
    
    metrics = {
        "euclidean": lambda a, b: math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))),
        "manhattan": lambda a, b: sum(abs(x - y) for x, y in zip(a, b)),
        "chebyshev": lambda a, b: max(abs(x - y) for x, y in zip(a, b)),
        "hamming": lambda a, b: sum(1 for x, y in zip(a, b) if x != y),
    }
    
    if metric not in metrics:
        raise StateSpaceError(f"Unknown distance metric: {metric}")
    
    return metrics[metric](state1.values, state2.values)


def state_space_graph(state_space: StateSpace,
                     neighbor_fn: Optional[Callable[[State], List[State]]] = None) -> Dict[State, List[State]]:
    """
    Build graph representation of state space.

    Args:
        state_space: State space
        neighbor_fn: Function to generate neighbors

    Returns:
        Adjacency list representation
    """
    if neighbor_fn is None:
        def neighbor_fn(state: State) -> List[State]:
            neighbors = []
            dims = state_space.dimensions
            values = list(state.values)
            
            for i in range(len(dims)):
                for delta in [-1, 1]:
                    new_values = values.copy()
                    new_values[i] += delta
                    if 0 <= new_values[i] < dims[i]:
                        new_state = State(tuple(new_values))
                        if new_state in state_space.states:
                            neighbors.append(new_state)
            
            return neighbors
    
    return {state: neighbor_fn(state) for state in state_space.states}


def compress_state_space(state_space: StateSpace,
                        equivalence_fn: Callable[[State, State], bool]) -> StateSpace:
    """
    Compress state space by merging equivalent states.

    Args:
        state_space: Original state space
        equivalence_fn: Function to test state equivalence

    Returns:
        Compressed state space
    """
    # Group states into equivalence classes
    equivalence_classes = []
    for state in sorted(state_space.states):
        found = False
        for eq_class in equivalence_classes:
            if equivalence_fn(state, eq_class[0]):
                eq_class.append(state)
                found = True
                break
        
        if not found:
            equivalence_classes.append([state])
    
    # Create representative states
    compressed_states = [eq_class[0] for eq_class in equivalence_classes]
    state_to_rep = {state: rep for eq_class, rep in zip(equivalence_classes, compressed_states)
                   for state in eq_class}
    
    # Build compressed state space
    compressed = StateSpace(
        dimensions=state_space.dimensions,
        states=set(compressed_states),
        variable_names=state_space.variable_names,
        metadata={
            **state_space.metadata,
            "compressed": True,
            "original_size": state_space.size
        }
    )
    
    # Compress transitions if present
    if state_space.transitions:
        compressed.transitions = {}
        for from_state, to_states in state_space.transitions.items():
            from_rep = state_to_rep[from_state]
            to_reps = {state_to_rep[s] for s in to_states}
            
            if from_rep not in compressed.transitions:
                compressed.transitions[from_rep] = []
            
            for to_rep in to_reps:
                if to_rep not in compressed.transitions[from_rep]:
                    compressed.transitions[from_rep].append(to_rep)
    
    return compressed


# ============================================================================
# CACHING
# ============================================================================

class EnumerationCache:
    """LRU cache for enumeration results."""
    
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
            # Remove oldest item
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
_enumeration_cache = EnumerationCache(maxsize=16)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_enumeration_utils() -> Dict[str, bool]:
    """Run internal test suite to verify enumeration utilities."""
    results = {}
    
    try:
        # Test 1: State creation
        s1 = State((1, 2, 3))
        s2 = State((1, 2, 3))
        s3 = State((4, 5, 6))
        results["state_creation"] = (s1 == s2 and s1 != s3 and hash(s1) == hash(s2))
        
        # Test 2: State space enumeration
        dimensions = [3, 3]
        state_space = enumerate_state_space(dimensions, EnumerationMethod.LEXICOGRAPHIC)
        results["state_space"] = state_space.size == 9
        
        # Test 3: Combinations
        elements = [1, 2, 3, 4]
        combos = list(generate_combinations(elements, 2))
        results["combinations"] = len(combos) == 6
        
        # Test 4: Permutations
        perms = list(generate_permutations([1, 2, 3], 2))
        results["permutations"] = len(perms) == 6
        
        # Test 5: Power set
        power_set = list(generate_power_set([1, 2, 3]))
        results["power_set"] = len(power_set) == 8  # 2^3 = 8 subsets
        
        # Test 6: Partitions
        partitions = list(generate_partitions(4))
        results["partitions"] = len(partitions) == 5  # 4, 3+1, 2+2, 2+1+1, 1+1+1+1
        
        # Test 7: State distance
        dist = state_distance(State((1, 2)), State((4, 6)), "euclidean")
        results["state_distance"] = abs(dist - 5.0) < 0.001  # sqrt(3^2 + 4^2) = 5
        
        logger.info("✅ Enumeration utilities validation passed")
    except Exception as e:
        logger.error(f"❌ Enumeration utilities validation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        results["validation_error"] = str(e)
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("Enumerations Module - Validation & Demo")
    print("=" * 60)
    
    # Run validation
    results = validate_enumeration_utils()
    
    print("\nValidation Results:")
    print("-" * 40)
    success = sum(1 for v in results.values() if v is True)
    total = len(results)
    for key, value in results.items():
        if key == "validation_error":
            print(f"❌ {key}: {value}")
        else:
            print(f"{'✅' if value else '❌'} {key}: {'PASSED' if value else 'FAILED'}")
    print("-" * 40)
    print(f"Overall: {success}/{total-1 if 'validation_error' in results else total} tests passed")
    
    if "validation_error" in results:
        sys.exit(1)
    
    # Demo
    print("\n" + "=" * 60)
    print("Quick Demo")
    print("=" * 60)
    
    # State space
    dimensions = [3, 4]
    ss = enumerate_state_space(dimensions, EnumerationMethod.LEXICOGRAPHIC, max_states=10)
    print(f"\nState space {dimensions}: {ss.size} states (showing first 5)")
    for state in sorted(ss.states)[:5]:
        print(f"  {state.values}")
    
    # Combinations
    elements = ['A', 'B', 'C', 'D']
    print(f"\nCombinations of {elements} taken 2:")
    combos = list(generate_combinations(elements, 2, max_results=5))
    for c in combos[:3]:
        print(f"  {c.elements}")
    print(f"  ... ({len(combos)} total)")
    
    print("\n" + "=" * 60)
    print("✅ Enumerations Module Ready")
    print("=" * 60)