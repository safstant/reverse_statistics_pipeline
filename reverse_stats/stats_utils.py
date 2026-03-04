import numpy as np
from .exceptions import ReverseStatsError
import math
from collections import Counter
from fractions import Fraction
from typing import List, Tuple, Dict, Any, Optional, Callable, Dict, Optional
from dataclasses import dataclass
from functools import lru_cache, wraps
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# EXCEPTIONS
# ============================================================================

class StatsError(ReverseStatsError):
    """Base exception for statistical operations."""
    def __init__(self, message: str):
        self.message = message

        super().__init__(message)


class MomentError(StatsError):
    """Raised for moment constraint violations."""
    pass


class SymmetryError(StatsError):
    """Raised for symmetry-related errors."""
    pass


class HistogramError(StatsError):
    """Raised for histogram constraint violations."""
    pass


class EnumerationError(StatsError):
    """Raised when enumeration limits are exceeded."""
    pass


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class Histogram:
    """Immutable histogram representation for L₀ sparsity constraints."""
    bins: Tuple[int, ...]
    counts: Tuple[int, ...]

    def __post_init__(self):
        if len(self.bins) != len(self.counts):
            raise HistogramError(
                f"Bins ({len(self.bins)}) and counts ({len(self.counts)}) length mismatch",

            )
        if any(c < 0 for c in self.counts):
            raise HistogramError("Histogram counts must be non-negative")

    @property
    def total(self) -> int:
        return sum(self.counts)

    @property
    def num_bins(self) -> int:
        return sum(1 for c in self.counts if c > 0)

    @property
    def support(self) -> List[int]:
        return [b for b, c in zip(self.bins, self.counts) if c > 0]

    def moments(self) -> Tuple[Fraction, Fraction]:
        s1 = Fraction(0)
        s2 = Fraction(0)
        for b, c in zip(self.bins, self.counts):
            s1 += b * c
            s2 += b * b * c
        return s1, s2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bins": list(self.bins),
            "counts": list(self.counts),
            "total": self.total,
            "num_bins": self.num_bins,
            "support": self.support,
            "moments": [str(m) for m in self.moments()]
        }


@dataclass(frozen=True)
class MomentConstraints:
    """Immutable moment constraints container."""
    s1: Fraction
    s2: Fraction
    n: int

    def __post_init__(self):
        if self.n <= 0:
            raise MomentError(f"n must be positive, got {self.n}")
        if self.s2 <= 0:
            raise MomentError(f"S₂ must be positive, got {self.s2}")

        # Cauchy-Schwarz: N·S₂ ≥ S₁²
        if self.n * self.s2 < self.s1 * self.s1:
            raise MomentError(
                f"S₂={self.s2} < S₁²/N={self.s1 * self.s1 / self.n}",

            )

    @property
    def mean(self) -> Fraction:
        return self.s1 / self.n

    @property
    def variance_bound(self) -> Fraction:
        return (self.s2 / self.n) - (self.s1 * self.s1 / (self.n * self.n))

    def is_feasible(self, values: List[int]) -> bool:
        if len(values) != self.n:
            return False
        s1_val = sum(values)
        s2_val = sum(x * x for x in values)
        return (s1_val == self.s1 and s2_val == self.s2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "S1": str(self.s1),
            "S2": str(self.s2),
            "N": self.n,
            "mean": str(self.mean),
            "variance_bound": str(self.variance_bound)
        }


@dataclass(frozen=True)
class SymmetryOrbit:
    """
    Permutation symmetry orbit representation.
    Generalised: supports any alphabet size k (not k=4 only).
    frequencies: (f₁, f₂, ..., f_k), Σfⱼ = N.


    and raised SymmetryError for k≠4. Now accepts Tuple[int,...] for any k.
    """
    frequencies: Tuple[int, ...]  # f₁, ..., f_k for alphabet of size k

    def __post_init__(self):
        if len(self.frequencies) < 1:
            raise SymmetryError(
                f"SymmetryOrbit requires at least 1 frequency, got 0",

            )
        if any(f < 0 for f in self.frequencies):
            raise SymmetryError("Frequencies must be ≥ 0")

    @property
    def k(self) -> int:
        """Alphabet size (number of frequency bins)."""
        return len(self.frequencies)

    @property
    def N(self) -> int:
        return sum(self.frequencies)

    @property
    def weight(self) -> int:
        """Orbit weight: N! / ∏fⱼ!

        This is the multinomial coefficient counting ordered arrangements of the
        frequencies.  It is the correct orbit size when *all* alphabet bin
        values are distinct (the common case in the RSP pipeline, which always
        uses ``alphabet_values = range(min_val, max_val + 1)``).

        For alphabets with *repeated* values use :meth:`alphabet_aware_weight`.
        """
        from math import factorial
        result = factorial(self.N)
        for f in self.frequencies:
            result //= factorial(f)
        return result

    def alphabet_aware_weight(self, alphabet_values: Tuple[int, ...]) -> int:
        """Stabilizer-corrected orbit weight for arbitrary alphabets.

        The multinomial coefficient ``N! / ∏fⱼ!`` counts ordered arrangements
        of the full sequence.  When two or more alphabet bins share the same
        *value* (e.g. ``alphabet = [1, 2, 1, 3]`` with positions 0 and 2 both
        having value 1), swapping the frequencies at those positions produces a
        *different* frequency vector that corresponds to the *same* multiset of
        values.  Without correction the total multiset count is inflated by the
        stabilizer size.

        The stabilizer of f under the value-based symmetry group consists of
        all permutations of bin positions that (a) map each bin to a bin with
        the same value AND (b) leave the frequency vector f invariant.  Its
        size is the product of the factorials of the *frequency-run lengths*
        within each group of equal-valued bins.

        Formally, group bins by alphabet value.  For each group g with bins
        having identical value v, let m_g(c) = number of bins in g that have
        frequency c.  Then

            |Stab(f)| = ∏_g  ∏_c  m_g(c)!

        and the corrected weight is ``weight() // |Stab(f)|``.

        For distinct alphabets (the pipeline default) every group has exactly
        one bin, all m_g(c) ≤ 1, and |Stab(f)| = 1, so this reduces to the
        plain multinomial weight.

        Args:
            alphabet_values: Tuple of alphabet bin values aligned with
                ``self.frequencies``, e.g. ``(1, 2, 3, 4)`` or ``(1, 2, 1, 3)``.

        Returns:
            Stabilizer-corrected orbit weight (always a positive integer).

        Raises:
            SymmetryError: if ``len(alphabet_values) != self.k``.
        """
        if len(alphabet_values) != self.k:
            raise SymmetryError(
                f"alphabet_values length {len(alphabet_values)} != k={self.k}"
            )
        from math import factorial
        from collections import Counter as _Counter

        # Plain multinomial
        raw = self.weight

        # Early exit for the common distinct-alphabet case
        if len(set(alphabet_values)) == self.k:
            return raw

        # Group bin indices by alphabet value
        # value_groups: {value: [freq at positions with this value]}
        value_groups: Dict[int, List[int]] = {}
        for pos, val in enumerate(alphabet_values):
            value_groups.setdefault(val, []).append(self.frequencies[pos])

        # Compute |Stab(f)|: for each value-group, count how many positions
        # in the group share the *same* frequency, then take the product of
        # those count-factorials.
        stab_size = 1
        for freq_list in value_groups.values():
            if len(freq_list) < 2:
                continue  # singleton group: stab contribution = 1
            freq_counts = _Counter(freq_list)
            for cnt in freq_counts.values():
                if cnt > 1:
                    stab_size *= factorial(cnt)

        return raw // stab_size

    @property
    def multiplicity(self) -> int:
        return self.weight

    @property
    def is_balanced(self) -> bool:
        return all(f == self.frequencies[0] for f in self.frequencies)

    def to_histogram(self, bin_values: Tuple[int, ...]) -> Histogram:

        if len(bin_values) != self.k:
            raise ValueError(
                f"bin_values length ({len(bin_values)}) must match orbit k ({self.k})"
            )
        bins = list(bin_values)
        counts = list(self.frequencies)
        sorted_pairs = sorted(zip(bins, counts))
        sorted_bins, sorted_counts = zip(*sorted_pairs)
        return Histogram(bins=tuple(sorted_bins), counts=tuple(sorted_counts))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frequencies": list(self.frequencies),
            "N": self.N,
            "weight": self.weight,
            "multiplicity": self.multiplicity,
            "is_balanced": self.is_balanced
        }


# ============================================================================
# DECORATORS
# ============================================================================

def validate_statistical(func: Callable) -> Callable:
    """Validate inputs: no NaN/Inf, non-negative."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        for arg in args:
            if isinstance(arg, (list, np.ndarray)):
                arr = np.asarray(arg)
                if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                    raise StatsError("Input contains NaN or Inf")
                if np.any(arr < 0):
                    raise StatsError("Input must be non-negative")
        return func(*args, **kwargs)
    return wrapper


def time_statistical(func: Callable) -> Callable:
    """Log execution time for slow operations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        if elapsed > 1.0:
            logger.warning(f"Slow statistical operation {func.__name__}: {elapsed:.3f}s")
        return result
    return wrapper


# ============================================================================

# ============================================================================

def _generate_orbits_recursive(N: int, k: int, max_orbits: int) -> List[Tuple[int, ...]]:
    """Generate all compositions of N into exactly k non-negative parts."""
    if k == 1:
        return [(N,)]
    result = []
    for f in range(N + 1):
        for tail in _generate_orbits_recursive(N - f, k - 1, max_orbits):
            result.append((f,) + tail)
            if len(result) >= max_orbits:
                return result
    return result


@lru_cache(maxsize=1024)
def generate_orbits(N: int, k: int, max_orbits: int = 1000) -> List[SymmetryOrbit]:
    """
    General orbit generator: all compositions of N into k non-negative parts.
    Works for any alphabet size k.

    Args:
        N: Total count (sum of all frequencies)
        k: Alphabet size
        max_orbits: Maximum number of orbits to generate
    Returns:
        List of SymmetryOrbit objects
    """
    freq_tuples = _generate_orbits_recursive(N, k, max_orbits)
    if len(freq_tuples) >= max_orbits:
        raise EnumerationError(
            f"Number of orbits for N={N}, k={k} exceeds limit {max_orbits}",

        )
    orbits = [SymmetryOrbit(t) for t in freq_tuples]
    logger.info(f"Generated {len(orbits)} orbits for N={N}, k={k}")
    return orbits


@validate_statistical
def compute_orbit_statistics(
    orbit: SymmetryOrbit,
    bin_values: Tuple[int, ...]
) -> Dict[str, Any]:
    """Compute statistical properties of a symmetry orbit."""
    hist = orbit.to_histogram(bin_values)
    s1, s2 = hist.moments()
    total = hist.total
    mean = s1 / total if total > 0 else Fraction(0)
    variance = (s2 / total - mean * mean) if total > 0 else Fraction(0)
    return {
        "orbit": orbit.to_dict(),
        "histogram": hist.to_dict(),
        "moments": {
            "S1": str(s1),
            "S2": str(s2),
            "mean": str(mean),
            "variance": str(variance)
        },
        "sparsity": hist.num_bins,
        "multiplicity": orbit.multiplicity
    }


def filter_orbits_by_constraints(
    orbits: List[SymmetryOrbit],
    constraints: MomentConstraints,
    bin_values: Tuple[int, ...]
) -> List[SymmetryOrbit]:
    """Filter orbits that satisfy moment constraints."""
    valid = []
    for orbit in orbits:
        hist = orbit.to_histogram(bin_values)
        if hist.total != constraints.n:
            continue
        s1, s2 = hist.moments()
        if s1 == constraints.s1 and s2 == constraints.s2:
            valid.append(orbit)
    logger.info(f"Filtered {len(valid)}/{len(orbits)} orbits by constraints")
    return valid


# ============================================================================

# ============================================================================

@validate_statistical
def create_histogram(
    values: List[int],
    bins: Optional[List[int]] = None
) -> Histogram:
    """Create histogram from integer values."""
    if not values:
        return Histogram(bins=(), counts=())
    if bins is None:
        bins = sorted(set(values))
    else:
        bins = sorted(set(bins))
    counter = Counter(values)
    counts = [counter.get(b, 0) for b in bins]
    return Histogram(bins=tuple(bins), counts=tuple(counts))


@validate_statistical
def apply_l0_constraint(
    histogram: Histogram,
    max_bins: Optional[int] = None
) -> Histogram:
    """Apply L₀ sparsity constraint to histogram."""
    if max_bins is None:
        return histogram
    if histogram.num_bins > max_bins:
        raise HistogramError(
            f"Histogram has {histogram.num_bins} non-zero bins, "
            f"exceeds L₀ constraint max_bins={max_bins}",

        )
    return histogram


@time_statistical
def enumerate_histograms(
    constraints: MomentConstraints,
    max_bins: Optional[int] = None,
    enumeration_limit: int = 1_000_000
) -> List[Histogram]:
    """
    Enumerate all histograms satisfying moment and L₀ constraints.
    """
    N = constraints.n
    S1 = int(constraints.s1)
    S2 = int(constraints.s2)

    if N == 0:
        return [Histogram(bins=(), counts=())]

    # Estimate and guard
    estimated = estimate_histogram_count(N, S1, S2, max_bins)
    if estimated > enumeration_limit:
        raise EnumerationError(
            f"Estimated histogram count {estimated:,} exceeds limit {enumeration_limit:,}",

        )

    # FIX(Bug-5): The original code allocated dp[N+1][S1+1][S2+1] as a dense
    # 3D array of sets.  For N=720 with realistic S1/S2 this creates ~10^8+
    # cells (each a Python set object) even before any data is inserted,
    # consuming hundreds of GB of RAM.
    #
    # Fix: use a sparse dict keyed by (i, s1, s2) that only stores populated
    # cells.  This transforms worst-case memory from O(N·S1·S2) to O(|reachable
    # states|), which is bounded by the actual number of distinct histograms —
    # a quantity that is already guarded by the `estimate_histogram_count` check
    # above and the `enumeration_limit` parameter.
    dp: Dict[Tuple[int, int, int], set] = {}
    dp[(0, 0, 0)] = {Histogram(bins=(), counts=())}
    total_histograms = 1  # running count for early abort

    possible_bins = list(range(1, min(S1, int(math.sqrt(S2))) + 1))

    for i in range(N):
        for s1 in range(S1 + 1):
            for s2 in range(S2 + 1):
                cell = dp.get((i, s1, s2))
                if not cell:
                    continue
                for hist in cell:
                    for b in possible_bins:
                        ni = i + 1
                        ns1 = s1 + b
                        ns2 = s2 + b * b
                        if ni <= N and ns1 <= S1 and ns2 <= S2:
                            # Add one observation to bin b
                            if b in hist.bins:
                                idx = hist.bins.index(b)
                                new_counts = list(hist.counts)
                                new_counts[idx] += 1
                                new_hist = Histogram(
                                    bins=hist.bins,
                                    counts=tuple(new_counts)
                                )
                            else:
                                new_bins = tuple(sorted(hist.bins + (b,)))
                                new_counts = hist.counts + (1,)
                                # Reorder counts to match sorted bins
                                if new_bins[-1] == b:
                                    new_hist = Histogram(
                                        bins=new_bins,
                                        counts=new_counts
                                    )
                                else:
                                    mapping = dict(zip(hist.bins, hist.counts))
                                    mapping[b] = mapping.get(b, 0) + 1
                                    sorted_counts = tuple(mapping[x] for x in new_bins)
                                    new_hist = Histogram(
                                        bins=new_bins,
                                        counts=sorted_counts
                                    )
                            key = (ni, ns1, ns2)
                            if key not in dp:
                                dp[key] = set()
                                total_histograms += 1
                                if total_histograms > enumeration_limit:
                                    raise EnumerationError(
                                        f"Histogram count exceeded limit {enumeration_limit:,} "
                                        f"during DP enumeration."
                                    )
                            dp[key].add(new_hist)

    result = []
    final_cell = dp.get((N, S1, S2), set())
    if final_cell:
        for hist in final_cell:
            if max_bins is None or hist.num_bins <= max_bins:
                result.append(hist)

    logger.info(
        f"Enumerated {len(result)} unique histograms for "
        f"N={N}, S1={S1}, S2={S2}"
    )

    if len(result) >= enumeration_limit:
        raise EnumerationError(
            f"Reached enumeration limit {enumeration_limit:,}",

        )

    return result


def estimate_histogram_count(
    N: int,
    S1: int,
    S2: int,
    max_bins: Optional[int] = None
) -> int:
    """Rough upper bound for enumeration limit checking."""
    if max_bins is None:
        max_bins = N
    stars_bars = math.comb(N + max_bins - 1, max_bins - 1)
    estimate = stars_bars // (S1 + 1) // (S2 + 1)
    return max(1, estimate)


# ============================================================================
# MOMENT COMPUTATIONS AND CONSTRAINTS
# ============================================================================

@validate_statistical
def compute_moments(values: List[int]) -> Dict[str, Fraction]:
    """Compute exact first and second moments."""
    if not values:
        return {
            "N": 0,
            "S1": Fraction(0),
            "S2": Fraction(0),
            "mean": Fraction(0),
            "variance": Fraction(0)
        }
    N = len(values)
    S1 = sum(values)
    S2 = sum(x * x for x in values)
    mean = Fraction(S1, N)
    variance = Fraction(S2, N) - mean * mean
    return {
        "N": N,
        "S1": Fraction(S1),
        "S2": Fraction(S2),
        "mean": mean,
        "variance": variance
    }


@validate_statistical
def compute_moment_bounds(
    N: int,
    mean_bounds: Tuple[float, float],
    variance_bounds: Tuple[float, float]
) -> Dict[str, Any]:
    """Compute bounds on S₁ and S₂ from mean and variance bounds."""
    min_mean, max_mean = mean_bounds
    min_var, max_var = variance_bounds

    min_S1 = math.ceil(N * min_mean)
    max_S1 = math.floor(N * max_mean)

    min_S2 = math.ceil(N * (min_var + min_mean * min_mean))
    max_S2 = math.floor(N * (max_var + max_mean * max_mean))

    # Cauchy-Schwarz lower bound
    min_S2 = max(min_S2, (min_S1 * min_S1 + N - 1) // N)
    max_S2 = min(max_S2, max_S1 * max_S1)

    return {
        "N": N,
        "S1_bounds": (min_S1, max_S1),
        "S2_bounds": (min_S2, max_S2),
        "mean_bounds": mean_bounds,
        "variance_bounds": variance_bounds
    }


def verify_moment_constraints(
    values: List[int],
    constraints: MomentConstraints
) -> bool:
    """Check if values satisfy given moment constraints."""
    moments = compute_moments(values)
    return (
        len(values) == constraints.n and
        moments["S1"] == constraints.s1 and
        moments["S2"] == constraints.s2
    )


# ============================================================================

# ============================================================================

@validate_statistical
def statistical_lift(histogram: Histogram, target_N: int) -> List[Histogram]:
    """Lift histogram to higher N while preserving moments."""
    current_N = histogram.total
    if target_N < current_N:
        raise StatsError(
            f"Target N={target_N} must be ≥ current N={current_N}",

        )
    if target_N == current_N:
        return [histogram]

    lifted = []
    s1, s2 = histogram.moments()
    # Try to add zeros
    if s1.denominator == 1 and s2.denominator == 1:
        s1_int = s1.numerator
        s2_int = s2.numerator
        if (
            (s1_int * target_N) % current_N == 0 and
            (s2_int * target_N) % current_N == 0
        ):
            if 0 in histogram.bins:
                idx = histogram.bins.index(0)
                new_counts = list(histogram.counts)
                new_counts[idx] += target_N - current_N
                lifted.append(
                    Histogram(bins=histogram.bins, counts=tuple(new_counts))
                )
            else:
                new_bins = (0,) + histogram.bins
                new_counts = (target_N - current_N,) + histogram.counts
                lifted.append(Histogram(bins=new_bins, counts=new_counts))
    return lifted


@validate_statistical
def statistical_projection(histogram: Histogram, target_N: int) -> List[Histogram]:
    """Project histogram to lower N while preserving moments."""
    current_N = histogram.total
    if target_N > current_N:
        raise StatsError(
            f"Target N={target_N} must be ≤ current N={current_N}",

        )
    if target_N == current_N:
        return [histogram]

    scale = Fraction(target_N, current_N)
    new_counts = []
    for c in histogram.counts:
        scaled = c * scale
        if scaled.denominator != 1:
            return []   # cannot scale exactly
        new_counts.append(scaled.numerator)
    return [Histogram(bins=histogram.bins, counts=tuple(new_counts))]


# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

class StatsCache:
    """LRU cache for statistical computations."""
    def __init__(self, maxsize: int = 512):
        self._cache = {}
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple, compute_func: Callable, *args, **kwargs) -> Any:
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
        self._cache.clear()
        self.hits = self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# Global cache instances (optional)
_symmetry_cache = StatsCache(maxsize=256)
_histogram_cache = StatsCache(maxsize=512)
_moment_cache = StatsCache(maxsize=128)


# ============================================================================
# VALIDATION AND TESTING
# ============================================================================

def validate_stats_utils() -> Dict[str, bool]:
    """Run internal test suite to verify all utilities."""
    results = {}
    try:
        # 1. Histogram creation
        vals = [1, 1, 2, 3, 3, 3]
        h = create_histogram(vals)
        results["histogram_creation"] = (
            h.total == 6 and h.num_bins == 3 and h.support == [1, 2, 3]
        )

        # 2. L₀ constraint
        try:
            apply_l0_constraint(h, max_bins=2)
            results["l0_constraint"] = False
        except HistogramError:
            results["l0_constraint"] = True

        # 3. S₄ orbit generation
        # FIX(Bug-6): generate_s4_orbits was renamed to generate_orbits(N, k).
        # The old name was never defined anywhere in the codebase, causing a
        # NameError at runtime. k=4 corresponds to a 4-letter alphabet (S₄).
        orbits = generate_orbits(N=4, k=4, max_orbits=100)
        results["s4_orbit_generation"] = (len(orbits) == 35)

        # 4. Orbit statistics
        if orbits:
            stats = compute_orbit_statistics(orbits[0])
            results["orbit_statistics"] = (
                "histogram" in stats and "moments" in stats
            )

        # 5. Moment constraints
        cons = MomentConstraints(s1=Fraction(10), s2=Fraction(30), n=5)
        results["moment_constraints"] = (
            cons.mean == 2 and cons.variance_bound == 2
        )

        # 6. Moment computation
        moments = compute_moments([1, 2, 3, 4])
        results["moment_computation"] = (
            moments["S1"] == 10 and
            moments["S2"] == 30 and
            moments["mean"] == Fraction(5, 2)
        )

        logger.info("✅ Statistical utilities validation passed")
    except Exception as e:
        logger.error(f"❌ Statistical utilities validation failed: {e}")
        results["validation_error"] = str(e)
    return results


# ============================================================================
# MAIN TESTING
# ============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("Testing Production-Ready Statistical Utilities")
    print("=" * 60)

    results = validate_stats_utils()
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

    # Specification demonstration
    print("\n" + "=" * 60)
    print("Specification Compliance Demo")
    print("=" * 60)


    print("\n1.  - S₄ Symmetry Orbits:")
    N = 6
    # FIX(Bug-6): generate_s4_orbits → generate_orbits(N, k=4)
    orbits = generate_orbits(N=N, k=4, max_orbits=100)
    print(f"   Generated {len(orbits)} orbits for N={N}")
    if orbits:
        sample = orbits[0]
        print(f"   Sample orbit: frequencies={sample.frequencies}")
        print(f"   Orbit weight: {sample.weight}")


    print("\n2.  - Moment Constraints:")
    vals = [1, 2, 2, 3, 4]
    moms = compute_moments(vals)
    print(f"   Values: {vals}")
    print(f"   S₁ = {moms['S1']}, S₂ = {moms['S2']}")
    print(f"   Mean = {float(moms['mean']):.2f}, Variance = {float(moms['variance']):.2f}")


    print("\n3.  - Histogram Operations:")
    h = create_histogram([1, 1, 2, 3, 3, 3])
    print(f"   Histogram: bins={h.bins}, counts={h.counts}")
    print(f"   L₀ norm: {h.num_bins}")

    print("\n" + "=" * 60)
    print("✅ Statistical Utilities Ready for Production")
    print("=" * 60)