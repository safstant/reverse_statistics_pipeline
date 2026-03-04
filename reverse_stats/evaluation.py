"""
Evaluation & Counting
=====================
Evaluates the Barvinok generating function at z=1 to extract the exact
lattice-point count, then lifts that count to a multiset count via
orbit (multinomial) weights.

Functions:
    evaluate_gf_at_unity  -- compute lim_{z->1} GF(z)
    lift_count_via_orbit_weights -- multiply each frequency vector by its orbit weight
    run_evaluation_phase  -- convenience wrapper for the full evaluation pipeline
    compute_marginal_distribution -- marginal probability of each alphabet value
"""

import numpy as np
import sympy
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import math
from fractions import Fraction

logger = logging.getLogger(__name__)

# Import from canonical enumeration utils (optional - for type hints)
try:
    from .enumerations import State, StateSpace, Combination
except ImportError:
    # Fallback for standalone usage
    pass


@dataclass
class CountingResult:
    """
    Pure mathematical output of the counting computation.

    Separates the mathematical answer from pipeline operational metadata.
    Added in v14 (Upgrade 3 — CountingResult split).
    """
    exact_count: int           # Total ordered sequences satisfying all constraints
    frequency_states: int      # Number of distinct frequency vectors
    method: str                # Algorithm used: 'barvinok_gf' | 'direct_enumeration'
    instance_hash: str         # SHA-256 of (N, S1, S2, min_val, max_val) — citable
    proof_metadata: Dict[str, Any] = None   # Cone counts, vertex counts, algorithm path


@dataclass
class EvaluationResult:
    """
    Full pipeline result — mathematical output plus operational metadata.

    The mathematical core is accessible via .counting_result.
    """
    frequency_state_count: int              # Number of distinct frequency vectors
    total_multiset_count: int               # Total ordered sequences (after orbit lifting)
    marginal_distribution: Dict[int, float] # P(x_i = v | constraints)
    evaluation_time: float                  # Wall-clock seconds for evaluation phase
    orbit_weights_computed: int             # Frequency states for which weights were computed
    metadata: Dict[str, Any] = None        # Full pipeline operational metadata

    @property
    def counting_result(self) -> 'CountingResult':
        """Return the pure mathematical result, separate from operational metadata."""
        meta = self.metadata or {}
        return CountingResult(
            exact_count=self.total_multiset_count,
            frequency_states=self.frequency_state_count,
            method=meta.get('method', 'unknown'),
            instance_hash=meta.get('instance_hash', ''),
            proof_metadata={
                k: v for k, v in meta.items()
                if k in ('num_vertices', 'num_cones', 'unimodular_cones',
                         'algorithm', 'evaluation_method', 'version',
                         'dimension', 'num_equations')
            },
        )

    def to_proof_record(self) -> Dict[str, Any]:
        """
        Return a JSON-serialisable proof record suitable for publication.

        Contains everything needed to reproduce and cite the result:
          - instance_hash: SHA-256 of the input — reproducible across runs
          - mathematical counts
          - algorithm path taken
          - pipeline version

        Example citation in a paper:
            "Result reproduced with instance_hash=<hash> using reverse_stats v14."
        """
        meta = self.metadata or {}
        return {
            'instance_hash':       meta.get('instance_hash', ''),
            'frequency_states':    self.frequency_state_count,
            'total_sequences':     self.total_multiset_count,
            'method':              meta.get('method', 'unknown'),
            'num_vertices':        meta.get('num_vertices', None),
            'num_cones':           meta.get('num_cones', None),
            'unimodular_cones':    meta.get('unimodular_cones', None),
            'effective_dimension': meta.get('dimension', None),
            'evaluation_time_s':   round(self.evaluation_time, 4),
            'version':             meta.get('version', 'v14'),
        }


def evaluate_gf_at_unity(gf: sympy.Expr, variables: List[sympy.Symbol]) -> int:
    """
    Extract lattice point count from generating function.

    Args:
        gf: Generating function from Barvinok assembly
        variables: List of variables (z0...z_{m-1})

    Returns:
        Number of integer points (frequency states) in polytope

    NOTE: This function assumes the generating function has no poles at z=1.
    For proper usage, ensure gf is canonicalized first.
    """
    logger.info("Evaluating generating function at z=1")

    subs_dict = {var: 1 for var in variables}

    # FIX(Bug-5): The original code computed the multivariate limit z→1 by
    # taking sequential *univariate* limits:
    #   for var in variables: result = limit(result, var, 1)
    # For vertex generating functions from Barvinok's algorithm the GF has
    # poles at z=1.  Univariate nested limits are *path-dependent* at
    # multi-dimensional poles and generally do not recover the correct constant
    # term of the multivariate Laurent expansion (which is what the lattice
    # point count equals).
    #
    # Correct approach: extract the constant term of the Laurent expansion
    # by performing a series expansion around z_j = 1 (substituting z_j = 1+t,
    # expanding in t, extracting the t⁰ coefficient, then substituting t=0)
    # for each variable sequentially.  This correctly handles poles because
    # `series` computes the full Laurent expansion rather than a directional
    # limit applied to the already-substituted expression.
    try:
        result = gf
        t = sympy.Symbol('_t_barvinok_', complex=True)

        for var in variables:
            # Substitute var = 1 + t, expand as Laurent series in t around 0,
            # take the constant (t^0) term, then recover by setting t=0.
            try:
                substituted = result.subs(var, 1 + t)
                # series up to O(t^1) gives the constant term (degree-0 coefficient)
                expanded = sympy.series(substituted, t, 0, 1)
                # Extract the t^0 coefficient (= constant term of Laurent expansion)
                coeff = expanded.coeff(t, 0)
                result = coeff
            except Exception as inner_e:
                # If series expansion fails for this variable (e.g. it has no
                # pole), fall back to direct substitution for that variable only.
                logger.debug(
                    f"evaluate_gf_at_unity: series expansion failed for {var}: "
                    f"{inner_e}; falling back to direct substitution for this variable."
                )
                result = result.subs(var, 1)

        count = int(sympy.simplify(result))
        logger.info(f"Found {count} frequency states")
        return count

    except Exception as e:
        logger.error(f"GF evaluation (Laurent series method) failed: {e}")
        # BUG-FIX (v14): The original fallback silently substituted z=1 and
        # returned the result even when the GF has poles at z=1.  For Barvinok
        # vertex-cone GFs the denominator factors are (1 − z^grade) which all
        # vanish at z=1, making direct substitution mathematically invalid.
        # The silent fallback masked the failure and returned garbage integers.
        #
        # Fix: attempt direct substitution ONLY after explicitly verifying that
        # the denominator does not vanish at z=1.  If it does, raise immediately
        # with a clear message rather than returning a silently wrong value.
        try:
            # Check whether the denominator vanishes at z=1 before substituting.
            den_at_unity = gf.subs(subs_dict)
            # If SymPy returns zoo (complex infinity), oo, or nan the GF has a
            # pole — direct substitution is invalid.
            if den_at_unity.has(sympy.zoo, sympy.nan, sympy.oo, -sympy.oo):
                raise ValueError(
                    "GF has a pole at z=1 and the Laurent series method failed. "
                    "Direct substitution is mathematically invalid here. "
                    "Check that barvinok_generating_function produced a fully "
                    "simplified rational function before calling evaluate_gf_at_unity."
                )
            count = int(sympy.simplify(den_at_unity))
            # Only reach here when the GF is already pole-free at z=1
            # (e.g. after full Barvinok signed cancellation).  Log at WARNING
            # so the caller knows which path was taken.
            logger.warning(
                "evaluate_gf_at_unity: Laurent series method failed; used direct "
                "substitution fallback.  Result is valid only because denominator "
                "does not vanish at z=1 for this GF."
            )
            return count
        except ValueError:
            # Re-raise the pole-at-unity error without wrapping
            raise
        except Exception as e2:
            raise ValueError(
                f"Could not evaluate generating function at z=1 "
                f"(Laurent series failed: {e}; direct substitution also failed: {e2})"
            ) from e2


def compute_multinomial_weight(frequency_vector: np.ndarray, N: int) -> int:
    """
    Compute orbit weight w(f) = N! / (f₁! · f₂! · ... · fₘ!)

    Args:
        frequency_vector: Array of frequencies [f₁, f₂, ..., fₘ]
        N: Total count (sum of frequencies)

    Returns:
        Multinomial coefficient
    """
    from math import factorial
    from functools import reduce
    import operator
    
    # Verify sum matches N
    if abs(sum(frequency_vector) - N) > 1e-10:
        logger.warning(f"Frequency sum {sum(frequency_vector)} != N={N}")
    
    # Compute N! / Π(fᵢ!)
    # Use integer arithmetic to avoid floating point
    result = 1
    remaining = int(N)

    # Compute as product of binomials:
    # C(N, f₁) * C(N-f₁, f₂) * ... * C(N-f₁-...-f_{m-1}, fₘ)
    for f_raw in frequency_vector[:-1]:  # All but last
        f = int(f_raw)  # Convert Fraction or float to int
        if f > remaining:
            return 0

        k = min(f, remaining - f)
        binom = 1
        for i in range(1, k + 1):
            binom = binom * (remaining - k + i) // i

        result *= binom
        remaining -= f

    return result


def lift_count_via_orbit_weights(frequency_count: int,
                                 feasible_vectors: List[np.ndarray],
                                 N: int,
                                 alphabet_values: Optional[Tuple[int, ...]] = None,
                                 ) -> int:
    """Transform frequency state count to actual multiset count via orbit weights.

    Args:
        frequency_count: Number of frequency states (from GF evaluation)
        feasible_vectors: List of feasible frequency vectors
        N: Total count (for orbit weights)
        alphabet_values: Tuple of alphabet bin values aligned with each position
            in the frequency vectors.  When provided, the stabilizer-corrected
            weight ``SymmetryOrbit.alphabet_aware_weight(alphabet_values)`` is
            used instead of the raw multinomial.  This is necessary when two or
            more alphabet bins share the same value (e.g. ``[1, 2, 1, 3]``);
            without the correction the same multiset is counted once per
            indistinguishable frequency-vector permutation, inflating the total
            by the stabilizer size.  For distinct alphabets (the pipeline
            default) the two methods give identical results, so passing
            ``alphabet_values`` is always safe.

    Returns:
        Total number of multisets satisfying constraints

    NOTE: This is the critical step that converts frequency states
    to actual multisets via orbit-stabilizer theorem.
    Without this step, you get the wrong answer!
    """
    logger.info(f"Lifting {frequency_count} frequency states via orbit weights")

    if not feasible_vectors:
        # constraints — return 0 rather than crashing the pipeline with ValueError.
        logger.info("lift_count_via_orbit_weights: feasible_vectors is empty — returning 0 (no valid configurations)")
        return 0

    # Import SymmetryOrbit for stabilizer-aware weighting when alphabet provided
    _SymOrbit = None
    if alphabet_values is not None:
        try:
            try:
                from .stats_utils import SymmetryOrbit as _SO
            except ImportError:
                from stats_utils import SymmetryOrbit as _SO
            _SymOrbit = _SO
        except ImportError:
            logger.debug("SymmetryOrbit not available; falling back to plain multinomial weight")

    total = 0
    vectors_to_process = feasible_vectors

    for i, f in enumerate(vectors_to_process):
        if i % 10000 == 0 and i > 0:
            logger.debug(f"Processed {i}/{len(vectors_to_process)} vectors")

        if _SymOrbit is not None and alphabet_values is not None:
            # FIX-B: Use stabilizer-corrected orbit weight so that repeated
            # alphabet values do not inflate the count.
            try:
                orbit = _SymOrbit(frequencies=tuple(int(x) for x in f))
                weight = orbit.alphabet_aware_weight(alphabet_values)
            except Exception:
                weight = compute_multinomial_weight(f, N)
        else:
            weight = compute_multinomial_weight(f, N)
        total += weight

    logger.info(f"Lifted count: {total} multisets")
    return total


def validate_reverse_count(frequency_count: int, 
                          multiset_count: int,
                          expected_ratio: Optional[float] = None) -> bool:
    """
    Validate that the lifted count is reasonable.

    Typically: multiset_count >> frequency_count
    """
    ratio = multiset_count / frequency_count if frequency_count > 0 else 0
    
    if expected_ratio:
        is_valid = abs(ratio - expected_ratio) / expected_ratio < 0.1
    else:
        # Basic sanity: lifted count should be larger
        is_valid = multiset_count > frequency_count
    
    logger.info(f"Lift ratio: {ratio:.2e} ({'✓' if is_valid else '✗'})")
    return is_valid


def compute_marginal_distribution(feasible_vectors: List[np.ndarray],
                                  alphabet: List[int],
                                  N: int) -> Dict[int, float]:
    """
    Compute P(x_i = v | constraints) from feasible vectors.

    Args:
        feasible_vectors: List of frequency vectors
        alphabet: Alphabet values [v₁, v₂, ..., vₘ]
        N: Total count

    Returns:
        Dictionary mapping value -> probability
    """
    m = len(alphabet)
    total_weight = 0
    weighted_counts = [0] * m  # Use Python ints to avoid numpy dtype issues

    for f in feasible_vectors:
        weight = compute_multinomial_weight(f, N)
        total_weight += weight
        for i, fi in enumerate(f):
            weighted_counts[i] += int(fi) * weight

    if total_weight > 0:
        probs = [wc / (total_weight * N) for wc in weighted_counts]
    else:
        probs = [0.0] * m

    return {alphabet[i]: float(probs[i]) for i in range(m)}


def run_evaluation_phase(gf: sympy.Expr,
                        variables: List[sympy.Symbol],
                        feasible_vectors: List[np.ndarray],
                        alphabet: List[int],
                        N: int) -> EvaluationResult:
    """
    Run the complete evaluation pipeline.

    Args:
        gf: Generating function
        variables: GF variables
        feasible_vectors: List of feasible frequency vectors
        alphabet: Alphabet values (bin values, length = number of bins k)
        N: Total count

    Returns:
        Complete evaluation result

    Example:
        >>> from reverse_stats.evaluation import run_evaluation_phase
        >>> result = run_evaluation_phase(gf, variables, vectors, [1,2,3], 100)
        >>> print(f"Total multisets: {result.total_multiset_count}")
    """
    import time
    start_time = time.time()

    freq_count = evaluate_gf_at_unity(gf, variables)

    # ── Step 22: Orbit weight multiplication (V15.3 C5) ──────────────────────
    # Explicitly logged per spec §9.2.  Previously embedded silently.
    logger.info(
        f"Step 22: orbit weight lifting — {len(feasible_vectors)} frequency vectors"
    )

    # FIX-B: Pass alphabet_values so that repeated-value alphabets are handled
    # with a stabilizer-corrected weight.  For the standard pipeline case
    # (distinct consecutive integers) this has no effect on the result.
    multiset_count = lift_count_via_orbit_weights(
        freq_count, feasible_vectors, N,
        alphabet_values=tuple(alphabet) if alphabet else None,
    )

    is_valid = validate_reverse_count(freq_count, multiset_count)

    # Compute marginal distribution
    marginals = compute_marginal_distribution(feasible_vectors, alphabet, N)

    elapsed = time.time() - start_time

    return EvaluationResult(
        frequency_state_count=freq_count,
        total_multiset_count=multiset_count,
        marginal_distribution=marginals,
        evaluation_time=elapsed,
        orbit_weights_computed=len(feasible_vectors),
        metadata={
            "validation_passed": is_valid,
            "alphabet": alphabet,
            "N": N
        }
    )



if __name__ == "__main__":
    """
    Standalone demonstration of the evaluation module.
    Run this file directly to see how the evaluation functions work.
    """
    import sympy
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Evaluation & Counting Module Demo")
    print("=" * 60)
    print("\nNOTE: This module is intentionally standalone.")
    print("To use in your code, import directly:")
    print("  from reverse_stats.evaluation import run_evaluation_phase")
    print("=" * 60)
    
    # Small test case: N=3, alphabet=[1,2]
    # All frequency vectors summing to 3: [3,0], [2,1], [1,2], [0,3]
    test_vectors = [
        np.array([3, 0]),
        np.array([2, 1]), 
        np.array([1, 2]),
        np.array([0, 3])
    ]
    alphabet = [1, 2]
    N = 3
    
    print(f"\nTest case: N={N}, alphabet={alphabet}")
    print(f"Frequency vectors: {[list(v) for v in test_vectors]}")
    
    # Compute orbit weights manually
    print("\nOrbit weights:")
    total = 0
    for f in test_vectors:
        w = compute_multinomial_weight(f, N)
        print(f"  f={list(f)}: weight={w}")
        total += w
    
    print(f"Total multisets: {total}")
    
    # Should equal 2^3 = 8 (all sequences of length 3 from {1,2})
    print(f"Expected: 2^{N} = {2**N}")
    
    # Marginal distribution
    marginals = compute_marginal_distribution(test_vectors, alphabet, N)
    print(f"\nMarginal distribution: {marginals}")
    
    print("\n" + "=" * 60)
    print("✅ Evaluation Module Ready")
    print("=" * 60)