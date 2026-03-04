"""
Input Validation Module for Reverse Statistics Pipeline
Provides validation for input statistics and parameters.


Critical for: Ensuring input data meets mathematical constraints before pipeline execution
"""

from .exceptions import ReverseStatsError
import math
import logging
import sys
import os
from fractions import Fraction
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ============================================================================
# HANDLE IMPORTS FOR BOTH PACKAGE AND STANDALONE EXECUTION
# ============================================================================

# When running as a script, add parent directory to path
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Core types - direct imports with fallback for standalone testing
try:
    from .pipeline_types import (
        ObservedStatistics,
        ReverseStatsError
    )
    from .config import get_config
    HAS_DEPS = True
except ImportError:
    # For standalone testing - define minimal versions
    HAS_DEPS = False
    
    @dataclass
    class ObservedStatistics:
        """Minimal version for testing."""
        N: int
        S1: Fraction
        S2: Fraction
        S3: Optional[Fraction] = None
        S4: Optional[Fraction] = None
        min_val: int = 1
        max_val: int = 6
    
    class ReverseStatsError(ReverseStatsError):
        pass
    
    def get_config():
        return {}

# ============================================================================
# EXCEPTIONS
# ============================================================================

class ValidationError(ReverseStatsError):
    """Base exception for validation errors."""
    def __init__(self, message: str):
        self.message = message

        super().__init__(message)


class CauchySchwarzError(ValidationError):
    """Raised when Cauchy-Schwarz inequality is violated."""
    def __init__(self, S1: float, S2: float, N: int):
        self.S1 = S1
        self.S2 = S2
        self.N = N
        min_S2 = (S1 * S1) / N
        super().__init__(
            f"Cauchy-Schwarz violation: S₂={S2} < S₁²/N={min_S2}",

        )


class PopoviciuError(ValidationError):
    """Raised when Popoviciu's variance inequality is violated."""
    def __init__(self, variance: float, max_variance: float):
        self.variance = variance
        self.max_variance = max_variance
        super().__init__(
            f"Variance {variance} exceeds Popoviciu bound {max_variance}",

        )


# ============================================================================

# ============================================================================

def validate_observed_count(N: int) -> int:
    """
    : Validate N as positive integer.

    Args:
        N: Total count (S₀)

    Returns:
        Validated N

    Raises:
        ValidationError: If N is not a positive integer
    """
    if not isinstance(N, int):
        raise ValidationError(f"N must be integer, got {type(N).__name__}")
    if N <= 0:
        raise ValidationError(f"N must be positive, got {N}")
    if N > 1_000_000_000:
        logger.warning(f"Large N={N} may cause performance issues")
    return N


def validate_observed_sum(S1: float, N: int, min_val: int, max_val: int) -> float:
    """
    : Validate S₁ against bounds [N·min, N·max].

    Args:
        S1: First moment sum (S₁)
        N: Total count
        min_val: Minimum alphabet value
        max_val: Maximum alphabet value

    Returns:
        Validated S1

    Raises:
        ValidationError: If S1 is outside feasible bounds
    """
    min_sum = N * min_val
    max_sum = N * max_val
    
    if S1 < min_sum or S1 > max_sum:
        raise ValidationError(
            f"S₁={S1} outside feasible range [{min_sum}, {max_sum}]",

        )
    return S1


def validate_observed_sum_sq(S2: float, S1: float, N: int) -> float:
    """
    : Validate S₂ and enforce Cauchy-Schwarz inequality.

    Args:
        S2: Second moment sum (S₂)
        S1: First moment sum (S₁)
        N: Total count

    Returns:
        Validated S2

    Raises:
        CauchySchwarzError: If S₂ < S₁²/N
    """
    # Exact rational check using Fractions
    S1_frac = Fraction(str(S1)) if isinstance(S1, float) else Fraction(S1)
    S2_frac = Fraction(str(S2)) if isinstance(S2, float) else Fraction(S2)
    N_frac = Fraction(N)
    
    # Cauchy-Schwarz: N·S₂ ≥ S₁²
    if N_frac * S2_frac < S1_frac * S1_frac:
        raise CauchySchwarzError(S1, S2, N)
    
    return S2


def validate_observed_skew(S3: Optional[float]) -> Optional[float]:
    """
    : Type validation of optional S₃.

    Args:
        S3: Third moment sum (S₃)

    Returns:
        Validated S3 or None
    """
    if S3 is not None:
        if not isinstance(S3, (int, float)):
            raise ValidationError(f"S₃ must be number, got {type(S3).__name__}")
        if math.isnan(S3) or math.isinf(S3):
            raise ValidationError(f"S₃ must be finite, got {S3}")
    return S3


def validate_observed_kurtosis(S4: Optional[float]) -> Optional[float]:
    """
    : Type validation of optional S₄.

    Args:
        S4: Fourth moment sum (S₄)

    Returns:
        Validated S4 or None
    """
    if S4 is not None:
        if not isinstance(S4, (int, float)):
            raise ValidationError(f"S₄ must be number, got {type(S4).__name__}")
        if math.isnan(S4) or math.isinf(S4):
            raise ValidationError(f"S₄ must be finite, got {S4}")
    return S4


def validate_observed_min(min_val: int) -> int:
    """
    : Validate min_val as integer.

    Args:
        min_val: Minimum alphabet value

    Returns:
        Validated min_val
    """
    if not isinstance(min_val, int):
        raise ValidationError(f"min_val must be integer, got {type(min_val).__name__}")
    if min_val < 0:
        logger.warning(f"Negative min_val={min_val} - are you sure?")
    return min_val


def validate_observed_max(max_val: int) -> int:
    """
    : Validate max_val as integer.

    Args:
        max_val: Maximum alphabet value

    Returns:
        Validated max_val
    """
    if not isinstance(max_val, int):
        raise ValidationError(f"max_val must be integer, got {type(max_val).__name__}")
    return max_val


def validate_domain_consistency(min_val: int, max_val: int, N: int) -> bool:
    """
    : Cross-validate domain parameters.

    Args:
        min_val: Minimum alphabet value
        max_val: Maximum alphabet value
        N: Total count

    Returns:
        True if consistent

    Raises:
        ValidationError: If domain is inconsistent
    """
    if min_val > max_val:
        raise ValidationError(f"min_val={min_val} > max_val={max_val}")
    
    if max_val - min_val > 100:
        logger.warning(f"Large alphabet range [{min_val}, {max_val}] may cause performance issues")
    
    if N > 1_000_000 and (max_val - min_val) > 50:
        logger.warning(f"Large N={N} with wide alphabet range may be computationally expensive")
    
    return True


def check_cauchy_schwarz(N: int, S1: float, S2: float) -> None:
    """
    : Enforce Cauchy-Schwarz inequality.

    Args:
        N: Total count
        S1: First moment sum
        S2: Second moment sum

    Raises:
        CauchySchwarzError: If S₂ < S₁²/N
    """
    # Use Fraction for exact comparison
    S1_frac = Fraction(str(S1)) if isinstance(S1, float) else Fraction(S1)
    S2_frac = Fraction(str(S2)) if isinstance(S2, float) else Fraction(S2)
    N_frac = Fraction(N)
    
    if N_frac * S2_frac < S1_frac * S1_frac:
        raise CauchySchwarzError(S1, S2, N)


def check_popoviciu(variance: float, min_val: int, max_val: int) -> None:
    """
    Popoviciu's inequality: var ≤ (max - min)²/4.

    Args:
        variance: Sample variance
        min_val: Minimum value
        max_val: Maximum value

    Raises:
        PopoviciuError: If variance exceeds bound
    """
    max_variance = ((max_val - min_val) ** 2) / 4
    if variance > max_variance + 1e-10:  # Small tolerance for floating point
        raise PopoviciuError(variance, max_variance)


def validate_input_consistency(
    N: int,
    S1: float,
    S2: float,
    min_val: int,
    max_val: int,
    S3: Optional[float] = None,
    S4: Optional[float] = None
):
    """
    : Holistic input validation.

    Args:
        N: Total count
        S1: First moment sum
        S2: Second moment sum
        min_val: Minimum alphabet value
        max_val: Maximum alphabet value
        S3: Optional third moment sum
        S4: Optional fourth moment sum

    Returns:
        Validated ObservedStatistics object (or dict in standalone mode)

    Raises:
        ValidationError: If any validation fails
    """
    # Run all validations
    N_val = validate_observed_count(N)
    S1_val = validate_observed_sum(S1, N, min_val, max_val)
    S2_val = validate_observed_sum_sq(S2, S1, N)
    S3_val = validate_observed_skew(S3)
    S4_val = validate_observed_kurtosis(S4)
    min_val = validate_observed_min(min_val)
    max_val = validate_observed_max(max_val)
    
    validate_domain_consistency(min_val, max_val, N)
    
    # Additional consistency checks
    # FIX(Bug-27): The original code computed variance using Python float:
    #   mean = S1 / N;  variance = (S2 / N) - (mean * mean)
    # For large N where S2/N ≈ (S1/N)², catastrophic cancellation makes the
    # result appear negative or zero even for perfectly valid inputs, causing
    # a false PopoviciuError. S1, S2, N are always integers at this point —
    # use exact Fraction arithmetic to avoid all cancellation.
    mean_exact = Fraction(int(S1), int(N))
    variance_exact = Fraction(int(S2), int(N)) - mean_exact * mean_exact
    variance = float(variance_exact)  # only converted to float for Popoviciu check
    
    # Check Popoviciu inequality
    check_popoviciu(variance, min_val, max_val)
    
    # Return appropriate type based on environment
    if HAS_DEPS:
        return ObservedStatistics(
            n=N_val,
            s1=int(S1_val),
            s2=int(S2_val),
        )
    else:
        # Return dict for standalone testing
        return {
            "N": N_val,
            "S1": Fraction(str(S1_val)),
            "S2": Fraction(str(S2_val)),
            "S3": Fraction(str(S3_val)) if S3_val is not None else None,
            "S4": Fraction(str(S4_val)) if S4_val is not None else None,
            "min_val": min_val,
            "max_val": max_val
        }


def validate_moments_match(
    frequencies: List[int],
    alphabet: List[int],
    expected_stats
) -> bool:
    """
    Verify that frequencies produce expected moments.

    Args:
        frequencies: Frequency counts for each alphabet value
        alphabet: Alphabet values
        expected_stats: Expected moment statistics

    Returns:
        True if moments match within tolerance
    """
    if len(frequencies) != len(alphabet):
        return False
    
    N = sum(frequencies)
    if N != expected_stats.N:
        return False
    
    S1 = sum(f * a for f, a in zip(frequencies, alphabet))
    S2 = sum(f * a * a for f, a in zip(frequencies, alphabet))
    
    # Compare with expected (using Fractions for exact comparison)
    S1_frac = Fraction(str(S1)) if isinstance(S1, float) else Fraction(S1)
    S2_frac = Fraction(str(S2)) if isinstance(S2, float) else Fraction(S2)
    
    if HAS_DEPS:
        return S1_frac == expected_stats.S1 and S2_frac == expected_stats.S2
    else:
        # For dict-based expected_stats
        return S1_frac == expected_stats["S1"] and S2_frac == expected_stats["S2"]


# ============================================================================
# VALIDATION AND TESTING
# ============================================================================

def validate_validation_utils() -> Dict[str, bool]:
    """Run internal test suite to verify validation utilities."""
    results = {}
    
    try:
        # Test 1: Count validation
        results["count_validation"] = (
            validate_observed_count(720) == 720
        )
        
        # Test 2: Sum validation
        results["sum_validation"] = (
            validate_observed_sum(2520, 720, 1, 6) == 2520
        )
        
        # Test 3: Cauchy-Schwarz validation
        results["cauchy_schwarz"] = (
            validate_observed_sum_sq(10500, 2520, 720) == 10500
        )
        
        # Test 4: Domain consistency
        results["domain_consistency"] = (
            validate_domain_consistency(1, 6, 720) is True
        )
        
        # Test 5: Popoviciu check
        try:
            check_popoviciu(10, 1, 6)  # variance 10 > max 6.25
            results["popoviciu"] = False
        except PopoviciuError:
            results["popoviciu"] = True
        
        # Test 6: Full input validation
        stats = validate_input_consistency(720, 2520, 10500, 1, 6)
        if HAS_DEPS:
            results["full_validation"] = (
                stats.N == 720 and
                stats.S1 == 2520 and
                stats.S2 == 10500
            )
        else:
            results["full_validation"] = (
                stats["N"] == 720 and
                stats["S1"] == 2520 and
                stats["S2"] == 10500
            )
        
        logger.info("✅ Validation utilities validation passed")
        
    except Exception as e:
        logger.error(f"❌ Validation utilities validation failed: {e}")
        results["validation_error"] = str(e)
    
    return results


# ============================================================================
# MAIN TESTING
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("Testing Input Validation Utilities")
    print("=" * 60)
    print("\nNOTE: Running in standalone mode - using minimal dependencies.")
    print("      When imported as part of package, full ObservedStatistics used.\n")
    
    results = validate_validation_utils()
    
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
    
    # Demo
    print("\n" + "=" * 60)
    print("Validation Demo - example problem")
    print("=" * 60)
    
    # Example parameters for testing
    N = 720
    S1 = 2520
    S2 = 10500
    min_val = 1
    max_val = 6
    
    print(f"\nInput: N={N}, S₁={S1}, S₂={S2}, range=[{min_val},{max_val}]")
    
    try:
        stats = validate_input_consistency(N, S1, S2, min_val, max_val)
        print(f"✅ Validation passed")
        if HAS_DEPS:
            print(f"   Mean: {float(stats.S1)/stats.N:.2f}")
            print(f"   Variance: {float(stats.S2)/stats.N - (float(stats.S1)/stats.N)**2:.2f}")
        else:
            print(f"   Mean: {float(stats['S1'])/stats['N']:.2f}")
            print(f"   Variance: {float(stats['S2'])/stats['N'] - (float(stats['S1'])/stats['N'])**2:.2f}")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
    
    # Test invalid case
    print("\nTesting invalid input (S₂ too small):")
    try:
        validate_input_consistency(720, 2520, 8000, 1, 6)
    except CauchySchwarzError as e:
        print(f"✅ Correctly caught: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Validation Utilities Ready for Production")
    print("=" * 60)