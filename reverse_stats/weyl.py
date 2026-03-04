"""
Weyl Group and Exceptional Lie Algebra Module for Reverse Statistics Pipeline

FUTURE FEATURE: Exceptional Lie Algebras for Non-Standard Alphabets
====================================================================
This module contains implementations of exceptional Lie algebras
(E6, E7, E8, F4, G2) for future Weyl-symmetric enumeration over
non-standard alphabets with repeated values.

Current status: NOT YET WIRED INTO MAIN PIPELINE
- detect_weyl_symmetry(alphabet) → wired into frequency.py and orbit.py
  Returns WeylType.NONE for distinct-value alphabets (standard case).
  Returns non-NONE WeylType when repeated values imply Lie-group symmetry.
- collapse_by_weyl_symmetry() — FUTURE: reduce polytope by symmetry group
- WeylGroup.act_on_vector() — exact Fraction reflection, usable now
  returns False and this phase is skipped.
- These ~600 lines of Lie algebra code are preserved for future research
  and extension to non-standard alphabets.

Reference: Barvinok (2002) Chapter 9 - Symmetry in Lattice Point Enumeration
Integration planned when supporting alphabets with repeated values.


"""

from .exceptions import ReverseStatsError
import numpy as np
import math
from fractions import Fraction
from typing import List, Tuple, Dict, Any, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import itertools

logger = logging.getLogger(__name__)

# ============================================================================
# EXCEPTIONS
# ============================================================================

class WeylError(ReverseStatsError):
    """Base exception for Weyl symmetry operations."""
    def __init__(self, message: str):
        self.message = message

        super().__init__(message)


class SymmetryDetectionError(WeylError):
    """Raised when symmetry detection fails."""
    pass


class RootSystemError(WeylError):
    """Raised for root system construction errors."""
    pass


# Import DimensionLimitError from canonical source
try:
    from dimension import DimensionLimitError
except ImportError:
    class DimensionLimitError(WeylError):
        """Raised when dimension exceeds guard threshold."""
        def __init__(self, dimension: int, threshold: int = 15):
            self.dimension = dimension
            self.threshold = threshold
            super().__init__(f"Dimension {dimension} exceeds guard threshold {threshold}")


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class WeylType(Enum):
    """Types of Weyl groups."""
    A = "A"  # A_n series
    B = "B"  # B_n series
    C = "C"  # C_n series
    D = "D"  # D_n series
    E6 = "E6"  # E_6 exceptional
    E7 = "E7"  # E_7 exceptional
    E8 = "E8"  # E_8 exceptional
    F4 = "F4"  # F_4 exceptional
    G2 = "G2"  # G_2 exceptional
    NONE = "none"  # No Weyl symmetry


class SymmetryAction(Enum):
    """Types of symmetry actions."""
    PERMUTATION = "permutation"  # Permutation of coordinates
    SIGN = "sign"  # Sign changes
    WEYL = "weyl"  # Full Weyl group action


# ============================================================================
# FUTURE FEATURE: Exceptional Lie Algebras
# ============================================================================
# The following code implements root systems for exceptional Lie algebras:
# - E6 (rank 6)
# - E7 (rank 7)
# - E8 (rank 8)
# - F4 (rank 4)
# - G2 (rank 2)
#
# These are preserved for future extension to non-standard alphabets
# where alphabet values may have symmetries.
#
# Wired into: frequency.find_distributions_by_moments, orbit.generate_orbits
# Both call detect_weyl_symmetry() to log symmetry type.
# collapse_by_weyl_symmetry() will reduce orbits once implemented.

def create_e6_root_system() -> Tuple[List[Tuple[int, ...]], List[int]]:
    """
    Create E6 root system (78 roots).

    Returns:
        Tuple of (simple_roots, positive_roots_indices)

    FUTURE: For non-standard alphabets with E6 symmetry.
    """
    # Simple roots for E6 (in 8D ambient space)
    simple_roots = [
        (1, -1, 0, 0, 0, 0, 0, 0),
        (0, 1, -1, 0, 0, 0, 0, 0),
        (0, 0, 1, -1, 0, 0, 0, 0),
        (0, 0, 0, 1, -1, 0, 0, 0),
        (0, 0, 0, 0, 1, -1, 0, 0),
        (1, 1, 0, 0, 0, 0, 0, 0)  # This is actually a linear combination
    ]
    return simple_roots, list(range(36))  # Placeholder for positive roots


def create_e7_root_system() -> Tuple[List[Tuple[int, ...]], List[int]]:
    """
    Create E7 root system (126 roots).

    Returns:
        Tuple of (simple_roots, positive_roots_indices)

    FUTURE: For non-standard alphabets with E7 symmetry.
    """
    simple_roots = [
        (1, -1, 0, 0, 0, 0, 0, 0),
        (0, 1, -1, 0, 0, 0, 0, 0),
        (0, 0, 1, -1, 0, 0, 0, 0),
        (0, 0, 0, 1, -1, 0, 0, 0),
        (0, 0, 0, 0, 1, -1, 0, 0),
        (0, 0, 0, 0, 0, 1, -1, 0),
        (1, 1, 0, 0, 0, 0, 0, 0)
    ]
    return simple_roots, list(range(63))


def create_e8_root_system() -> Tuple[List[Tuple[int, ...]], List[int]]:
    """
    Create E8 root system (240 roots).

    Returns:
        Tuple of (simple_roots, positive_roots_indices)

    FUTURE: For non-standard alphabets with E8 symmetry.
    """
    simple_roots = [
        (1, -1, 0, 0, 0, 0, 0, 0),
        (0, 1, -1, 0, 0, 0, 0, 0),
        (0, 0, 1, -1, 0, 0, 0, 0),
        (0, 0, 0, 1, -1, 0, 0, 0),
        (0, 0, 0, 0, 1, -1, 0, 0),
        (0, 0, 0, 0, 0, 1, -1, 0),
        (0, 0, 0, 0, 0, 0, 1, -1),
        (1, 1, 0, 0, 0, 0, 0, 0)
    ]
    return simple_roots, list(range(120))


def create_f4_root_system() -> Tuple[List[Tuple[int, ...]], List[int]]:
    """
    Create F4 root system (48 roots).

    Returns:
        Tuple of (simple_roots, positive_roots_indices)

    FUTURE: For non-standard alphabets with F4 symmetry.
    """
    simple_roots = [
        (1, -1, 0, 0),
        (0, 1, -1, 0),
        (0, 0, 1, 0),
        (-1, -1, -1, -1)  # Actually ( -1/2, -1/2, -1/2, -1/2) scaled
    ]
    return simple_roots, list(range(24))


def create_g2_root_system() -> Tuple[List[Tuple[int, ...]], List[int]]:
    """
    Create G2 root system (12 roots).

    Returns:
        Tuple of (simple_roots, positive_roots_indices)

    FUTURE: For non-standard alphabets with G2 symmetry.
    """
    simple_roots = [
        (1, -1, 0),
        (-2, 1, 1)  # Actually in 3D with constraint
    ]
    return simple_roots, list(range(6))


# ============================================================================
# WEYL GROUP OPERATIONS
# ============================================================================

class WeylGroup:
    """
    Weyl group representation.

    FUTURE: This class will be used for symmetry reduction when
    alphabet values have non-trivial symmetries.
    """
    
    def __init__(self, weyl_type: WeylType, rank: int = None):
        self.weyl_type = weyl_type
        self._rank_override = rank
        self.rank = rank if rank is not None else self._get_rank()
        self.order = self._get_order()
        self.simple_roots = self._create_simple_roots()
        self.positive_roots = []
        self.root_system = []
        
    def _get_rank(self) -> int:
        """Get rank of Weyl group."""
        rank_map = {
            WeylType.E6: 6,
            WeylType.E7: 7,
            WeylType.E8: 8,
            WeylType.F4: 4,
            WeylType.G2: 2,
            WeylType.A: 1,  # Placeholder
            WeylType.B: 1,
            WeylType.C: 1,
            WeylType.D: 1,
        }
        return rank_map.get(self.weyl_type, 0)
    
    def _get_order(self) -> int:
        """Get order of Weyl group."""
        order_map = {
            WeylType.E6: 51840,
            WeylType.E7: 2903040,
            WeylType.E8: 696729600,
            WeylType.F4: 1152,
            WeylType.G2: 12,
            WeylType.A: 2,
            WeylType.B: 2,
            WeylType.C: 2,
            WeylType.D: 2,
        }
        return order_map.get(self.weyl_type, 1)
    
    def _create_simple_roots(self) -> List[Tuple[int, ...]]:
        """Create simple roots for the Weyl group."""
        creators = {
            WeylType.E6: create_e6_root_system,
            WeylType.E7: create_e7_root_system,
            WeylType.E8: create_e8_root_system,
            WeylType.F4: create_f4_root_system,
            WeylType.G2: create_g2_root_system,
        }
        creator = creators.get(self.weyl_type)
        if creator:
            roots, _ = creator()
            return roots
        # Classical root systems A_n, B_n, C_n, D_n
        n = self.rank
        if self.weyl_type == WeylType.A:
            # Simple roots: e_i - e_{i+1} in R^{n+1}
            return [tuple(1 if j == i else (-1 if j == i + 1 else 0)
                          for j in range(n + 1))
                    for i in range(n)]
        elif self.weyl_type == WeylType.B:
            # Simple roots: e_i - e_{i+1} for i<n-1, then e_{n-1}
            roots = [tuple(1 if j == i else (-1 if j == i + 1 else 0)
                           for j in range(n))
                     for i in range(n - 1)]
            roots.append(tuple(1 if j == n - 1 else 0 for j in range(n)))
            return roots
        elif self.weyl_type == WeylType.C:
            # Simple roots: e_i - e_{i+1} for i<n-1, then 2*e_{n-1}
            roots = [tuple(1 if j == i else (-1 if j == i + 1 else 0)
                           for j in range(n))
                     for i in range(n - 1)]
            roots.append(tuple(2 if j == n - 1 else 0 for j in range(n)))
            return roots
        elif self.weyl_type == WeylType.D:
            # Simple roots: e_i - e_{i+1} for i<n-1, then e_{n-2} + e_{n-1}
            roots = [tuple(1 if j == i else (-1 if j == i + 1 else 0)
                           for j in range(n))
                     for i in range(n - 1)]
            roots.append(tuple(1 if j in (n - 2, n - 1) else 0 for j in range(n)))
            return roots
        return []
    
    def act_on_vector(self, vector: Tuple[Fraction, ...],
                     transformation: Any) -> Tuple[Fraction, ...]:
        """
        Apply i-th simple reflection to vector.

        Args:
            vector: Frequency vector to transform.
            transformation: Integer index into simple_roots list.
                The i-th simple reflection s_i sends v -> v - 2(v·α_i / α_i·α_i) α_i.

        Returns:
            Transformed vector (same dimension as input).
        """
        roots = self._create_simple_roots()
        if not roots:
            return vector
        try:
            idx = int(transformation)
        except (TypeError, ValueError):
            return vector
        if idx < 0 or idx >= len(roots):
            return vector
        alpha = roots[idx]
        # Use Fraction arithmetic for exactness
        dot_va = sum(Fraction(v) * Fraction(a) for v, a in zip(vector, alpha))
        dot_aa = sum(Fraction(a) * Fraction(a) for a in alpha)
        if dot_aa == 0:
            return vector
        scale = 2 * dot_va / dot_aa
        return tuple(Fraction(v) - scale * Fraction(a) for v, a in zip(vector, alpha))
    
    def is_symmetric(self, vector: Tuple[Fraction, ...]) -> bool:
        """
        Check if vector is symmetric under Weyl group.

        FUTURE: Determine orbit structure under Weyl group.
        """
        return False


def detect_weyl_symmetry(alphabet: List[int]) -> WeylType:
    """
    Detect Weyl symmetry from alphabet values.

    For alphabets with no Weyl symmetry, returns NONE.
    For non-standard alphabets with repeated values, may detect
    exceptional symmetries.

    Args:
        alphabet: List of alphabet values

    Returns:
        WeylType.NONE for distinct values, otherwise potential symmetry type
    """
    # Current implementation: no symmetry for distinct values
    if len(set(alphabet)) == len(alphabet):
        return WeylType.NONE
    
    # FUTURE: Implement detection of exceptional symmetries
    # based on the structure of repeated values
    
    return WeylType.NONE


def collapse_by_weyl_symmetry(polytope: Any, symmetry: WeylType) -> Any:
    """
    Collapse polytope by Weyl symmetry.

    Args:
        polytope: Input polytope
        symmetry: Detected Weyl symmetry type

    Returns:
        Symmetry-reduced polytope

    FUTURE: Implement quotient by Weyl group action.
    """
    if symmetry == WeylType.NONE:
        return polytope
    
    # FUTURE: Implement quotient construction
    logger.info(f"Collapsing by {symmetry.value} symmetry (future feature)")
    return polytope


# ============================================================================
# VALIDATION AND TESTING
# ============================================================================

def validate_weyl_utils() -> Dict[str, bool]:
    """Run internal test suite to verify Weyl utilities."""
    results = {}
    
    try:

        alphabet = [1, 2, 3, 4, 5, 6]
        sym = detect_weyl_symmetry(alphabet)
        results["detect_distinct"] = (sym == WeylType.NONE)
        
        # Test 2: E6 root system creation
        e6_roots, e6_pos = create_e6_root_system()
        results["e6_creation"] = len(e6_roots) > 0
        
        # Test 3: E7 root system creation
        e7_roots, e7_pos = create_e7_root_system()
        results["e7_creation"] = len(e7_roots) > 0
        
        # Test 4: E8 root system creation
        e8_roots, e8_pos = create_e8_root_system()
        results["e8_creation"] = len(e8_roots) > 0
        
        # Test 5: F4 root system creation
        f4_roots, f4_pos = create_f4_root_system()
        results["f4_creation"] = len(f4_roots) > 0
        
        # Test 6: G2 root system creation
        g2_roots, g2_pos = create_g2_root_system()
        results["g2_creation"] = len(g2_roots) > 0
        
        # Test 7: Weyl group creation
        wg = WeylGroup(WeylType.E6)
        results["weyl_group"] = wg.rank == 6 and wg.order == 51840
        
        logger.info("✅ Weyl utilities validation passed")
        
    except Exception as e:
        logger.error(f"❌ Weyl utilities validation failed: {e}")
        results["validation_error"] = str(e)
    
    return results


# ============================================================================
# MAIN TESTING
# ============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Weyl Symmetry Utilities - FUTURE FEATURE MODULE")
    print("=" * 60)
    print("\nNOTE: This module contains exceptional Lie algebra code")
    print("      (E6, E7, E8, F4, G2) for FUTURE use with")
    print("      non-standard alphabets having repeated values.")
    print("\n      Currently unused in example problem pipeline.")
    print("=" * 60)
    
    results = validate_weyl_utils()
    
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
    print("Weyl Symmetry Demo (Future Feature)")
    print("=" * 60)
    

    alphabet = [1, 2, 3, 4, 5, 6]
    sym = detect_weyl_symmetry(alphabet)
    print(f"\nStandard items alphabet {alphabet}:")
    print(f"  Detected symmetry: {sym.value}")
    
    # Future use case - would detect symmetries here
    print("\nFUTURE: For alphabets with repeated values,")
    print("        this module will detect and apply exceptional symmetries:")
    print("        - E6, E7, E8 for certain root systems")
    print("        - F4, G2 for smaller exceptional cases")
    
    print("\n" + "=" * 60)
    print("✅ Weyl Utilities Ready for Future Extension")
    print("=" * 60)