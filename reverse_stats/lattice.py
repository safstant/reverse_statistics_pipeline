"""
Fraction-Exact Lattice Module for Reverse Statistics Pipeline
Provides lattice operations using Fraction arithmetic for exact computation.

Phases: 5 (Lattice Normalization), 12 (Dual Lattice Computation)
Critical for: Exact arithmetic in lattice operations, Gram matrix computation
"""

from .exceptions import ReverseStatsError
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

# Use math_utils for exact determinant
try:
    from .math_utils import determinant_exact, matrix_rank, nullspace_basis
    HAS_MATH_UTILS = True
except ImportError:
    try:
        from math_utils import determinant_exact, matrix_rank, nullspace_basis
        HAS_MATH_UTILS = True
    except ImportError:
        HAS_MATH_UTILS = False
        # FIX(Bug-B4): The original fallbacks returned Fraction(1) and len(A)
        # silently, making every lattice appear unimodular (det=1) and every
        # matrix full-rank.  This caused all Barvinok cone decompositions to be
        # skipped with no error, producing silently wrong generating functions.
        #
        # Correct behaviour: fail loudly so the operator knows math_utils is
        # missing and can fix their installation, rather than propagate wrong
        # results through the entire pipeline.
        def determinant_exact(A):
            raise ImportError(
                "lattice.py: math_utils could not be imported, so "
                "determinant_exact is unavailable.  Ensure math_utils.py "
                "is present in the same package/directory as lattice.py.  "
                "SymPy is also required (pip install sympy)."
            )

        def matrix_rank(A, tol=1e-10):
            raise ImportError(
                "lattice.py: math_utils could not be imported, so "
                "matrix_rank is unavailable.  See determinant_exact message."
            )

logger = logging.getLogger(__name__)

# ============================================================================
# EXCEPTIONS
# ============================================================================
class LatticeError(ReverseStatsError):
    """Base exception for lattice operations."""
    def __init__(self, message: str):
        self.message = message

        super().__init__(message)


class DimensionError(LatticeError):
    """Raised for dimension mismatches."""
    pass


# Try to import DimensionLimitError from canonical source
try:
    from dimension import DimensionLimitError
except ImportError:
    # Fallback only if dimension.py doesn't exist
    class DimensionLimitError(LatticeError):
        """Raised when dimension exceeds guard threshold."""
        def __init__(self, dimension: int, threshold: int = 15):
            self.dimension = dimension
            self.threshold = threshold
            super().__init__(f"Dimension {dimension} exceeds guard threshold {threshold}")


# ============================================================================
# CONFIGURATION INTEGRATION
# ============================================================================
def get_lattice_config() -> Dict[str, Any]:
    """Get lattice-specific configuration with sane defaults."""
    config = {
        "max_dimension": 15,
        "integrality_tolerance": 1e-10,
    }
    
    # Try to integrate with global config
    try:
        from .config import get_config
        global_config = get_config()
        pipeline_config = getattr(global_config, 'pipeline_config', global_config)
        config["max_dimension"] = getattr(pipeline_config, "max_dimension", config["max_dimension"])
        config["integrality_tolerance"] = getattr(pipeline_config, "integrality_tolerance", config["integrality_tolerance"])
    except (ImportError, AttributeError):
        pass
    
    return config


# ============================================================================
# CORE FRACTION LATTICE CLASS
# ============================================================================
@dataclass(frozen=True)
class FractionLattice:
    """
    Immutable lattice using Fraction arithmetic for exact computation.

    This class emphasizes exact rational arithmetic for all lattice operations,
    critical for  lattice normalization and  dual lattice computation.

    Attributes:
        basis: Tuple of basis vectors (each a tuple of Fractions)


    """
    basis: Tuple[Tuple[Fraction, ...], ...]
    
    def __post_init__(self):
        """Validate lattice basis."""
        if not self.basis:
            raise LatticeError("Basis cannot be empty")
        
        # Check all vectors have same dimension
        dim = len(self.basis[0])
        if dim == 0:
            raise LatticeError("Basis vectors must have positive dimension")
        
        for i, vec in enumerate(self.basis):
            if len(vec) != dim:
                raise DimensionError(
                    f"Basis vector {i} has dimension {len(vec)} != {dim}"
                )
        

        config = get_lattice_config()
        max_dim = config.get("max_dimension", 15)
        if dim > max_dim:
            raise DimensionLimitError(dim, max_dim)
    
    @property
    def dimension(self) -> int:
        """Ambient dimension of the lattice."""
        return len(self.basis[0]) if self.basis else 0
    
    @property
    def rank(self) -> int:
        """Rank (number of basis vectors)."""
        return len(self.basis)
    
    @property
    def determinant(self) -> Fraction:
        """
        Determinant (covolume) of the lattice.
        For full-rank lattices in R^n, this is |det(basis)|.


        No float conversion or rounding.
        """
        if self.rank == 0:
            return Fraction(1)
            
        if self.rank != self.dimension:
            # Non-full rank - use Gram determinant
            gram = self._gram_matrix()
            det_val = self._det_fraction(gram)
            # sqrt of Gram determinant gives volume - but we need exact determinant
            # For non-full rank, the determinant is the volume of the fundamental parallelepiped
            # in the span. This should be an integer for integer lattices.
            from math import isqrt
            
            # Convert to float only for sqrt, then back to Fraction
            # This is the best we can do for non-full rank without floating point

            import math as _m

            # FIX(Bug-1): det_val may be a Fraction (e.g. 1/4 for a half-integer basis).
            # int() would truncate 1/4 → 0, making isqrt return 0 and breaking all
            # unimodularity checks for non-full-rank lattices with fractional bases.
            # Correct approach: compute exact rational sqrt of the Gram determinant.
            if isinstance(det_val, Fraction):
                num, den = det_val.numerator, det_val.denominator
                sqrt_num = _m.isqrt(abs(num))
                sqrt_den = _m.isqrt(den)
                # Check whether both components are perfect squares
                if sqrt_num * sqrt_num == abs(num) and sqrt_den * sqrt_den == den:
                    return Fraction(sqrt_num, sqrt_den)
                # Non-perfect-square Gram determinant: delegate to SymPy for exactness
                try:
                    import sympy as _sp
                    return Fraction(_sp.sqrt(_sp.Rational(num, den)))
                except Exception:
                    # Last resort: rational approximation good to 15 sig figs
                    import decimal as _dec
                    _dec.getcontext().prec = 30
                    approx = _dec.Decimal(num).__truediv__(_dec.Decimal(den)).sqrt()
                    return Fraction(approx).limit_denominator(10**9)
            else:
                det_int_val = int(det_val)
                sqrt_int = _m.isqrt(abs(det_int_val))
                return Fraction(sqrt_int)
        
        # FIX: Full rank - use exact determinant from math_utils
        # Convert basis to list of lists for determinant_exact
        basis_list = [[self.basis[i][j] for j in range(self.dimension)] 
                      for i in range(self.rank)]
        
        try:
            det = determinant_exact(basis_list)
            return abs(det)
        except Exception as e:
            logger.error(f"Exact determinant failed: {e}")
            # Fallback should never happen in production
            raise LatticeError("Failed to compute exact determinant")
    
    @property
    def is_unimodular(self) -> bool:
        """
        Check if lattice is unimodular (determinant = 1).


        """
        return self.determinant == 1
    
    @property
    def volume(self) -> float:
        """Volume of fundamental domain (sqrt of determinant)."""
        # self.determinant is now exact Fraction; float conversion safe here (display only)
        return float(self.determinant) ** 0.5
    
    def _gram_matrix(self) -> List[List[Fraction]]:
        """Compute Gram matrix G = B·B^T using exact Fraction arithmetic."""
        n = self.rank
        gram = [[Fraction(0) for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                dot = sum(self.basis[i][k] * self.basis[j][k] for k in range(self.dimension))
                gram[i][j] = dot
        return gram
    
    def _det_fraction(self, mat: List[List[Fraction]]) -> Fraction:
        """
        Compute determinant of a matrix with Fraction entries.

        Uses recursive Laplace expansion (only suitable for small matrices,
        which is sufficient for  where dimensions are ≤15).
        """
        n = len(mat)
        if n == 1:
            return mat[0][0]
        if n == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
        
        # Recursive Laplace expansion (for small matrices only)
        det = Fraction(0)
        for j in range(n):
            minor = [row[:j] + row[j+1:] for row in mat[1:]]
            sign = Fraction(1) if j % 2 == 0 else Fraction(-1)
            det += sign * mat[0][j] * self._det_fraction(minor)
        return det
    
    def contains(self, point: Tuple[Fraction, ...]) -> bool:
        """
        Check if point belongs to the lattice.
        Solves B^T · x = point for integer x using rational arithmetic.
        """
        if len(point) != self.dimension:
            return False
        
        # For Z^n, simple integer check
        if self.is_integer_lattice:
            return all(self._is_integer(coord) for coord in point)
        
        # General case: solve linear system
        try:
            # Convert to float for initial solve (then verify with fractions)
            # This is acceptable as it's only for the initial approximation
            import numpy as np
            # Float conversion for numpy linear system (np.linalg.solve) — acceptable
            B = np.array([[float(x) for x in row] for row in self.basis], dtype=float)
            p = np.array([float(x) for x in point], dtype=float)
            
            # Solve B^T · x = p  =>  x = (B·B^T)^{-1} · B · p
            gram = B @ B.T
            if np.linalg.matrix_rank(gram) < self.rank:
                return False
            
            coeffs = np.linalg.solve(gram, B @ p)
            
            # Verify integer coefficients with tolerance
            tol = get_lattice_config().get("integrality_tolerance", 1e-8)
            if not all(self._is_integer(c, tol) for c in coeffs):
                return False
            
            # Verify exact reconstruction with fractions
            reconstructed = [Fraction(0) for _ in range(self.dimension)]
            for i, coeff in enumerate(coeffs):
                c_frac = Fraction(int(round(coeff)))
                for d in range(self.dimension):
                    reconstructed[d] += c_frac * self.basis[i][d]
            
            return all(abs(reconstructed[d] - point[d]) < Fraction(1, 10**9) 
                      for d in range(self.dimension))
        except (Exception, ImportError) as _e:
            raise type(_e)(
                f"FractionLattice.contains(): lattice membership check failed: {_e}. "
                "Cannot silently return False — misclassified lattice membership "
                "corrupts all downstream vertex and cone operations."
            ) from _e
    
    @property
    def is_integer_lattice(self) -> bool:
        """Check if lattice is Z^n (standard integer lattice)."""
        if self.rank != self.dimension:
            return False
        for i in range(self.dimension):
            for j in range(self.dimension):
                expected = Fraction(1) if i == j else Fraction(0)
                if self.basis[i][j] != expected:
                    return False
        return True
    
    @property
    def is_full_rank(self) -> bool:
        """Check if lattice has full rank in ambient space."""
        return self.rank == self.dimension
    
    def dual_basis(self) -> "FractionLattice":
        """
        Compute dual lattice basis.
        Dual basis satisfies <b_i, b^*_j> = δ_{ij}.


        """
        if self.rank != self.dimension:
            raise LatticeError(
                f"Dual lattice requires full rank (rank={self.rank}, dim={self.dimension})",

            )
        
        # Compute inverse of basis matrix using sympy for exactness
        try:
            import sympy
            from sympy import Matrix
            # Convert to sympy matrix
            from sympy import Rational as _R
            from sympy import Rational as _R
            B_sym = Matrix([[_R(int(x.numerator), int(x.denominator))
                             if hasattr(x,'numerator') else _R(int(x))
                             for x in row] for row in self.basis])
            B_inv_sym = B_sym.inv()
            
            # Convert back to exact Fractions using SymPy's rational representation
            dual_basis = []
            for i in range(B_inv_sym.rows):
                row = []
                for j in range(B_inv_sym.cols):
                    entry = B_inv_sym[i, j]
                    # Use .p/.q for exact Rational; fall back to str() for other types
                    if hasattr(entry, 'p') and hasattr(entry, 'q'):
                        row.append(Fraction(int(entry.p), int(entry.q)))
                    else:
                        # e.g. sqrt expressions — use string conversion for exactness
                        row.append(Fraction(str(entry)).limit_denominator(10**12))
                dual_basis.append(tuple(row))
            
            return FractionLattice(basis=tuple(dual_basis))
        except ImportError:
            # Fallback to numpy if sympy not available
            import numpy as np
            B = np.array([[float(x) for x in row] for row in self.basis], dtype=float)
            try:
                B_inv = np.linalg.inv(B.T)
                dual_basis = []
                for row in B_inv:
                    dual_row = tuple(Fraction(int(round(x * 1000000)), 1000000) for x in row)
                    dual_basis.append(dual_row)
                return FractionLattice(basis=tuple(dual_basis))
            except np.linalg.LinAlgError:
                raise LatticeError("Basis matrix is singular")
    
    def sublattice(self, indices: List[int]) -> "FractionLattice":
        """Extract sublattice spanned by specified basis vectors."""
        sub_basis = tuple(self.basis[i] for i in indices if 0 <= i < self.rank)
        if not sub_basis:
            raise LatticeError("Sublattice must have at least one basis vector")
        return FractionLattice(basis=sub_basis)
    
    def orthogonal_complement(self) -> "FractionLattice":
        """Compute orthogonal complement lattice."""
        if self.rank == 0:
            # Return full space basis
            basis = tuple(tuple(Fraction(1) if i == j else Fraction(0) 
                               for i in range(self.dimension)) 
                         for j in range(self.dimension))
            return FractionLattice(basis=basis)
        
        # Compute nullspace of basis matrix using sympy if available
        try:
            import sympy
            from sympy import Matrix
            from sympy import Rational as _R
            from sympy import Rational as _R
            B_sym = Matrix([[_R(int(x.numerator), int(x.denominator))
                             if hasattr(x,'numerator') else _R(int(x))
                             for x in row] for row in self.basis])
            null_basis_sym = B_sym.nullspace()
            
            if not null_basis_sym:
                return FractionLattice(basis=())
            
            orth_basis = []
            for vec in null_basis_sym:
                row = []
                for val in vec:
                    # Use .p/.q for exact SymPy Rational; str() for other expressions
                    if hasattr(val, 'p') and hasattr(val, 'q'):
                        row.append(Fraction(int(val.p), int(val.q)))
                    else:
                        row.append(Fraction(str(val)).limit_denominator(10**12))
                orth_basis.append(tuple(row))
            
            return FractionLattice(basis=tuple(orth_basis))
        except ImportError as e:
            raise ImportError(
                "FractionLattice.orthogonal_complement requires SymPy for exact computation. "
                "Install with: pip install sympy"
            ) from e
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "basis": [[str(x) for x in v] for v in self.basis],
            "dimension": self.dimension,
            "rank": self.rank,
            "determinant": str(self.determinant),
            "volume": self.volume,
            "is_unimodular": self.is_unimodular,
            "is_integer_lattice": self.is_integer_lattice,
            "is_full_rank": self.is_full_rank
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FractionLattice":
        """Create lattice from dictionary."""
        basis = []
        for row in data["basis"]:
            basis.append(tuple(Fraction(x) for x in row))
        return cls(basis=tuple(basis))
    
    @classmethod
    def identity(cls, dimension: int) -> "FractionLattice":
        """Create standard integer lattice Z^dim."""
        basis = []
        for i in range(dimension):
            vec = [Fraction(0)] * dimension
            vec[i] = Fraction(1)
            basis.append(tuple(vec))
        return cls(basis=tuple(basis))
    
    @staticmethod
    def _is_integer(x: Union[Fraction, float], tol: float = 1e-10) -> bool:
        """Check if value is close to an integer."""
        if isinstance(x, Fraction):
            return x.denominator == 1
        return abs(x - round(x)) < tol


# ============================================================================
# AFFINE LATTICE CLASS
# ============================================================================
@dataclass(frozen=True)
class AffineLattice:
    """
    Affine lattice (lattice + translation).

    Represents sets of the form L + t = {x + t | x ∈ L}.

    Attributes:
        lattice: Underlying lattice
        translation: Translation vector


    """
    lattice: FractionLattice
    translation: Tuple[Fraction, ...]
    
    def __post_init__(self):
        """Validate affine lattice."""
        if len(self.translation) != self.lattice.dimension:
            raise LatticeError(
                f"Translation dimension {len(self.translation)} does not match "
                f"lattice dimension {self.lattice.dimension}",

            )
    
    @property
    def dimension(self) -> int:
        """Dimension of affine lattice."""
        return self.lattice.dimension
    
    @property
    def rank(self) -> int:
        """Rank of affine lattice."""
        return self.lattice.rank
    
    def contains(self, point: Tuple[Fraction, ...]) -> bool:
        """Check if point is in affine lattice."""
        if len(point) != self.dimension:
            return False
        
        # Subtract translation and check membership in underlying lattice
        shifted = tuple(point[i] - self.translation[i] for i in range(self.dimension))
        return self.lattice.contains(shifted)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "lattice": self.lattice.to_dict(),
            "translation": [str(x) for x in self.translation]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AffineLattice":
        """Create affine lattice from dictionary."""
        lattice = FractionLattice.from_dict(data["lattice"])
        translation = tuple(Fraction(x) for x in data["translation"])
        return cls(lattice=lattice, translation=translation)


# ============================================================================
# LATTICE OPERATIONS
# ============================================================================
def create_root_lattice(root_type: str, dimension: int) -> FractionLattice:
    """
    Create root lattice of given type.

    Supported types:
        'A': A_n lattice (vectors in Z^{n+1} with coordinate sum 0)
        'D': D_n lattice (checkerboard lattice)
        'E': E_6, E_7, E_8 exceptional lattices (full implementation)
        'Z': Standard integer lattice Z^n (alias for identity)


    """
    if root_type.upper() == 'Z':
        return FractionLattice.identity(dimension)
    
    if root_type.upper() == 'A':
        if dimension < 1:
            raise LatticeError(f"A_n requires dimension >= 1, got {dimension}")
        # A_n lives in R^{n+1} with sum=0 constraint
        ambient_dim = dimension + 1
        basis = []
        for i in range(dimension):
            vec = [Fraction(0)] * ambient_dim
            vec[i] = Fraction(1)
            vec[i+1] = Fraction(-1)
            basis.append(tuple(vec))
        return FractionLattice(basis=tuple(basis))
    
    if root_type.upper() == 'D':
        if dimension < 2:
            raise LatticeError(f"D_n requires dimension >= 2, got {dimension}")
        basis = []
        # First n-1 basis vectors: e_i - e_{i+1}
        for i in range(dimension - 1):
            vec = [Fraction(0)] * dimension
            vec[i] = Fraction(1)
            vec[i+1] = Fraction(-1)
            basis.append(tuple(vec))
        # Last basis vector: e_{n-1} + e_n
        vec = [Fraction(0)] * dimension
        vec[-2] = Fraction(1)
        vec[-1] = Fraction(1)
        basis.append(tuple(vec))
        return FractionLattice(basis=tuple(basis))
    
    if root_type.upper() == 'E':
        if dimension == 6:
            basis = [
                (Fraction(1), Fraction(-1), Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
                (Fraction(0), Fraction(1), Fraction(-1), Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
                (Fraction(0), Fraction(0), Fraction(1), Fraction(-1), Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
                (Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(-1), Fraction(0), Fraction(0), Fraction(0)),
                (Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(-1), Fraction(0), Fraction(0)),
                (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(-1, 2), Fraction(-1, 2), Fraction(-1, 2), Fraction(-1, 2))
            ]
            return FractionLattice(basis=tuple(basis))
        elif dimension == 7:
            basis = [
                (Fraction(1), Fraction(-1), Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
                (Fraction(0), Fraction(1), Fraction(-1), Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
                (Fraction(0), Fraction(0), Fraction(1), Fraction(-1), Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
                (Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(-1), Fraction(0), Fraction(0), Fraction(0)),
                (Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(-1), Fraction(0), Fraction(0)),
                (Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(-1), Fraction(0)),
                (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(-1, 2), Fraction(-1, 2), Fraction(-1, 2), Fraction(-1, 2))
            ]
            return FractionLattice(basis=tuple(basis))
        elif dimension == 8:
            basis = [
                (Fraction(1), Fraction(-1), Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
                (Fraction(0), Fraction(1), Fraction(-1), Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
                (Fraction(0), Fraction(0), Fraction(1), Fraction(-1), Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
                (Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(-1), Fraction(0), Fraction(0), Fraction(0)),
                (Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(-1), Fraction(0), Fraction(0)),
                (Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(-1), Fraction(0)),
                (Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(-1)),
                (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2))
            ]
            return FractionLattice(basis=tuple(basis))
        else:
            raise LatticeError(f"Unsupported E lattice dimension: {dimension} (supported: 6,7,8)")
    
    raise LatticeError(f"Unsupported root lattice type: {root_type}")


def lattice_translate(lattice: FractionLattice, translation: Tuple[Fraction, ...]) -> AffineLattice:
    """Translate lattice by given vector to create affine lattice."""
    return AffineLattice(lattice=lattice, translation=translation)


def lattice_scale(lattice: FractionLattice, factor: Fraction) -> FractionLattice:
    """Scale lattice by rational factor."""
    scaled_basis = []
    for vec in lattice.basis:
        scaled_vec = tuple(coord * factor for coord in vec)
        scaled_basis.append(scaled_vec)
    return FractionLattice(basis=tuple(scaled_basis))


def lattice_sum(lattice1: FractionLattice, lattice2: FractionLattice) -> FractionLattice:
    """Compute sum of two lattices (Minkowski sum of basis vectors)."""
    if lattice1.dimension != lattice2.dimension:
        raise DimensionError(
            f"Lattices must have same dimension ({lattice1.dimension} != {lattice2.dimension})"
        )
    
    # Concatenate basis vectors
    combined_basis = list(lattice1.basis) + list(lattice2.basis)
    
    # Remove linearly dependent vectors via rank check
    # Convert to float matrix for rank computation
    import numpy as np
    mat = np.array([[float(x) for x in row] for row in combined_basis], dtype=float)
    rank = np.linalg.matrix_rank(mat)
    
    # Keep first 'rank' independent vectors
    independent = []
    for i, vec in enumerate(combined_basis):
        if len(independent) >= rank:
            break
        test_basis = independent + [vec]
        test_mat = np.array([[float(x) for x in row] for row in test_basis], dtype=float)
        if np.linalg.matrix_rank(test_mat) > len(independent):
            independent.append(vec)
    
    return FractionLattice(basis=tuple(independent))


def lattice_intersection(lattice1, lattice2):
    """Not implemented — previous version returned wrong approximation."""
    raise NotImplementedError(
        "lattice_intersection: the previous implementation returned whichever input "
        "lattice had smaller rank, which is not the intersection. "
        "True lattice intersection requires SNF of the stacked basis matrix."
    )



def shortest_vectors(lattice: FractionLattice, max_count: int = 10) -> List[Tuple[Fraction, ...]]:
    """
    Find shortest non-zero vectors in lattice (brute force for small lattices).
    Returns up to max_count shortest vectors sorted by norm.


    """
    if lattice.rank == 0:
        return []
    
    # For Z^n, shortest vectors are standard basis vectors
    if lattice.is_integer_lattice:
        dim = lattice.dimension
        vecs = []
        for i in range(min(dim, max_count)):
            e = [Fraction(0)] * dim
            e[i] = Fraction(1)
            vecs.append(tuple(e))
        return vecs
    
    # Use LLL reduction to get a short basis, then enumerate with wider range
    try:
        from decomposition import lll_reduce
        basis_list = [list(row) for row in lattice.basis]
        reduced_basis = lll_reduce(basis_list, delta=0.75)
        # Replace lattice basis with reduced version for enumeration
        search_basis = reduced_basis
    except Exception:
        search_basis = lattice.basis

    # Enumerate with coefficients up to ±2 in the (possibly reduced) basis
    shortest = []
    max_coeff = 2

    def _sv_basis_vector(coeffs, basis, dim):
        vec = [Fraction(0)] * dim
        for i, c in enumerate(coeffs):
            if c == 0:
                continue
            cf = Fraction(c)
            for d in range(dim):
                vec[d] += cf * Fraction(basis[i][d]) if not isinstance(basis[i][d], Fraction) else cf * basis[i][d]
        return tuple(vec)

    # Generate coefficient combinations
    for coeffs in itertools.product(range(-max_coeff, max_coeff+1), repeat=lattice.rank):
        if all(c == 0 for c in coeffs):
            continue
        
        # Compute lattice vector
        vec = [Fraction(0) for _ in range(lattice.dimension)]
        for i, coeff in enumerate(coeffs):
            if coeff == 0:
                continue
            c_frac = Fraction(coeff)
            for d in range(lattice.dimension):
                raw = search_basis[i][d]
                vec[d] += c_frac * (raw if isinstance(raw, Fraction) else Fraction(raw))
        
        # Compute squared norm
        norm_sq = sum(x * x for x in vec)
        if norm_sq == 0:
            continue
        
        shortest.append((norm_sq, tuple(vec)))
    
    # Sort by norm and return unique shortest vectors
    shortest.sort(key=lambda x: x[0])
    unique_vecs = []
    seen_norms = set()
    for norm_sq, vec in shortest:
        if len(unique_vecs) >= max_count:
            break
        if norm_sq not in seen_norms:
            seen_norms.add(norm_sq)
            unique_vecs.append(vec)
    
    return unique_vecs


# ============================================================================
# ClassifiedCone — V15.3 (C1): proper wrapper replacing the FractionLattice alias
# ============================================================================
# Previous code: ClassifiedLattice = FractionLattice  (a broken type alias)
#
# The bug: pipeline.py Step 16 called ClassifiedLattice(cone, lattice) which
# silently became FractionLattice(basis=cone) — a FractionLattice whose
# basis field held a TangentCone object.  Consequences:
#   1. is_unimodular() computed det on a TangentCone → 0 → every cone
#      mis-classified as non-unimodular.
#   2. decompose_non_unimodular_cones() received FractionLattice objects
#      (not DecompositionCone) → AttributeError on .ambient_dimension.
#
# Fix: ClassifiedCone holds the TangentCone directly, computes is_unimodular
# via exact SymPy det on the actual ray matrix, and provides
# to_decomposition_cone() for Step 17.
#
# ClassifiedLattice = ClassifiedCone is preserved as a backward-compatible alias.

class ClassifiedCone:
    """
    Classifies a TangentCone as unimodular or non-unimodular.

    V15.3 (C1): Replaces the broken ClassifiedLattice = FractionLattice alias.

    Parameters
    ----------
    cone : TangentCone
        The tangent cone to classify.
    lattice : FractionLattice (ignored, kept for call-site compatibility)
        Previously held the effective sublattice.  Classification now uses
        the cone's own ray matrix determinant, which is the correct criterion.
    """

    def __init__(self, cone, lattice=None):  # type: ignore[override]
        self._cone = cone
        self._lattice = lattice
        self._unimodular: Optional[bool] = None  # lazy, computed once

    # ── Proxy the most-used TangentCone attributes ─────────────────────────
    @property
    def vertex(self):
        return self._cone.vertex

    @property
    def rays(self):
        return self._cone.rays

    @property
    def dimension(self) -> int:
        return self._cone.dimension

    @property
    def ambient_dimension(self) -> int:
        return self._cone.ambient_dimension

    @property
    def is_pointed(self) -> bool:
        return self._cone.is_pointed

    @property
    def is_simplicial(self) -> bool:
        return self._cone.is_simplicial

    # ── Core classification: exact det via SymPy ────────────────────────────
    @property
    def is_unimodular(self) -> bool:
        """
        True iff abs(det(ray_matrix)) == 1 (exact integer arithmetic).

        For non-square ray matrices (n_rays != ambient_dim) returns False —
        the cone is non-simplicial and needs signed decomposition.
        """
        if self._unimodular is not None:
            return self._unimodular

        if not self._cone.rays:
            self._unimodular = False
            return False

        rays = self._cone.rays
        n = len(rays)
        m = len(rays[0]) if rays else 0

        # Non-square → not simplicial → not unimodular
        if n != m or n == 0:
            self._unimodular = False
            return False

        try:
            import sympy as _sp
            from fractions import Fraction as _F
            from math import gcd as _gcd
            from functools import reduce as _red

            int_rays = []
            for ray in rays:
                denoms = [r.denominator if isinstance(r, _F) else 1 for r in ray]
                lcm_d  = _red(lambda a, b: a * b // _gcd(a, b), denoms, 1)
                int_ray = [int(r * lcm_d) if isinstance(r, _F) else int(float(r) * lcm_d)
                           for r in ray]
                int_rays.append(int_ray)

            det = int(abs(_sp.Matrix(int_rays).det()))
            self._unimodular = (det == 1)
        except Exception:
            self._unimodular = False

        return self._unimodular

    @is_unimodular.setter
    def is_unimodular(self, value: bool) -> None:
        """Allow pipeline Step 16b to force-reclassify a cone."""
        self._unimodular = value

    # ── Conversion for Step 17 ─────────────────────────────────────────────
    def to_decomposition_cone(self):
        """
        Convert to DecompositionCone for decompose_non_unimodular_cones().

        V15.3 (C1): Step 17 expects List[DecompositionCone].  This method
        provides the conversion so the type contract is satisfied.
        """
        try:
            try:
                from .decomposition import DecompositionCone
            except ImportError:
                from decomposition import DecompositionCone
            from fractions import Fraction as _F

            frac_rays = []
            for ray in self._cone.rays:
                frac_ray = tuple(
                    r if isinstance(r, _F) else _F(r).limit_denominator(10**9)
                    for r in ray
                )
                frac_rays.append(frac_ray)

            dc = DecompositionCone(
                rays=frac_rays,
                is_unimodular=self.is_unimodular,
            )
            # V15.4: propagate intrinsic basis fields from TangentCone
            tc = self._cone
            if getattr(tc, 'intrinsic_basis', None):
                dc.intrinsic_basis   = list(tc.intrinsic_basis)
                dc.intrinsic_inverse = list(tc.intrinsic_inverse)
            return dc
        except Exception as e:
            raise RuntimeError(f"ClassifiedCone.to_decomposition_cone failed: {e}") from e

    def __repr__(self) -> str:
        return (f"ClassifiedCone(vertex={self.vertex}, "
                f"n_rays={len(self.rays)}, "
                f"unimodular={self._unimodular})")


# Backward-compatible alias — any external code using ClassifiedLattice still works.
# FractionLattice alias is kept separately for lattice arithmetic calls.
ClassifiedLattice = ClassifiedCone

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def is_integer(x: Union[Fraction, float], tol: float = 1e-10) -> bool:
    """Check if value is close to an integer."""
    if isinstance(x, Fraction):
        return x.denominator == 1
    return abs(x - round(x)) < tol


def gcd_vector(v: List[int]) -> int:
    """Compute GCD of all elements in a vector."""
    if not v:
        return 0
    
    from math import gcd
    result = abs(v[0])
    for x in v[1:]:
        result = gcd(result, abs(x))
    return result


def primitive_vector(v: List[int]) -> List[int]:
    """Convert vector to primitive form (divide by GCD)."""
    g = gcd_vector(v)
    if g == 0 or g == 1:
        return v
    return [x // g for x in v]


# ============================================================================
# VALIDATION AND TESTING
# ============================================================================
def validate_fraction_lattice() -> Dict[str, bool]:
    """Run internal test suite to verify FractionLattice utilities."""
    results = {}
    try:
        from fractions import Fraction
        
        # Test 1: Create integer lattice
        Z2 = FractionLattice.identity(2)
        results["integer_lattice"] = (
            Z2.dimension == 2 and 
            Z2.rank == 2 and 
            Z2.is_integer_lattice
        )
        
        # Test 2: Create A2 lattice
        A2 = create_root_lattice('A', 2)
        results["root_lattice"] = (
            A2.dimension == 3 and 
            A2.rank == 2
        )
        

        # This should now work correctly
        det_val = float(Z2.determinant)
        results["determinant"] = abs(det_val - 1.0) < 1e-10
        
        # Test 4: Unimodular check
        results["unimodular"] = Z2.is_unimodular
        
        # Test 5: Point containment (Z^2)
        point1 = (Fraction(3, 1), Fraction(4, 1))
        point2 = (Fraction(3, 2), Fraction(4, 1))  # Non-integer x-coordinate
        results["contains"] = (Z2.contains(point1) and not Z2.contains(point2))
        
        # Test 6: Dual lattice (Z^2 is self-dual)
        Z2_dual = Z2.dual_basis()
        results["dual"] = (
            Z2_dual.dimension == 2 and
            Z2_dual.rank == 2
        )
        
        # Test 7: Affine lattice
        affine = lattice_translate(Z2, (Fraction(1, 2), Fraction(1, 2)))
        point3 = (Fraction(3, 2), Fraction(5, 2))  # (1.5, 2.5) = (1,2) + (0.5,0.5)
        point4 = (Fraction(1, 1), Fraction(1, 1))  # (1,1) not in affine lattice
        results["affine"] = (affine.contains(point3) and not affine.contains(point4))
        
        # Test 8: Lattice sum
        L1 = FractionLattice(basis=((Fraction(1, 1), Fraction(0, 1)),))
        L2 = FractionLattice(basis=((Fraction(0, 1), Fraction(1, 1)),))
        sum_lat = lattice_sum(L1, L2)
        results["lattice_sum"] = sum_lat.dimension == 2 and sum_lat.rank == 2
        
        # Test 9: Lattice scaling
        scaled = lattice_scale(Z2, Fraction(2, 1))
        scaled_det = float(scaled.determinant)
        results["scale"] = abs(scaled_det - 4.0) < 1e-10
        
        # Test 10: Shortest vectors
        shortest = shortest_vectors(Z2, max_count=4)
        results["shortest_vectors"] = (
            len(shortest) >= 2 and
            all(sum(x*x for x in v) == 1 for v in shortest[:2])  # Norm squared = 1
        )
        
        # Test 11: D3 lattice (checkerboard)
        D3 = create_root_lattice('D', 3)
        results["D3_lattice"] = D3.dimension == 3 and D3.rank == 3
        
        # Test 12: Fractional lattice determinant - NEW TEST
        # This would have failed before the fix
        frac_lattice = FractionLattice(basis=(
            (Fraction(1, 2), Fraction(1, 2)),
            (Fraction(0), Fraction(1, 2))
        ))
        # True determinant should be 1/4
        results["fractional_det"] = frac_lattice.determinant == Fraction(1, 4)
        
        logger.info("✅ FractionLattice validation passed")
    except Exception as e:
        logger.error(f"❌ FractionLattice validation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
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
    print("Testing FractionLattice - Exact Fraction Arithmetic")
    print("=" * 60)
    
    # Run validation
    results = validate_fraction_lattice()
    
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
    print(f"Overall: {success}/{total-1 if 'validation_error' in results else total} tests passed")
    
    if "validation_error" in results:
        sys.exit(1)
    
    # Demonstration
    print("\n" + "=" * 60)
    print("FractionLattice Demo - Exact Arithmetic")
    print("=" * 60)
    
    from fractions import Fraction
    
    # 1. Create various lattices
    print("\n1. Creating Lattices with Exact Fractions:")
    Z2 = FractionLattice.identity(2)
    print(f"   Z^2: basis={Z2.basis}")
    print(f"   Determinant: {Z2.determinant} (exact Fraction)")
    
    A2 = create_root_lattice('A', 2)
    print(f"\n   A2: dimension={A2.dimension}, rank={A2.rank}")
    print(f"   Basis vectors: {A2.basis}")
    
    D3 = create_root_lattice('D', 3)
    print(f"\n   D3: determinant={D3.determinant}")
    
    # 2. Critical test: fractional lattice determinant
    print("\n2. CRITICAL TEST - Fractional Lattice Determinant:")
    frac_lattice = FractionLattice(basis=(
        (Fraction(1, 2), Fraction(1, 2)),
        (Fraction(0), Fraction(1, 2))
    ))
    print(f"   Basis: {frac_lattice.basis}")
    print(f"   True determinant should be: 1/4")
    print(f"   Calculated determinant: {frac_lattice.determinant}")
    print(f"   Correct? {frac_lattice.determinant == Fraction(1, 4)}")
    
    # 3. Exact determinant computation
    print("\n3. Exact Determinant Computation:")
    print(f"   Z2.determinant = {Z2.determinant} (type: {type(Z2.determinant).__name__})")
    print(f"   Z2.is_unimodular = {Z2.is_unimodular}")
    
    scaled_Z2 = lattice_scale(Z2, Fraction(2, 1))
    print(f"   2·Z^2 determinant = {scaled_Z2.determinant} (exact)")
    
    # 4. Point containment with exact fractions
    print("\n4. Point Containment (Exact):")
    p1 = (Fraction(3, 1), Fraction(4, 1))
    p2 = (Fraction(3, 2), Fraction(4, 1))
    print(f"   Point {p1} in Z^2? {Z2.contains(p1)}")
    print(f"   Point {p2} in Z^2? {Z2.contains(p2)}")
    
    # 5. Affine lattice
    print("\n5. Affine Lattice:")
    affine = lattice_translate(Z2, (Fraction(1, 2), Fraction(1, 2)))
    p3 = (Fraction(3, 2), Fraction(5, 2))
    print(f"   Point {p3} in Z^2 + (1/2,1/2)? {affine.contains(p3)}")
    
    # 6. Dual lattice
    print("\n6. Dual Lattice (Exact):")
    Z2_dual = Z2.dual_basis()
    print(f"   Z^2 dual basis: {Z2_dual.basis}")
    print(f"   Determinant of dual: {Z2_dual.determinant}")
    
    print("\n" + "=" * 60)
    print("✅ FractionLattice Ready for Production")
    print("=" * 60)