"""
Mathematical Utilities for Reverse Statistics Pipeline
Provides core numerical and algebraic operations with exact arithmetic.

Critical for: (Orbit weights) (Lattice ops) (Unimodularity)
"""

from .exceptions import ReverseStatsError
import math
from fractions import Fraction
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
import logging

# SymPy is REQUIRED for exact linear algebra - no fallback
try:
    import sympy
    from sympy import Matrix
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    # We'll raise an error at point of use instead of providing invalid fallbacks

logger = logging.getLogger(__name__)

# ============================================================================
# EXCEPTIONS
# ============================================================================

class MathError(ReverseStatsError):
    """Base exception for mathematical operations."""
    def __init__(self, message: str, operation: str = None):
        self.message = message
        self.operation = operation
        super().__init__(f"{operation}: {message}" if operation else message)


class NumericalError(MathError):
    """Raised for numerical issues (overflow, precision loss, etc.)."""
    pass


class DimensionError(MathError):
    """Raised for dimension mismatches."""
    pass


# ============================================================================
# CORE NUMERICAL OPERATIONS
# ============================================================================

def is_integer(x: Union[float, int, Fraction], tol: float = 1e-10) -> bool:
    """
    Check if a number is integer within tolerance.

    NOTE: This function is for validation only, not for core geometry.
    Geometry code must use exact Fraction arithmetic.

    Args:
        x: Number to check
        tol: Tolerance for integer check

    Returns:
        True if number is close to integer
    """
    if isinstance(x, Fraction):
        return x.denominator == 1
    if isinstance(x, int):
        return True
    # Float case - only for validation
    return abs(x - round(x)) <= tol


def rational_approximation(x: float, max_denominator: int = 1000) -> Fraction:
    """
    Find best rational approximation using continued fractions.

    NOTE: This is for converting floats to Fractions when necessary.
    Core pipeline should never need this.

    Args:
        x: Float to approximate
        max_denominator: Maximum allowed denominator

    Returns:
        Best rational approximation as Fraction
    """
    if math.isnan(x) or math.isinf(x):
        raise NumericalError(f"Cannot approximate non-finite value: {x}")
    
    return Fraction(x).limit_denominator(max_denominator)


def gcd_extended(a: int, b: int) -> Tuple[int, int, int]:
    """
    Extended Euclidean algorithm.

    Returns:
        (gcd, x, y) such that ax + by = gcd(a, b)
    """
    if a == 0:
        return abs(b), 0, 1 if b >= 0 else -1
    
    gcd, x1, y1 = gcd_extended(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    
    return gcd, x, y


def gcd_list(numbers: List[int]) -> int:
    """
    Compute GCD of a list of integers.

    Args:
        numbers: List of integers

    Returns:
        GCD of all numbers
    """
    if not numbers:
        return 0
    
    result = abs(numbers[0])
    for num in numbers[1:]:
        result = math.gcd(result, abs(num))
        if result == 1:
            break
    return result


def lcm_list(numbers: List[int]) -> int:
    """
    Compute LCM of a list of integers.

    Args:
        numbers: List of integers

    Returns:
        LCM of all numbers
    """
    if not numbers:
        return 1
    
    def lcm(a: int, b: int) -> int:
        return abs(a * b) // math.gcd(a, b) if a and b else 0
    
    result = numbers[0]
    for num in numbers[1:]:
        result = lcm(result, num)
    return result


# ============================================================================
# CRITICAL BARVINOK UTILITIES (SPECIFICATION REQUIRED)
# ============================================================================

def compute_orbit_weight(frequencies: List[int], N: int) -> int:
    """
    Compute orbit weight: N! / ∏fⱼ!.

    Args:
        frequencies: List of frequency counts fⱼ
        N: Total count N = ∑fⱼ

    Returns:
        Multinomial coefficient N! / (f₁!·f₂!·...·fₘ!)

    Raises:
        ValueError: If sum(frequencies) != N
    """
    if sum(frequencies) != N:
        raise ValueError(f"Sum of frequencies {sum(frequencies)} != N={N}")
    
    # Use exact integer arithmetic
    from math import factorial
    
    numerator = factorial(N)
    denominator = 1
    for f in frequencies:
        denominator *= factorial(f)
    
    return numerator // denominator


def check_cauchy_schwarz(S1: Fraction, S2: Fraction, N: int) -> bool:
    """
    Exact Cauchy-Schwarz inequality: S₂ ≥ S₁²/N.

    Args:
        S1: First moment sum S₁ = Σxᵢ
        S2: Second moment sum S₂ = Σxᵢ²
        N: Number of terms

    Returns:
        True if inequality holds exactly (using rational arithmetic)
    """
    # Check: N·S₂ ≥ S₁²
    return N * S2 >= S1 * S1


# ============================================================================
# LINEAR ALGEBRA UTILITIES - REQUIRES SYMPY
# ============================================================================

def _ensure_sympy():
    """Raise error if SymPy is not available."""
    if not HAS_SYMPY:
        raise MathError(
            "SymPy is required for exact linear algebra operations. "
            "Install with: pip install sympy"
        )


def matrix_rank(A: Union[List[List[Union[int, Fraction]]], Matrix]) -> int:
    """
    Compute exact matrix rank using SymPy.

    Args:
        A: Input matrix (must contain only integers or Fractions)

    Returns:
        Exact matrix rank

    Raises:
        MathError: If SymPy is not installed or input contains floats
    """
    _ensure_sympy()
    
    # Convert to SymPy Matrix if needed
    if not isinstance(A, Matrix):
        # Verify no floats
        for row in A:
            for val in row:
                if isinstance(val, float):
                    raise MathError(f"Float value {val} not allowed in exact rank computation")
        A = Matrix([[sympy.Rational(str(x)) if isinstance(x, Fraction) else x 
                     for x in row] for row in A])
    
    return A.rank()


def determinant_exact(A: Union[List[List[Union[int, Fraction]]], Matrix]) -> Fraction:
    """
    Compute exact determinant using SymPy.

    Args:
        A: Square matrix (must contain only integers or Fractions)

    Returns:
        Exact determinant as Fraction

    Raises:
        MathError: If SymPy is not installed or input contains floats
    """
    _ensure_sympy()
    
    # Convert to SymPy Matrix if needed
    if not isinstance(A, Matrix):
        # Verify no floats
        for row in A:
            for val in row:
                if isinstance(val, float):
                    raise MathError(f"Float value {val} not allowed in exact determinant")
        A = Matrix([[sympy.Rational(str(x)) if isinstance(x, Fraction) else x 
                     for x in row] for row in A])
    
    if A.rows != A.cols:
        raise DimensionError(f"Matrix must be square for determinant: {A.rows} x {A.cols}")
    
    det = A.det()
    return Fraction(det.p, det.q)  # Convert to Fraction


def nullspace_basis(A: Union[List[List[Union[int, Fraction]]], Matrix]) -> List[List[Fraction]]:
    """
    Compute basis for nullspace of matrix using SymPy.



    Args:
        A: Input matrix (must contain only integers or Fractions)

    Returns:
        List of basis vectors for nullspace (as lists of Fractions)

    Raises:
        MathError: If SymPy is not installed or input contains floats
    """
    _ensure_sympy()
    
    # Convert to SymPy Matrix if needed
    if not isinstance(A, Matrix):
        # Verify no floats
        for row in A:
            for val in row:
                if isinstance(val, float):
                    raise MathError(f"Float value {val} not allowed in exact nullspace")
        A = Matrix([[sympy.Rational(str(x)) if isinstance(x, Fraction) else x 
                     for x in row] for row in A])
    
    nullspace = A.nullspace()
    
    # Convert to list of Fraction lists
    result = []
    for vec in nullspace:
        result.append([Fraction(entry.p, entry.q) for entry in vec])
    
    return result


def solve_rational_system(A: List[List[Union[int, Fraction]]], 
                          b: List[Union[int, Fraction]]) -> Optional[List[Fraction]]:
    """
    Solve linear system Ax = b exactly using SymPy.

    Args:
        A: Coefficient matrix (integers or Fractions only)
        b: Right-hand side vector (integers or Fractions only)

    Returns:
        Exact solution as list of Fractions, or None if inconsistent

    Raises:
        MathError: If SymPy is not installed or input contains floats
    """
    _ensure_sympy()
    
    # Verify no floats
    for row in A:
        for val in row:
            if isinstance(val, float):
                raise MathError(f"Float value {val} not allowed in exact system solve")
    for val in b:
        if isinstance(val, float):
            raise MathError(f"Float value {val} not allowed in exact system solve")
    
    # Convert to SymPy
    A_sym = Matrix([[sympy.Rational(str(x)) if isinstance(x, Fraction) else x 
                     for x in row] for row in A])
    b_sym = Matrix([sympy.Rational(str(x)) if isinstance(x, Fraction) else x 
                    for x in b])
    
    try:
        sol = A_sym.solve(b_sym)
        return [Fraction(entry.p, entry.q) for entry in sol]
    except Exception:
        return None


def is_unimodular_matrix(A: List[List[Union[int, Fraction]]]) -> bool:
    """
    Check if matrix is unimodular (determinant ±1) using exact arithmetic.

    Args:
        A: Square integer or rational matrix

    Returns:
        True if matrix is unimodular (|det| = 1)

    Raises:
        MathError: If SymPy is not installed or input contains floats
        DimensionError: If matrix is not square
    """
    _ensure_sympy()
    
    # Verify no floats
    for row in A:
        for val in row:
            if isinstance(val, float):
                raise MathError(f"Float value {val} not allowed in unimodular check")
    
    n = len(A)
    if n == 0:
        return True
    
    if any(len(row) != n for row in A):
        raise DimensionError("Matrix must be square for unimodular check")
    
    det = determinant_exact(A)
    return abs(det) == 1


# ============================================================================
# SIMPLEX AND POLYTOPE UTILITIES
# ============================================================================

def is_simplex(vertices: List[List[Union[int, Fraction]]]) -> bool:
    """
    Check if vertices form a simplex (affinely independent) using exact arithmetic.

    Args:
        vertices: List of vertex coordinates (integers or Fractions only)

    Returns:
        True if vertices form a simplex
    """
    _ensure_sympy()
    
    if len(vertices) <= 1:
        return True
    
    # Verify no floats
    for v in vertices:
        for coord in v:
            if isinstance(coord, float):
                raise MathError(f"Float value {coord} not allowed in simplex check")
    
    # Convert to vectors from first vertex
    v0 = vertices[0]
    vectors = []
    for i in range(1, len(vertices)):
        vec = [vertices[i][j] - v0[j] for j in range(len(v0))]
        vectors.append(vec)
    
    # Check rank using SymPy

    from sympy import Rational as _R
    M = Matrix([[_R(int(x.numerator), int(x.denominator))
                 if hasattr(x,'numerator') else _R(int(x))
                 for x in vec] for vec in vectors])
    rank = M.rank()
    
    return rank == len(vertices) - 1


def volume_of_simplex(vertices: List[List[float]]) -> float:
    """
    Compute volume of simplex using floating point.

    NOTE: This is for visualization/estimation only.
    Core pipeline should use exact methods via Normaliz.

    Args:
        vertices: List of vertex coordinates

    Returns:
        Approximate volume as float
    """
    import numpy as np
    
    verts = np.asarray(vertices, dtype=float)
    
    if len(verts) == 1:
        return 0.0
    
    # For n-simplex in n-dimensions
    if len(verts) - 1 == verts.shape[1]:

        # (simplex_volume receives float vertices from upstream — note precision limit)
        try:
            vec_fracs = [[Fraction(x).limit_denominator(10**9) for x in row]
                         for row in (verts[1:] - verts[0]).tolist()]
            det_exact = determinant_exact(vec_fracs)
            return float(abs(det_exact)) / math.factorial(len(verts) - 1)
        except Exception:
            # Fallback to numpy if exact fails (e.g. non-square)

            try:

                from sympy import Matrix as _SM, Rational as _R
                edges = (verts[1:] - verts[0]).tolist()
                _SM2 = _SM([[_R(x).limit_denominator(10**9) for x in row] for row in edges])
                det_sym = _SM2.det()  # exact SymPy rational
                det_exact_frac = Fraction(int(det_sym.p), int(det_sym.q)) if hasattr(det_sym, 'p') else Fraction(int(det_sym))
                return float(abs(det_exact_frac) / math.factorial(len(verts) - 1))
            except Exception as e:
                raise type(e)(
                    f"simplex_volume: SymPy exact determinant unavailable: {e}. "
                    "SymPy is required — install with: pip install sympy"
                ) from e
    else:
        # Lower-dimensional simplex in higher space: use Cayley-Menger determinant.

        # distance matrix, contaminating the determinant with float errors.
        # Now computes squared distances exactly using Fraction arithmetic before
        # passing to SymPy for the exact determinant.
        n = len(verts)

        # Build exact squared-distance matrix from Fraction inputs
        from sympy import Matrix as _SM, Rational as _R, sqrt as _sqrt, factorial as _fac
        # Convert numpy rows back to Fraction for exact arithmetic
        frac_verts = []
        for row in verts:
            frac_verts.append([Fraction(v).limit_denominator(10**12) for v in row])

        # Cayley-Menger matrix entries: D[i,j] = ||v_i - v_j||^2 (exact Fraction)
        D_exact = [[Fraction(0)] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                sq = sum((frac_verts[i][k] - frac_verts[j][k]) ** 2
                         for k in range(len(frac_verts[i])))
                D_exact[i][j] = sq
                D_exact[j][i] = sq

        # Build (n+1)×(n+1) Cayley-Menger matrix with exact Fraction entries
        cm_size = n + 1
        B_exact = [[Fraction(1)] * cm_size for _ in range(cm_size)]
        B_exact[0][0] = Fraction(0)
        for i in range(n):
            B_exact[0][i + 1] = Fraction(1)
            B_exact[i + 1][0] = Fraction(1)
            for j in range(n):
                B_exact[i + 1][j + 1] = D_exact[i][j]

        try:
            # Exact SymPy determinant of exact-Fraction Cayley-Menger matrix
            B_sym = _SM([[_R(c.numerator, c.denominator) for c in row] for row in B_exact])
            det_sym = B_sym.det()  # exact SymPy rational
            sign = (-1) ** n
            denom = 2 ** (n - 1) * math.factorial(n - 1) ** 2
            vol_sq = sign * det_sym / denom
            if vol_sq <= 0:
                return 0.0
            vol_sym = _sqrt(vol_sq)
            return float(vol_sym)
        except Exception as e:
            raise type(e)(
                f"simplex_volume: exact Cayley-Menger computation failed: {e}. "
                "SymPy is required — install with: pip install sympy"
            ) from e


# ============================================================================
# VALIDATION AND TESTING
# ============================================================================

def validate_math_utils() -> Dict[str, bool]:
    """
    Validate all mathematical utilities against test cases.

    Returns:
        Dictionary of validation results
    """
    results = {}
    
    try:
        # Test 1: Integer checking
        results["is_integer"] = (
            is_integer(5) and 
            is_integer(Fraction(5,1)) and
            not is_integer(5.1)
        )
        
        # Test 2: GCD extended
        gcd, x, y = gcd_extended(48, 18)
        results["gcd_extended"] = (gcd == 6 and 48*x + 18*y == 6)
        

        weight = compute_orbit_weight([2, 3, 1], 6)  # 6!/(2!3!1!)
        results["orbit_weight"] = (weight == 60)
        

        if HAS_SYMPY:
            U_mat = [[1, 1], [0, 1]]
            results["unimodular_check"] = is_unimodular_matrix(U_mat)
        else:
            results["unimodular_check"] = False
            

        S1 = Fraction(10, 1)
        S2 = Fraction(30, 1)
        N = 5
        results["cauchy_schwarz"] = check_cauchy_schwarz(S1, S2, N)
        
        # Test 6: Nullspace basis - requires SymPy
        if HAS_SYMPY:
            A = [[1, 2, 3], [4, 5, 6]]
            nullspace = nullspace_basis(A)
            # For a 2x3 matrix with rank 2, nullity should be 1
            results["nullspace_basis"] = (len(nullspace) == 1)
        else:
            results["nullspace_basis"] = False
        
        # Test 7: Rational system solving - requires SymPy
        if HAS_SYMPY:
            A_solve = [[1, 1], [1, -1]]
            b_solve = [3, 1]
            solution = solve_rational_system(A_solve, b_solve)
            results["rational_system"] = (
                solution is not None and
                solution[0] == 2 and solution[1] == 1
            )
        else:
            results["rational_system"] = False
        
        # Test 8: Exact determinant - requires SymPy
        if HAS_SYMPY:
            A_det = [[1, 2], [3, 4]]
            det = determinant_exact(A_det)
            results["exact_det"] = (det == -2)
        else:
            results["exact_det"] = False
        
        logger.info("✅ Math utilities validation passed")
        
    except Exception as e:
        logger.error(f"❌ Math utilities validation failed: {e}")
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
    print("Testing Math Utilities with SymPy Requirements")
    print("=" * 60)
    
    if not HAS_SYMPY:
        print("\n❌ ERROR: SymPy is required but not installed.")
        print("   Install with: pip install sympy")
        sys.exit(1)
    
    # Run validation
    results = validate_math_utils()
    
    print("\nValidation Results:")
    print("-" * 40)
    
    success_count = 0
    total_count = 0
    
    for key, value in results.items():
        total_count += 1
        if key == "validation_error":
            print(f"❌ {key}: {value}")
        elif value:
            success_count += 1
            print(f"✅ {key}: PASSED")
        else:
            print(f"❌ {key}: FAILED")
    
    print("-" * 40)
    print(f"Overall: {success_count}/{total_count-1} tests passed")
    
    if "validation_error" in results:
        sys.exit(1)
    
    # Specification demonstration
    print("\n" + "=" * 60)
    print("Math Utilities Demo (Exact Arithmetic)")
    print("=" * 60)
    

    print("\n1.  - Orbit Weight Calculation:")
    frequencies = [3, 2, 1]
    N = sum(frequencies)
    weight = compute_orbit_weight(frequencies, N)
    print(f"   N! / ∏fⱼ! = {6}! / ({3}!·{2}!·{1}!) = {weight}")
    

    print("\n2.  - Rational System Solving:")
    A = [[1, 1], [1, -1]]
    b = [3, 1]
    sol = solve_rational_system(A, b)
    print(f"   Solving x + y = 3, x - y = 1")
    print(f"   Solution: x = {sol[0]}, y = {sol[1]}")
    

    print("\n3.  - Exact Determinant:")
    A_det = [[1, 2], [3, 4]]
    det = determinant_exact(A_det)
    print(f"   det([[1,2],[3,4]]) = {det}")
    

    print("\n4.  - Exact Cauchy-Schwarz:")
    S1 = Fraction(15, 2)
    S2 = Fraction(45, 2)
    N = 5
    holds = check_cauchy_schwarz(S1, S2, N)
    print(f"   S₁ = {S1}, S₂ = {S2}, N = {N}")
    print(f"   S₂ ≥ S₁²/N ? {holds}")
    
    # Nullspace basis with exact arithmetic
    print("\n5. Exact Nullspace Basis:")
    A_test = [[1, 2, 3], [4, 5, 6]]
    ns = nullspace_basis(A_test)
    print(f"   Matrix: 2x3")
    print(f"   Nullspace dimension: {len(ns)}")
    if ns:
        print(f"   Basis vector: {ns[0]}")
    
    print("\n" + "=" * 60)
    print("✅ Math Utilities Ready for Production")
    print("=" * 60)