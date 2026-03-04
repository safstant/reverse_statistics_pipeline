"""
olll - LLL (Lenstra-Lenstra-Lovász) lattice basis reduction.

API-compatible implementation of the olll package (MIT License, orisano).
Matches the public interface exactly: Vector class + reduction() function.

Reference: Lenstra, A.K., Lenstra, H.W., Lovász, L. (1982).
"Factoring polynomials with rational coefficients."
Mathematische Annalen. 261 (4): 515-534.
"""

from fractions import Fraction
from typing import List, Sequence

__all__ = ['Vector', 'reduction']


class Vector(list):
    """
    Rational vector for LLL basis reduction.
    Stores elements as fractions.Fraction internally.
    """

    def __init__(self, x):
        super().__init__(Fraction(v) for v in x)

    def __repr__(self) -> str:
        return '[' + ', '.join(str(v) for v in self) + ']'

    def __mul__(self, rhs: Fraction) -> 'Vector':
        """
        >>> Vector(["3/2", "4/5", "1/4"]) * 2
        [3, 8/5, 1/2]
        """
        rhs = Fraction(rhs)
        return Vector([v * rhs for v in self])

    def __rmul__(self, lhs: Fraction) -> 'Vector':
        """
        >>> 2 * Vector(["3/2", "4/5", "1/4"])
        [3, 8/5, 1/2]
        """
        return self.__mul__(lhs)

    def __sub__(self, rhs: 'Vector') -> 'Vector':
        """
        >>> Vector([1, 2, 3]) - [6, 5, 4]
        [-5, -3, -1]
        """
        return Vector([a - Fraction(b) for a, b in zip(self, rhs)])

    def dot(self, rhs: 'Vector') -> Fraction:
        """
        >>> Vector([1, 2, 3]).dot([4, 5, 6])
        Fraction(32, 1)
        """
        return sum(Fraction(a) * Fraction(b) for a, b in zip(self, rhs))

    def sdot(self) -> Fraction:
        return self.dot(self)

    def proj(self, rhs: 'Vector') -> 'Vector':
        """
        Projection of rhs onto self.
        >>> Vector([1, 1, 1]).proj([-1, 0, 2])
        [1/3, 1/3, 1/3]
        """
        c = self.proj_coff(rhs)
        return self * c

    def proj_coff(self, rhs: 'Vector') -> Fraction:
        """
        Gram-Schmidt projection coefficient <rhs, self> / <self, self>.
        >>> Vector([1, 1, 1]).proj_coff([-1, 0, 2])
        Fraction(1, 3)
        """
        denom = self.sdot()
        if denom == 0:
            return Fraction(0)
        return Fraction(Vector(rhs).dot(self), denom)


def reduction(basis: List[List], delta: float = 0.75) -> List[Vector]:
    """
    LLL basis reduction.

    Args:
        basis: List of row vectors (list of lists of int or Fraction).
        delta: Lovász condition parameter. Must satisfy 0.25 < delta < 1.
               Typical value: 0.75.

    Returns:
        LLL-reduced basis as list of Vector objects.
    """
    delta = Fraction(delta)
    n = len(basis)
    if n == 0:
        return []

    # Work with Vector objects
    B = [Vector(row) for row in basis]

    # Gram-Schmidt orthogonalization (not normalised)
    # B_star[i] = B[i] - sum_{j<i} mu[i][j] * B_star[j]
    # mu[i][j] = <B[i], B_star[j]> / <B_star[j], B_star[j]>

    def gram_schmidt(vecs):
        gs = []
        mu = [[Fraction(0)] * len(vecs) for _ in range(len(vecs))]
        for i, v in enumerate(vecs):
            u = Vector(v)
            for j, uj in enumerate(gs):
                c = uj.proj_coff(v)
                mu[i][j] = c
                u = u - uj * c
            gs.append(u)
        return gs, mu

    k = 1
    while k < n:
        # Recompute Gram-Schmidt at each step
        gs, mu = gram_schmidt(B)

        # Step 1: Size reduction — make |mu[k][j]| <= 1/2 for j < k
        for j in range(k - 1, -1, -1):
            m = mu[k][j]
            r = Fraction(round(float(m)))
            if r != 0:
                B[k] = B[k] - B[j] * r
                # Recompute after modification
                gs, mu = gram_schmidt(B)

        # Step 2: Lovász condition
        # delta * ||B*[k-1]||^2 <= ||B*[k]||^2 + mu[k][k-1]^2 * ||B*[k-1]||^2
        gs_k = gs[k]
        gs_km1 = gs[k - 1]
        lhs = delta * gs_km1.sdot()
        rhs = gs_k.sdot() + mu[k][k - 1] ** 2 * gs_km1.sdot()

        if lhs <= rhs:
            k += 1
        else:
            # Swap B[k] and B[k-1]
            B[k], B[k - 1] = B[k - 1], B[k]
            k = max(k - 1, 1)

    return B
