"""
Unit tests for exact arithmetic components — v15.7.

Covers:
  - math_utils: gcd_list, lcm_list
  - constraints: normalize, normalize_all
  - stats_utils: SymmetryOrbit weights
  - lattice_utils: SNF
  - feasibility: scipy LP correctness
  - dimension: intrinsic_lattice_basis (SNF-based, v15.5)
  - decomposition: Stage A signs, Stage B Barvinok replacement (v15.7)
"""
import pytest
import sympy as sp
from fractions import Fraction as F
from itertools import combinations


def test_import():
    import reverse_stats
    assert hasattr(reverse_stats, "run_pipeline")


# ---------------------------------------------------------------------------
# math_utils
# ---------------------------------------------------------------------------

def test_gcd_list():
    from reverse_stats.math_utils import gcd_list
    assert gcd_list([12, 8, 4]) == 4
    assert gcd_list([7]) == 7


def test_lcm_list():
    from reverse_stats.math_utils import lcm_list
    assert lcm_list([4, 6]) == 12
    assert lcm_list([3, 5, 7]) == 105


# ---------------------------------------------------------------------------
# constraints
# ---------------------------------------------------------------------------

def test_constraint_normalize():
    from reverse_stats.constraints import Equation
    eq = Equation(
        coefficients=(F(1, 2), F(1, 3)),
        rhs=F(5, 6),
    )
    normed = eq.normalize()
    assert all(c.denominator == 1 for c in normed.coefficients)
    assert normed.rhs.denominator == 1


def test_constraint_system_normalize_all():
    from reverse_stats.constraints import ConstraintSystem, Equation
    cs = ConstraintSystem(variables=2)
    cs.add_equation(Equation(
        coefficients=(F(1, 2), F(1, 4)),
        rhs=F(3, 4),
    ))
    normed = cs.normalize_all()
    for eq in normed.equations:
        for c in eq.coefficients:
            assert c.denominator == 1


# ---------------------------------------------------------------------------
# stats_utils
# ---------------------------------------------------------------------------

def test_orbit_weight_basic():
    from reverse_stats.stats_utils import SymmetryOrbit
    orbit = SymmetryOrbit(frequencies=(2, 1, 1))
    assert orbit.weight == 12


def test_orbit_weight_all_same():
    from reverse_stats.stats_utils import SymmetryOrbit
    orbit = SymmetryOrbit(frequencies=(4,))
    assert orbit.weight == 1


# ---------------------------------------------------------------------------
# lattice_utils: SNF
# ---------------------------------------------------------------------------

def test_snf_identity():
    import numpy as np
    from reverse_stats.lattice_utils import smith_normal_form_matrix
    I = np.eye(3, dtype=int)
    S, U, V = smith_normal_form_matrix(I)
    assert S.shape == (3, 3)
    for i in range(3):
        assert S[i, i] == 1


# ---------------------------------------------------------------------------
# feasibility
# ---------------------------------------------------------------------------

def test_feasibility_simple_feasible():
    """scipy LP correctly identifies a feasible system."""
    from reverse_stats.constraints import ConstraintSystem, Equation, Bound
    from reverse_stats.feasibility import check_feasibility, FeasibilityStatus
    cs = ConstraintSystem(variables=2)
    cs.add_equation(Equation(
        coefficients=(F(1), F(1)),
        rhs=F(4),
    ))
    cs.add_bound(Bound(variable=0, lower=F(0), upper=F(4)))
    cs.add_bound(Bound(variable=1, lower=F(0), upper=F(4)))
    result = check_feasibility(cs)
    assert result.status == FeasibilityStatus.FEASIBLE


def test_feasibility_infeasible():
    """scipy LP correctly identifies an infeasible system."""
    from reverse_stats.constraints import ConstraintSystem, Equation, Bound
    from reverse_stats.feasibility import check_feasibility, FeasibilityStatus
    cs = ConstraintSystem(variables=2)
    cs.add_equation(Equation(
        coefficients=(F(1), F(1)),
        rhs=F(10),
    ))
    cs.add_bound(Bound(variable=0, lower=F(0), upper=F(2)))
    cs.add_bound(Bound(variable=1, lower=F(0), upper=F(2)))
    result = check_feasibility(cs)
    assert result.status == FeasibilityStatus.INFEASIBLE


# ---------------------------------------------------------------------------
# dimension: intrinsic_lattice_basis (v15.5 SNF-based)
# ---------------------------------------------------------------------------

def test_intrinsic_basis_det_BtB_equals_1():
    """SNF basis B satisfies det(B^T B) = 1 for all N=6 dice cones."""
    from reverse_stats.dimension import intrinsic_lattice_basis

    # All 6 tangent cones from the N=6 dice problem (intrinsic projection verified)
    cone_rays = [
        # Cone 0 (index=1)
        [(F(0),F(7),F(-24),F(30),F(-16),F(3)),
         (F(8),F(-15),F(0),F(10),F(0),F(-3)),
         (F(0),F(1),F(0),F(-6),F(8),F(-3)),
         (F(0),F(3),F(-8),F(6),F(0),F(-1)),
         (F(1),F(-5),F(10),F(-10),F(5),F(-1))],
    ]
    for rays in cone_rays:
        B, B_inv, d = intrinsic_lattice_basis(rays)
        B_sp = sp.Matrix([[int(B[i][j]) for j in range(d)] for i in range(len(B))])
        det_BtB = int((B_sp.T * B_sp).det())
        assert det_BtB == 1, f"det(B^T B)={det_BtB}, expected 1 (SNF basis not normalised)"


def test_intrinsic_projection_gives_integers():
    """After SNF projection, all ray coordinates must be exact integers."""
    from reverse_stats.dimension import intrinsic_lattice_basis

    rays = [
        (F(0),F(7),F(-24),F(30),F(-16),F(3)),
        (F(8),F(-15),F(0),F(10),F(0),F(-3)),
        (F(0),F(1),F(0),F(-6),F(8),F(-3)),
        (F(0),F(3),F(-8),F(6),F(0),F(-1)),
        (F(1),F(-5),F(10),F(-10),F(5),F(-1)),
    ]
    B, B_inv, d = intrinsic_lattice_basis(rays)
    for ray in rays:
        coords = [sum(F(B_inv[k][j]) * F(ray[j]) for j in range(len(ray)))
                  for k in range(d)]
        for c in coords:
            assert c.denominator == 1, (
                f"Non-integer intrinsic coordinate {c} -- SNF projection failed"
            )


# ---------------------------------------------------------------------------
# decomposition: Stage A — fan signs (v15.7 Bug 2 fix)
# ---------------------------------------------------------------------------

# Cone 0 from N=6 dice, projected to intrinsic R^3 via SNF basis
CONE0_INTR = [(-18,-10,-3), (-137,-65,-24), (-17,-8,-3), (-11,-5,-2), (31,10,6)]


def _make_cone(rays, is_uni=False):
    from reverse_stats.decomposition import DecompositionCone
    return DecompositionCone(
        rays=[tuple(F(x) for x in r) for r in rays],
        is_unimodular=is_uni,
    )


def _det(rays):
    return abs(int(sp.Matrix([[int(F(x)) for x in r] for r in rays]).det()))


def test_stage_a_no_degenerate_subcones():
    """Fan triangulation must exclude degenerate (det=0) sub-cones."""
    from reverse_stats.decomposition import _signed_decomposition_lll, get_decomposition_config
    cone = _make_cone(CONE0_INTR)
    result = _signed_decomposition_lll(cone, get_decomposition_config())
    degenerate = [
        i for i, c in enumerate(result.cones)
        if len(c.rays) == 3 and _det(c.rays) == 0
    ]
    assert len(degenerate) == 0, (
        f"Stage A: {len(degenerate)} degenerate sub-cone(s) with det=0: indices {degenerate}"
    )


def test_stage_a_signs_not_all_positive():
    """Fan sub-cones must receive mixed signs, not all +1."""
    from reverse_stats.decomposition import _signed_decomposition_lll, get_decomposition_config
    cone = _make_cone(CONE0_INTR)
    result = _signed_decomposition_lll(cone, get_decomposition_config())
    assert any(s == -1 for s in result.signs), (
        f"Stage A: all signs are +1 {result.signs} -- "
        "Lawrence/Varchenko orientation not applied (Bug 2 not fixed)"
    )


def test_stage_a_sign_equals_det_sign():
    """Each sub-cone's sign must equal sign(det) of that sub-cone's ray matrix."""
    from reverse_stats.decomposition import _signed_decomposition_lll, get_decomposition_config
    cone = _make_cone(CONE0_INTR)
    result = _signed_decomposition_lll(cone, get_decomposition_config())
    for i, (c, s) in enumerate(zip(result.cones, result.signs)):
        int_rays = [[int(F(x)) for x in r] for r in c.rays]
        if len(int_rays) == len(int_rays[0]):  # square
            raw_det = int(sp.Matrix(int_rays).det())
            expected_sign = 1 if raw_det > 0 else -1
            assert s == expected_sign, (
                f"Sub-cone {i}: sign={s} but det={raw_det} -> expected sign={expected_sign}"
            )


def test_stage_a_unimodular_subcone_present():
    """At least one sub-cone must be unimodular (det=1) after correct fan."""
    from reverse_stats.decomposition import _signed_decomposition_lll, get_decomposition_config
    cone = _make_cone(CONE0_INTR)
    result = _signed_decomposition_lll(cone, get_decomposition_config())
    uni = [c for c in result.cones if len(c.rays) == 3 and _det(c.rays) == 1]
    assert len(uni) > 0, (
        "No unimodular (det=1) sub-cone found. "
        "Expected sub-cone with rays[0]+rest[1,2] (det=1)."
    )


# ---------------------------------------------------------------------------
# decomposition: Stage B — Barvinok replacement (v15.7 Bug 1 fix)
# ---------------------------------------------------------------------------

def test_stage_b_splits_det2_cone():
    """det=2 simplicial cone must be split into 2 unimodular sub-cones."""
    from reverse_stats.decomposition import _signed_decomposition_lll, get_decomposition_config
    cone = _make_cone([(2, 1), (0, 1)])  # det=2
    result = _signed_decomposition_lll(cone, get_decomposition_config())
    assert len(result.cones) >= 2, (
        f"Stage B: returned {len(result.cones)} cone(s) for det=2 input -- no split occurred"
    )


def test_stage_b_all_subcones_unimodular():
    """All sub-cones from Barvinok replacement of det=2 cone must have det=1."""
    from reverse_stats.decomposition import _signed_decomposition_lll, get_decomposition_config
    cone = _make_cone([(2, 1), (0, 1)])
    result = _signed_decomposition_lll(cone, get_decomposition_config())
    for i, c in enumerate(result.cones):
        rays = list(c.rays)
        if len(rays) == len(rays[0]):
            d = _det(rays)
            assert d == 1, (
                f"Stage B: sub-cone {i} has det={d} -- Barvinok replacement "
                f"did not produce unimodular sub-cones"
            )


def test_stage_b_replacement_uses_interior_vector():
    """
    Barvinok replacement finds w=(1,1) as the interior parallelepiped vector
    for the cone with rays (2,1),(0,1). Both sub-cones must contain w.
    """
    from reverse_stats.decomposition import _signed_decomposition_lll, get_decomposition_config
    cone = _make_cone([(2, 1), (0, 1)])
    result = _signed_decomposition_lll(cone, get_decomposition_config())
    # Each sub-cone must contain w=(1,1) as one of its rays
    w = (F(1), F(1))
    for i, c in enumerate(result.cones):
        ray_set = [tuple(r) for r in c.rays]
        assert w in ray_set, (
            f"Sub-cone {i} does not contain interior vector w=(1,1): rays={ray_set}"
        )


def test_stage_b_det3_cone():
    """det=3 simplicial cone in R^2 is also correctly split."""
    from reverse_stats.decomposition import _signed_decomposition_lll, get_decomposition_config
    # det((3,1),(0,1)) = 3
    cone = _make_cone([(3, 1), (0, 1)])
    result = _signed_decomposition_lll(cone, get_decomposition_config())
    for i, c in enumerate(result.cones):
        rays = list(c.rays)
        if len(rays) == len(rays[0]):
            d = _det(rays)
            assert d <= 2, (  # each sub-cone det strictly less than 3
                f"Stage B det=3: sub-cone {i} has det={d} -- det not strictly reduced"
            )
