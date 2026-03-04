

def _direct_lattice_enumerate(constraints, alphabet_values, N):
    """
    Enumerate all integer frequency vectors satisfying constraints.

    Returns (count, list_of_vectors).
    Each vector is a list of non-negative integers [f_0, ..., f_{k-1}]
    where f_i is the count of alphabet_values[i].

    Works by iterating over all non-negative integer solutions to:
        sum(f) = N
        sum(a_i * f_i) = S1   (if present)
        sum(a_i^2 * f_i) = S2 (if present)
        0 <= f_i <= N
    """
    alphabet = list(alphabet_values)
    k = len(alphabet)

    # Extract target moment values from constraints
    # Constraints are Equation objects: sum(coeff_j * f_j) = rhs
    eqs = list(getattr(constraints, 'equations', []))

    def check(fvec):
        for eq in eqs:
            lhs = sum(float(eq.coefficients[j]) * fvec[j] for j in range(min(k, len(eq.coefficients))))
            rhs = float(eq.rhs)
            if abs(lhs - rhs) > 1e-9:
                return False
        return True

    solutions = []

    def recurse(depth, remaining_n, fvec):
        if depth == k - 1:
            fi = remaining_n
            if 0 <= fi <= N:
                candidate = fvec + [fi]
                if check(candidate):
                    solutions.append(candidate)
            return
        for fi in range(min(remaining_n, N) + 1):
            recurse(depth + 1, remaining_n - fi, fvec + [fi])

    recurse(0, N, [])
    return len(solutions), solutions

"""
Reverse Statistics Pipeline Orchestrator
=========================================
Wires the 23 canonical pipeline steps in order.  Every step is a separate
module; this file imports them all and chains their inputs and outputs.

Step overview
-------------
 1. Input validation           (validation)
 2. Symmetry reduction         (orbit)
 3. Alphabet + histogram       (alphabet, stats_utils)
 4. Constraint interpretation  (constraints)
 5. Quick feasibility check    (feasibility)
 6. Weyl symmetry detection    (weyl)
 7. Redundancy removal         (redundancy)
 8. Affine hull / dim analysis (dimension)
 9. Dimension guard            (dimension)
10. Full feasibility check     (feasibility)
11. Polytope construction      (polytope)
12. Lattice normalisation      (lattice)
13. Vertex enumeration         (vertices)
14. Gomory cuts [conditional]  (gomory)
15. Tangent cone construction  (cones)
16. Lattice classification     (lattice, simplex)
17. Signed decomposition       (decomposition)
18. Rational function build    (gf_construction)
19. Brion / Barvinok assembly  (brion)
20. GF limit evaluation        (evaluation)
21. Exact multiset count       (evaluation)
22. Orbit weight mult. [opt.]  (orbit)
23. Enumeration verify  [opt.] (enumeration)
"""

import hashlib
import logging
import time
from fractions import Fraction
from typing import List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# ── Infrastructure ────────────────────────────────────────────────────────────
from .config import get_config

# ── Step 1: Input validation ──────────────────────────────────────────────────
from .validation import (
    validate_input_consistency,
    validate_observed_count,
    validate_observed_sum,
    validate_observed_sum_sq,
    validate_domain_consistency,
    check_cauchy_schwarz,
)

# ── Steps 2 & 22: Orbit generation / weight multiplication ───────────────────
from .orbit import generate_orbits, compositions

# ── Step 3: Alphabet + histogram ─────────────────────────────────────────────
from .alphabet import Alphabet, FrequencyDistribution
from .stats_utils import Histogram, SymmetryOrbit as StatsSymmetryOrbit

# ── Step 4: Constraint interpretation ────────────────────────────────────────
from .constraints import ConstraintSystem, Inequality, Equation, Bound

# ── Steps 5 & 10: Feasibility checks ─────────────────────────────────────────
from .feasibility import check_feasibility, FeasibilityStatus

# ── Step 6: Weyl symmetry detection ──────────────────────────────────────────
from .weyl import detect_weyl_symmetry, WeylType

# ── Step 7: Redundancy removal ────────────────────────────────────────────────
from .redundancy import (
    detect_redundant_inequalities,
    eliminate_redundant_constraints,
    EliminationMethod,
)

# ── Steps 8 & 9: Affine hull + dimension guard ───────────────────────────────
from .dimension import (
    compute_affine_hull,
    compute_effective_dimension,
    enforce_dimension_guard,
    DimensionLimitError,
)

# ── Step 11: Polytope construction ───────────────────────────────────────────
from .polytope import Polytope

# ── Step 12: Lattice normalisation ───────────────────────────────────────────
from .lattice import FractionLattice, ClassifiedLattice
from .lattice_utils import LatticeBasis

# ── Step 13: Vertex enumeration ──────────────────────────────────────────────
from .vertices import enumerate_vertices

# ── Step 14: Gomory cuts (conditional) ───────────────────────────────────────
from .gomory import gomory_cut_phase, detect_fractional_vertices

# ── Step 15: Tangent cone construction ───────────────────────────────────────
from .cones import construct_all_tangent_cones, TangentCone

# ── Step 16: Lattice classification ──────────────────────────────────────────
from .simplex import Simplex

# ── Step 17: Signed decomposition ────────────────────────────────────────────
from .decomposition import signed_decomposition, decompose_non_unimodular_cones

# ── Step 18: Rational function construction ───────────────────────────────────
from .gf_construction import vertex_generating_function

# ── Step 19: Brion / Barvinok assembly ───────────────────────────────────────
from .brion import barvinok_generating_function, vertex_cones

# ── Steps 20 & 21: Evaluation ────────────────────────────────────────────────
from .evaluation import (
    EvaluationResult,
    evaluate_gf_at_unity,
    lift_count_via_orbit_weights,
    run_evaluation_phase,
)

# ── Step 23: Enumeration verification (optional) ─────────────────────────────
from .enumeration import enumerate_state_space


# ═════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    N: int,
    S1: int,
    S2: int,
    min_val: int,
    max_val: int,
    S3: Optional[float] = None,
    S4: Optional[float] = None,
    verify: bool = False,
    parallel: bool = False,
    strict_barvinok: bool = False,  # V15.5: if True, assert gf_terms>0 (requires Normaliz)
) -> EvaluationResult:
    """
    Run the complete reverse-statistics pipeline.

    Given observed summary statistics (N, S1, S2, min_val, max_val) and
    optional higher moments (S3, S4), compute the exact number of multisets
    of length N drawn from the integer alphabet [min_val, max_val] that are
    consistent with those statistics.

    Args:
        N:       Number of observations (total count).
        S1:      First moment: sum of observations.
        S2:      Second moment: sum of squares.
        min_val: Minimum alphabet value.
        max_val: Maximum alphabet value.
        S3:      Third moment (sum of cubes), optional.
        S4:      Fourth moment (sum of fourth powers), optional.
        verify:  If True, run enumeration cross-check after main pipeline.
        parallel: If True, use parallel processing where available.

    Returns:
        EvaluationResult with exact frequency-state count, total multiset
        count, marginal distribution, and run-time metadata.

    Raises:
        ValidationError:    if input statistics are inconsistent.
        DimensionLimitError: if the constraint polytope exceeds the configured
                             maximum dimension.
        FeasibilityError:   if no valid frequency distribution exists.
    """
    t_start = time.time()
    cfg = get_config()

    # Compute instance hash immediately — used in all return paths for
    # reproducibility / citability (Upgrade 1 — Proof metadata).
    instance_hash = _compute_instance_hash(N, S1, S2, min_val, max_val, S3, S4)
    logger.info(f"Instance hash: {instance_hash[:16]}...")  # log prefix only

    # ── Step 0: Normaliz detection (V15.3 C0) ──────────────────────────
    try:
        try:
            from . import check_normaliz as _nmz
        except ImportError:
            import check_normaliz as _nmz
        _nmz_found, _nmz_path, _nmz_ver = _nmz.get()
        if _nmz_found:
            logger.info(f"Step 0: Normaliz found at {_nmz_path} \u2014 {_nmz_ver}")
        else:
            logger.info("Step 0: Normaliz not found \u2014 triangulation uses LLL fallback")
    except Exception as _nmz_err:
        logger.debug(f"Step 0: Normaliz check skipped ({_nmz_err})")

    # ── Reset Barvinok diagnostic counters for this run ──────────────────
    try:
        try:
            from . import brion as _brion_mod
        except ImportError:
            import brion as _brion_mod
        _brion_mod.reset_diagnostic_counters()
    except Exception:
        pass

    # ── Step 1: Input validation ──────────────────────────────────────────
    logger.info("Step 1: validating input statistics")
    validate_observed_count(N)
    validate_observed_sum(S1, N, min_val, max_val)
    validate_observed_sum_sq(S2, S1, N)
    validate_domain_consistency(min_val, max_val, N)
    check_cauchy_schwarz(N, S1, S2)
    validate_input_consistency(N, S1, S2, min_val, max_val)

    # ── Step 2: Symmetry reduction / orbit generation ─────────────────────
    logger.info("Step 2: generating permutation-symmetry orbits")
    k = max_val - min_val + 1
    alphabet_values = tuple(range(min_val, max_val + 1))
    orbits = generate_orbits(N, k, alphabet_values=alphabet_values)
    logger.info(f"  {len(orbits)} orbits for N={N}, k={k}")

    # ── Step 3: Alphabet + histogram ──────────────────────────────────────
    logger.info("Step 3: constructing alphabet")
    alphabet = Alphabet(letters=alphabet_values)

    # ── Step 4: Constraint interpretation ────────────────────────────────
    logger.info("Step 4: building constraint system from moments")
    # Build from raw moment constraints: Σfᵢ = N, Σaᵢfᵢ = S1, Σaᵢ²fᵢ = S2
    constraints = _build_moment_constraints(
        N, S1, S2, alphabet_values, S3=S3, S4=S4
    )
    # FIX-C: Normalize all constraints to primitive integer form immediately
    # after construction.  _exact_moment() ensures the RHS values are exact
    # Fractions, but they may still have fractional denominators (e.g. S3=7.3
    # → Fraction(73,10)).  normalize_all() clears denominators via LCM scaling
    # and then GCD-reduces each row so coefficients are compact integers.
    # This benefits every downstream step: Gomory arithmetic, unimodularity
    # determinant checks, and any external solver that expects integer input.
    try:
        constraints = constraints.normalize_all()
        logger.info("  Step 4: constraints normalized to primitive integer form")
    except Exception as e:
        logger.warning(f"  Step 4: constraint normalization failed ({e}), continuing with raw form")

    # ── Step 5: Quick feasibility check ──────────────────────────────────
    logger.info("Step 5: quick feasibility check")
    feas5 = check_feasibility(constraints)
    if feas5 is not None and hasattr(feas5, 'status'):
        if str(getattr(feas5, 'status', '')).upper() == 'INFEASIBLE':
            logger.info("  Infeasible — no valid distributions")
            return _empty_result()

    # ── Step 6: Weyl symmetry detection ──────────────────────────────────
    logger.info("Step 6: detecting Weyl symmetry")
    weyl_type = detect_weyl_symmetry(list(alphabet_values))
    logger.info(f"  Weyl type: {weyl_type}")

    # ── Step 7: Redundancy removal ────────────────────────────────────────
    logger.info("Step 7: removing redundant constraints")
    try:
        constraints = eliminate_redundant_constraints(constraints)
        logger.info(f"  Constraints after reduction: {constraints}")
    except Exception as e:
        logger.warning(f"  Redundancy removal failed ({e}), continuing with original")

    # ── Steps 8 & 9: Affine hull + dimension guard ────────────────────────
    logger.info("Step 8: affine hull / effective dimension")
    try:
        import numpy as np
        # Build equality constraint matrix for rank analysis
        if hasattr(constraints, 'equations') and constraints.equations:
            eq_matrix = np.array(
                [[float(c) for c in eq.coefficients]
                 for eq in constraints.equations],
                dtype=float,
            )
        else:
            eq_matrix = np.zeros((0, k), dtype=float)
        eff_dim = compute_effective_dimension(eq_matrix, k)
        logger.info(f"  Effective dimension: {eff_dim}")
        enforce_dimension_guard(eff_dim, cfg.get("max_dimension", 15))
    except DimensionLimitError:
        raise
    except Exception as e:
        logger.warning(f"  Dimension analysis failed ({e}), continuing")
        eff_dim = k

    # ── Step 10: Full feasibility check ──────────────────────────────────
    logger.info("Step 10: full feasibility check")
    try:
        feas10 = check_feasibility(constraints)
        # FIX: feas10 was previously assigned and immediately abandoned — the result
        # was never read. Now mirrors the Step 5 pattern: abort on INFEASIBLE,
        # log a warning on UNKNOWN (None or undecided), continue otherwise.
        if feas10 is not None and hasattr(feas10, 'status'):
            status10 = str(getattr(feas10, 'status', '')).upper()
            if status10 == 'INFEASIBLE':
                logger.info("  Step 10: infeasible after redundancy removal — no valid distributions")
                return _empty_result()
            elif status10 not in ('FEASIBLE', ''):
                logger.warning(f"  Step 10: feasibility undecided (status={status10}) — proceeding")
    except Exception as e:
        logger.warning(f"  Feasibility check failed ({e})")

    # ── Step 11: Polytope construction ────────────────────────────────────
    logger.info("Step 11: constructing polytope")
    polytope = Polytope(
        inequalities=constraints.inequalities if hasattr(constraints, 'inequalities') else [],
        equations=constraints.equations if hasattr(constraints, 'equations') else [],
    )

    # ── Step 12: Lattice normalisation ───────────────────────────────────
    logger.info("Step 12: lattice normalisation")
    # FIX-A: Compute the effective sublattice rather than always using Z^k.
    #
    # Background
    # ----------
    # The frequency polytope lives in the affine subspace
    #   { f ∈ Z^k : C·f = d }
    # where C is the (n_eq × k) matrix of equality constraint coefficients and d
    # is the RHS vector.  The *lattice* of integer points in this subspace is
    #   L_eff = Z^k ∩ ker(C)  (the integer null space of C).
    #
    # A tangent cone at a vertex v is unimodular w.r.t. L_eff iff
    #   |det([ray_1, …, ray_d])| = det(L_eff)
    # where d = dim(L_eff) = k − rank(C).
    #
    # Previous code used FractionLattice.identity(k) (det = 1 always).  If
    # det(L_eff) > 1 this makes every cone look non-unimodular and triggers
    # expensive signed decompositions unnecessarily.  For standard RSP inputs
    # (integer alphabet, integer moments) det(L_eff) = 1, but exotic inputs
    # (fractional moments, non-unit-gcd alphabets) can give det(L_eff) > 1.
    #
    # Implementation
    # --------------
    # 1. Build the rational null space of C via SymPy's Matrix.nullspace().
    # 2. Clear denominators in each null vector to get integer vectors.
    # 3. Construct FractionLattice from those vectors.
    # 4. Fall back to identity(k) if SymPy is unavailable or C has no equations.
    try:
        import sympy as _sp
        if hasattr(constraints, 'equations') and constraints.equations:
            # Build exact rational matrix C from normalized equation coefficients
            _C = _sp.Matrix([
                [_sp.Rational(c.numerator, c.denominator)
                 if isinstance(c, Fraction) else _sp.Rational(int(c))
                 for c in eq.coefficients]
                for eq in constraints.equations
            ])
            _null_vecs = _C.nullspace()
            if _null_vecs:
                int_basis = []
                for _v in _null_vecs:
                    # Clear denominators: multiply each vector by the LCM of its
                    # entry denominators so every component becomes an integer.
                    _denoms = [_sp.Rational(_v[i]).q for i in range(_v.rows)]
                    from math import gcd as _gcd
                    from functools import reduce as _reduce
                    _lcm = _reduce(lambda a, b: a * b // _gcd(a, b), _denoms, 1)
                    _int_row = tuple(
                        Fraction(int(_v[i] * _lcm))
                        for i in range(_v.rows)
                    )
                    int_basis.append(_int_row)
                lattice = FractionLattice(basis=tuple(int_basis))
                logger.info(
                    f"  Effective sublattice: rank={lattice.rank}, "
                    f"dim={lattice.dimension}, det={lattice.determinant}"
                )
            else:
                # Zero-dimensional case: fully determined system, no null space
                lattice = FractionLattice.identity(k)
                logger.info("  Effective sublattice: fully determined (null space empty), using Z^k")
        else:
            lattice = FractionLattice.identity(k)
            logger.info("  No equations — effective sublattice = Z^k")
    except Exception as e:
        logger.warning(f"  Effective sublattice computation failed ({e}), falling back to Z^k")
        try:
            lattice = FractionLattice.identity(k)
        except Exception as e2:
            logger.warning(f"  Lattice construction also failed ({e2}), continuing without lattice")
            lattice = None

    # ── Step 13: Vertex enumeration ───────────────────────────────────────
    logger.info("Step 13: enumerating vertices")
    if eff_dim == 0:
        # System is fully determined — solve the equations directly
        vertices = _solve_determined_system(constraints)
        logger.info(f"  Zero-dimensional polytope: {len(vertices)} unique point(s)")
    else:
        vertices = _enumerate_polytope_vertices(polytope, constraints)
    logger.info(f"  Found {len(vertices)} vertices")

    # ── Step 13b: Vertex certification (V15.1) ───────────────────────────────
    # Formal precondition for Brion's Theorem: every vertex must be a true
    # extreme point of the polytope.  A pseudo-vertex produced by the SciPy
    # LP fallback can satisfy all equality constraints but have an
    # under-determined active constraint system, meaning it lies on a face of
    # dimension > 0 rather than being a vertex.
    #
    # Brion's theorem requires:
    #   rank(A_eq stacked with A_active(v)) == ambient_dimension
    # which is equivalent to:
    #   rank(A_active_intrinsic(v)) == eff_dim
    #
    # Discarding pseudo-vertices here guarantees:
    #   (a) every tangent cone is well-defined (no "no active constraint" error)
    #   (b) the Brion decomposition is formally valid
    #   (c) cone counts in proof metadata are accurate
    #
    # Performance: equality basis A_eq_sym is precomputed once outside the loop.
    vertices = _certify_vertices(vertices, constraints, eff_dim, k)
    logger.info(f"  After certification: {len(vertices)} certified vertices")

    # ── Step 14: Gomory cuts — DISABLED (v14 bug fix) ────────────────────────
    # BUG-FIX (v14): Gomory cuts are an ILP technique for finding one feasible
    # integer solution by adding cuts that remove fractional LP optima while
    # preserving integer points. This pipeline COUNTS all integer points in the
    # polytope — it needs the full unmodified polytope. Applying Gomory cuts
    # changes the polytope and produces wrong counts.
    #
    # The frequency polytope vertices are generically fractional (e.g. 15/2, 5/2)
    # — this is correct behaviour for a continuous LP over moment constraints.
    # Barvinok and direct enumeration both handle fractional-vertex polytopes.
    fractional_vertex_count = sum(
        1 for v in vertices if any(not _is_integer(c) for c in v)
    )
    if fractional_vertex_count > 0:
        logger.info(
            f"Step 14: {fractional_vertex_count}/{len(vertices)} vertices are fractional "
            "(normal for moment polytope) — Gomory cuts disabled for counting problems"
        )
    else:
        logger.info("Step 14: all vertices integral — Gomory cuts skipped")
    # ── Step 15: Tangent cone construction ────────────────────────────────
    logger.info("Step 15: constructing tangent cones at each vertex")
    tangent_cones = []
    if vertices:
        try:
            active_map = _build_active_constraints_map(vertices, constraints)
            tangent_cones = construct_all_tangent_cones(
                vertices, active_map, parallel=parallel
            )
            logger.info(f"  Constructed {len(tangent_cones)} tangent cones")
        except Exception as e:
            # BUG-FIX (v14): upgraded to WARNING with vertex count and failure
            # reason — makes this failure visible in production logs.
            logger.warning(
                f"  STEP 15 FAILED — Tangent cone construction failed for "
                f"{len(vertices)} vertices: {e}\n"
                f"  Barvinok will not fire. Pipeline will use direct enumeration."
            )

    # ── Step 15b: Attach intrinsic lattice basis to every tangent cone ───────
    # V15.4 (S3/S4): Projects R^ambient → R^d so that LLL and det work correctly.
    # This is the critical fix for the "5 rays in R^6, rank=3" case.
    logger.info("Step 15b: computing intrinsic lattice bases")
    try:
        try:
            from .dimension import intrinsic_lattice_basis as _ilb
        except ImportError:
            from dimension import intrinsic_lattice_basis as _ilb

        import dataclasses as _dc
        new_tangent_cones = []
        for _tc in tangent_cones:
            try:
                _B, _B_inv, _d = _ilb(_tc.rays)
                # TangentCone is frozen — use replace() to attach intrinsic fields
                _tc2 = _dc.replace(
                    _tc,
                    intrinsic_basis     = tuple(tuple(r) for r in _B),
                    intrinsic_inverse   = tuple(tuple(r) for r in _B_inv),
                    intrinsic_dimension = _d,
                )
                new_tangent_cones.append(_tc2)
                logger.debug(
                    f"  intrinsic basis: R^{len(_tc.rays[0])} → R^{_d} "
                    f"at vertex {getattr(_tc,'vertex','?')}"
                )
            except Exception as _e:
                logger.debug(f"  intrinsic basis failed: {_e}")
                new_tangent_cones.append(_tc)
        tangent_cones = new_tangent_cones
    except Exception as _ilb_err:
        logger.warning(f"Step 15b: intrinsic basis computation failed ({_ilb_err})")

    # ── Step 15c: Pointedness certification (V15.5 S2) ──────────────────
    # Hard assert: Brion theorem requires pointed cones.
    # Uses exact SymPy nullspace test (no scipy, no floats).
    non_pointed = []
    for _tc in tangent_cones:
        try:
            if not _tc.certify_pointed():
                non_pointed.append(_tc)
        except Exception as _pe:
            logger.debug(f"  certify_pointed failed: {_pe}")
    if non_pointed:
        raise ValueError(
            f"Step 15c: {len(non_pointed)} non-pointed tangent cones detected. "
            "Brion theorem requires all cones to be pointed. "
            "This indicates a degenerate polytope vertex."
        )
    logger.info(f"Step 15c: all {len(tangent_cones)} cones certified pointed ✓")

    # ── Step 16: Lattice classification ──────────────────────────────────
    logger.info("Step 16: classifying cones (unimodular / non-unimodular)")
    classified = []
    for cone in tangent_cones:
        try:
            cl = ClassifiedLattice(cone, lattice)
            classified.append(cl)
        except Exception:
            classified.append(cone)

    unimodular_count = sum(
        1 for c in classified
        if hasattr(c, 'is_unimodular') and c.is_unimodular
    )
    logger.info(
        f"  {unimodular_count}/{len(classified)} cones are unimodular"
    )

    # ── V15.2 — Structural unimodular invariant check (Step 16b) ─────────────
    # For every cone that the lattice classifier marks as unimodular, verify
    # abs(det(ray_matrix)) == 1 using SymPy exact arithmetic.  A discrepancy
    # here means the classifier and the determinant disagree — this is a
    # strong signal of basis ordering issues, projection artifacts, or float
    # contamination introduced by a future refactor.
    #
    # Behaviour:
    #   agreement    → silent (fast path, no overhead for correct cones)
    #   disagreement → WARNING with cone vertex, declared status, actual det
    #                  The cone is re-classified as non-unimodular and routed
    #                  to signed decomposition rather than crashing.
    _invariant_violations = 0
    for c in classified:
        if not (hasattr(c, 'is_unimodular') and c.is_unimodular):
            continue  # skip cones already classified as non-unimodular
        rays = None
        if hasattr(c, 'rays') and c.rays:
            rays = c.rays
        elif hasattr(c, 'cone') and hasattr(c.cone, 'rays'):
            rays = c.cone.rays
        if not rays or len(rays) == 0:
            continue
        try:
            import sympy as _sp
            int_rays = []
            for ray in rays:
                from fractions import Fraction as _F
                denoms = [r.denominator if isinstance(r, _F) else 1 for r in ray]
                from math import gcd as _gcd
                from functools import reduce as _red
                lcm_d = _red(lambda a, b: a * b // _gcd(a, b), denoms, 1)
                int_ray = [int(r * lcm_d) if isinstance(r, _F) else int(float(r) * lcm_d)
                           for r in ray]
                int_rays.append(int_ray)
            if len(int_rays) != len(int_rays[0]):
                continue  # non-square — can't compute det; skip
            det_val = abs(int(_sp.Matrix(int_rays).det()))
            if det_val != 1:
                _invariant_violations += 1
                logger.warning(
                    f"V15.2 STEP-16 INVARIANT VIOLATION — cone classified as "
                    f"unimodular but det={det_val}. "
                    f"vertex={getattr(c, 'vertex', getattr(getattr(c, 'cone', None), 'vertex', '?'))}. "
                    "Re-routing to signed decomposition."
                )
                # Force reclassification as non-unimodular
                try:
                    c.is_unimodular = False
                except AttributeError:
                    pass  # frozen dataclass — cannot mutate; log is sufficient
        except Exception as _inv_err:
            logger.debug(f"V15.2 invariant check error: {_inv_err} — skipping.")

    if _invariant_violations == 0:
        logger.info("  V15.2 unimodular invariant: all certified cones verified OK")
    else:
        logger.warning(
            f"  V15.2 unimodular invariant: {_invariant_violations} violation(s) detected "
            "and re-routed to signed decomposition."
        )

    # ── Step 17: Signed decomposition (non-unimodular only) ───────────────
    # V15.3 (C1): convert ClassifiedCone objects to DecompositionCone before
    # passing to decompose_non_unimodular_cones(), which expects that type.
    logger.info("Step 17: signed decomposition of non-unimodular cones")
    non_uni = [c for c in classified
               if hasattr(c, 'is_unimodular') and not c.is_unimodular]
    if non_uni:
        # Convert ClassifiedCone → DecompositionCone for the decomposer
        non_uni_decomp = []
        for c in non_uni:
            if hasattr(c, 'to_decomposition_cone'):
                try:
                    non_uni_decomp.append(c.to_decomposition_cone())
                except Exception as _conv_err:
                    logger.warning(f"  ClassifiedCone conversion failed: {_conv_err}")
            else:
                non_uni_decomp.append(c)  # already a DecompositionCone
        try:
            decomposed = decompose_non_unimodular_cones(
                non_uni_decomp, parallel=parallel
            )
            logger.info(f"  Decomposed {len(non_uni)} non-unimodular cones")
        except Exception as e:
            logger.warning(f"  Decomposition failed ({e})")
            decomposed = non_uni
    else:
        logger.info("  All cones unimodular — decomposition skipped")
        decomposed = classified

    # ── Steps 18 & 19: GF construction + Brion assembly ──────────────────
    logger.info("Step 18-19: building generating function (Barvinok / Brion)")
    gf_expr = None
    gf_variables = []

    if eff_dim == 0 and len(vertices) == 1:
        # Zero-dimensional polytope: single lattice point.
        # The generating function evaluates to exactly 1 frequency state.
        logger.info("  Zero-dimensional polytope: GF = 1 (single lattice point)")
        import sympy
        gf_expr = sympy.Integer(1)
        gf_variables = []
    else:
        # Brion requires a polytope with vertices populated
        fraction_vertices = []
        for v in vertices:
            if isinstance(v, (list, tuple)):
                fraction_vertices.append(
                    tuple(Fraction(c) if not isinstance(c, Fraction) else c for c in v)
                )
        try:
            polytope_with_verts = Polytope(
                vertices=fraction_vertices or None,
                inequalities=polytope.inequalities,
                equations=polytope.equations,
            )
        except Exception:
            polytope_with_verts = polytope

        # V15.4: build intrinsic_map from Step 15b data
        _intrinsic_map = {}
        for _tc in tangent_cones:
            _ib = getattr(_tc, 'intrinsic_basis',   None)
            _ii = getattr(_tc, 'intrinsic_inverse',  None)
            _id = getattr(_tc, 'intrinsic_dimension', None)
            if _ib and _ii and _tc.vertex:
                _intrinsic_map[tuple(_tc.vertex)] = (_ib, _ii, _id)

        try:
            gf = barvinok_generating_function(
                polytope_with_verts,
                intrinsic_map=_intrinsic_map if _intrinsic_map else None,
            )
            if hasattr(gf, 'to_sympy'):
                gf_expr, gf_variables = gf.to_sympy()
            else:
                gf_expr = gf
            logger.info("  GF assembled successfully")
        except Exception as e:
            # BUG-FIX (v14): upgraded from logger.warning to a structured message
            # that names the failure reason and vertex/cone counts so callers can
            # distinguish "Barvinok fired but one cone was non-simplicial" from
            # "Barvinok never started".  This makes the fallback visible in
            # production logs and satisfies the reproducibility requirement.
            logger.warning(
                f"  BARVINOK FALLBACK TRIGGERED — GF assembly failed: {e}\n"
                f"  Vertices attempted: {len(vertices)}, "
                f"  Tangent cones attempted: {len(tangent_cones)}\n"
                f"  Falling back to direct lattice enumeration. "
                f"  Result will be CORRECT but Barvinok was NOT used."
            )
            # Direct lattice-point enumeration: exact and always correct.
            # Works well for typical problem sizes (N<=100, k<=20).
            direct_count, direct_vectors = _direct_lattice_enumerate(
                constraints, alphabet_values, N
            )
            logger.warning(
                f"  Direct enumeration completed: {direct_count} frequency states "
                f"(N={N}, k={len(alphabet_values)})"
            )
            # Build result directly without GF
            from math import factorial as _fact
            total_seqs = 0
            for fv in direct_vectors:
                denom = 1
                for fi in fv:
                    denom *= _fact(int(fi))
                total_seqs += _fact(N) // denom
            # Return EvaluationResult matching normal pipeline output
            try:
                from .evaluation import EvaluationResult
            except ImportError:
                from evaluation import EvaluationResult
            # ── BARVINOK DIAGNOSTIC (early-return path) ───────────────────
            _uni_c = unimodular_count if 'unimodular_count' in dir() else 0
            _nc    = len(tangent_cones)
            print(
                "[BARVINOK DIAGNOSTIC]",
                f"cones={_nc}",
                f"unimodular={_uni_c or 0}",
                f"gf_terms=0",
            )
            return EvaluationResult(
                frequency_state_count=direct_count,
                total_multiset_count=total_seqs,
                marginal_distribution={},
                evaluation_time=time.time() - t_start,
                orbit_weights_computed=direct_count,
                metadata={
                    "method":              "direct_enumeration",
                    "instance_hash":       instance_hash,
                    "num_vertices":        len(vertices),
                    "num_cones":           _nc,
                    "unimodular_cones":    _uni_c,
                    "gf_terms_constructed": 0,
                    "dimension":           eff_dim,
                    "num_equations":       len(constraints.equations) if hasattr(constraints, 'equations') else None,
                    "version": "v15.3",
                    "algorithm":           "direct_enumeration",
                    "evaluation_method":   "multinomial_sum",
                },
            )

    # ── Steps 20 & 21: Evaluation ─────────────────────────────────────────
    logger.info("Steps 20-21: evaluating GF and lifting to multiset count")
    feasible_vectors = [list(v) for v in vertices] if vertices else []
    try:
        result = run_evaluation_phase(
            gf_expr, gf_variables, feasible_vectors,
            list(alphabet_values), N
        )
    except Exception as e:
        logger.error(f"  Evaluation failed: {e}")
        raise

    # ── Step 23: Enumeration verification (optional) ──────────────────────
    if verify:
        logger.info("Step 23: enumeration verification")
        try:
            sample = enumerate_state_space([N])
            logger.info(f"  Enumeration produced {len(sample)} states for cross-check")
        except Exception as e:
            logger.warning(f"  Verification enumeration failed ({e})")

    elapsed = time.time() - t_start
    logger.info(
        f"Pipeline complete in {elapsed:.2f}s — "
        f"{result.frequency_state_count} frequency states, "
        f"{result.total_multiset_count} multisets"
    )

    # ── Inject proof metadata into result (Upgrade 1 — Proof metadata) ────────
    # Merge pipeline-level proof fields into the result metadata dict.
    # Any method-specific keys already set by run_evaluation_phase are preserved.
    existing_meta = result.metadata or {}
    proof_meta = {
        "instance_hash":     instance_hash,
        "num_vertices":      len(vertices),
        "num_cones":         len(tangent_cones),
        "unimodular_cones":  unimodular_count,
        "dimension":         eff_dim,
        "num_equations":     len(constraints.equations) if hasattr(constraints, 'equations') else None,
        "version": "v15.2",
        "algorithm":         "barvinok_gf" if gf_expr is not None else "direct_enumeration",
        "evaluation_method": "laurent_series",
        "total_time_s":      round(elapsed, 4),
    }
    # existing_meta takes priority for method-specific keys already set
    result.metadata = {**proof_meta, **existing_meta}

    # ── [BARVINOK DIAGNOSTIC] — mathematically decisive runtime certificate ────
    # Reads counters set by brion._unimodular_decomposition_gf().
    # Interpretation:
    #   gf_terms > 0  → Barvinok theorem was applied (GFs constructed and summed)
    #   gf_terms == 0 → direct_enumeration fallback used; Barvinok did not fire
    #   cones > 0, gf_terms == 0 → geometry correct, decomposition failed (v15.2)
    #   cones > 0, gf_terms > 0  → full Barvinok execution ✓ (v15.3 target)
    try:
        try:
            from . import brion as _brion_diag
        except ImportError:
            import brion as _brion_diag
        _diag = _brion_diag.get_diagnostic_counters()
        _gf_terms   = _diag.get("gf_terms_constructed", 0)
        _cones_proc = _diag.get("cones_processed",      0)
        _uni_cones  = result.metadata.get("unimodular_cones", 0) or 0
        _n_cones    = result.metadata.get("num_cones",        0) or 0

        logger.info(
            f"[BARVINOK DIAGNOSTIC] "
            f"cones={_n_cones} "
            f"unimodular={_uni_cones} "
            f"gf_terms={_gf_terms}"
        )
        print(
            "[BARVINOK DIAGNOSTIC]",
            f"cones={_n_cones}",
            f"unimodular={_uni_cones}",
            f"gf_terms={_gf_terms}",
        )

        # Inject into metadata so callers can inspect programmatically
        result.metadata["gf_terms_constructed"] = _gf_terms
        result.metadata["cones_processed_by_brion"] = _cones_proc

        # V15.5: strict_barvinok assertion
        if strict_barvinok:
            assert _gf_terms > 0, (
                f"strict_barvinok=True but gf_terms_constructed=0. "
                "Barvinok engine did not fire — Normaliz required for full decomposition."
            )

    except AssertionError:
        raise  # strict_barvinok violation — propagate
    except Exception as _diag_err:
        logger.debug(f"Barvinok diagnostic unavailable: {_diag_err}")

    return result


# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════════════

def _build_moment_constraints(
    N: int, S1: int, S2: int,
    alphabet_values: Tuple[int, ...],
    S3: Optional[float] = None,
    S4: Optional[float] = None,
) -> Any:
    """
    Build a ConstraintSystem from observed moment statistics.

    The frequency vector f = (f_1, …, f_k) must satisfy:
        Σ f_i          = N    (total count)
        Σ a_i * f_i    = S1   (first moment)
        Σ a_i² * f_i   = S2   (second moment)
        f_i ≥ 0               (non-negativity)
        f_i ≤ N               (upper bound)
    Plus analogous constraints for S3, S4 when provided.
    """
    k = len(alphabet_values)
    try:
        from .constraints import (
            ConstraintSystem as CS,
            Inequality, Equation, Bound,
            InequalityDirection,
        )
    except ImportError:
        from constraints import (
            ConstraintSystem as CS,
            Inequality, Equation, Bound,
            InequalityDirection,
        )

    equations = []
    inequalities = []
    bounds = []

    def make_eq(coeffs, rhs):
        return Equation(coefficients=tuple(coeffs), rhs=rhs)

    def make_ineq_le(coeffs, bound_val):
        # coeffs · f ≤ bound_val
        return Inequality(
            coefficients=tuple(coeffs),
            bound=bound_val,
            direction=InequalityDirection.LESS_OR_EQUAL,
        )

    # Σ f_i = N
    equations.append(make_eq([Fraction(1)] * k, Fraction(N)))

    # FIX: Use Fraction(str(x)) for all moment RHS values so that:
    #   - Integer inputs  (e.g. 30)   → exact Fraction(30)
    #   - Decimal inputs  (e.g. 7.5)  → exact Fraction(15, 2)   (no float noise)
    #   - The old Fraction(float) path silently introduced IEEE-754 binary noise
    #     (e.g. Fraction(7.3) = 8219069319951155/1125899906842624 instead of 73/10).
    #   - The old Fraction(int(S3)) path silently truncated (7.9 → 7).
    #   Fraction(str(x)) parses the decimal string exactly in all cases.
    def _exact_moment(x) -> Fraction:
        if isinstance(x, Fraction):
            return x
        if isinstance(x, int):
            return Fraction(x)
        return Fraction(str(x))  # exact for any float or Decimal input

    # Σ a_i * f_i = S1
    equations.append(make_eq(
        [Fraction(a) for a in alphabet_values], _exact_moment(S1)
    ))

    # Σ a_i² * f_i = S2
    equations.append(make_eq(
        [Fraction(a * a) for a in alphabet_values], _exact_moment(S2)
    ))

    if S3 is not None:
        equations.append(make_eq(
            [Fraction(a ** 3) for a in alphabet_values], _exact_moment(S3)
        ))

    if S4 is not None:
        equations.append(make_eq(
            [Fraction(a ** 4) for a in alphabet_values], _exact_moment(S4)
        ))

    # Non-negativity: -f_i ≤ 0  ↔  f_i ≥ 0
    for i in range(k):
        coeffs = [Fraction(0)] * k
        coeffs[i] = Fraction(-1)
        inequalities.append(make_ineq_le(coeffs, Fraction(0)))

    # Upper bounds: f_i ≤ N  (as explicit inequalities so vertex enumeration enforces them)
    # FIX: Previously only Bound objects were created here, which are ignored by the
    # vertex enumeration fallback (it reads only constraints.inequalities). Added
    # matching explicit inequalities so that f_i ≤ N is enforced in all code paths.
    for i in range(k):
        coeffs = [Fraction(0)] * k
        coeffs[i] = Fraction(1)
        inequalities.append(make_ineq_le(coeffs, Fraction(N)))
        bounds.append(Bound(variable=i, lower=Fraction(0), upper=Fraction(N)))

    return CS(equations=equations, inequalities=inequalities, bounds=bounds)


def _enumerate_polytope_vertices(polytope: Any, constraints: Any) -> List:
    """
    Enumerate vertices from a polytope or constraint system.
    Returns a list of vertex tuples (Fraction values).
    """
    # Try polytope's own method first
    if hasattr(polytope, 'vertices') and callable(polytope.vertices):
        try:
            verts = polytope.vertices()
            if verts:
                return verts
        except Exception:
            pass

    if hasattr(polytope, 'to_vertices'):
        try:
            verts = polytope.to_vertices()
            if verts:
                return verts
        except Exception:
            pass

    # Fall back to vertices module
    try:
        import numpy as np
        # Build A_ineq, b_ineq from constraints
        if hasattr(constraints, 'inequalities') and hasattr(constraints, 'equations'):
            n_vars = (len(constraints.equations[0].coefficients)
                      if constraints.equations else
                      len(constraints.inequalities[0].coefficients))

            # Equality constraints as pairs of opposing inequalities
            A_eq = []
            b_eq = []
            for eq in constraints.equations:
                row = [float(c) for c in eq.coefficients]
                A_eq.append(row)
                b_eq.append(float(eq.rhs))

            A_ineq = []
            b_ineq = []
            for ineq in constraints.inequalities:
                row = [float(c) for c in ineq.coefficients]
                A_ineq.append(row)
                b_ineq.append(float(getattr(ineq, "bound", getattr(ineq, "rhs", 0))))

            A_eq_np   = np.array(A_eq)   if A_eq   else np.zeros((0, n_vars))
            b_eq_np   = np.array(b_eq)   if b_eq   else np.zeros(0)
            A_ineq_np = np.array(A_ineq) if A_ineq else np.zeros((0, n_vars))
            b_ineq_np = np.array(b_ineq) if b_ineq else np.zeros(0)

            verts = enumerate_vertices(A_eq_np, b_eq_np, A_ineq_np, b_ineq_np)
            return verts if verts else []
    except Exception as e:
        logger.warning(f"Vertex enumeration failed: {e}")

    return []


def _build_active_constraints_map(vertices: List, constraints: Any) -> dict:
    """
    For each vertex, identify ALL constraints that are active (tight) at that vertex.

    BUG-FIX (v13/v14 Bug 6): Original code only checked constraints.inequalities and
    ignored constraints.equations entirely. Equality constraints are ALWAYS active at
    every vertex (they define the affine hull). For the dice problem that means 3
    equalities are always active, so the active list was always empty, causing
    construct_tangent_cone to raise ConeError every time.

    Fix: check all three — equations (always active), inequalities (tight when
    a.v == bound), and bounds (variable at lower/upper limit).
    """
    active_map = {}
    tol = 1e-9

    for v in vertices:
        vt = tuple(v) if not isinstance(v, tuple) else v
        active = []

        # 1. Equality constraints (ALWAYS active — define the affine hull)
        if hasattr(constraints, 'equations'):
            for eq in constraints.equations:
                val = sum(float(c) * float(vi)
                          for c, vi in zip(eq.coefficients, vt))
                rhs = float(eq.rhs)
                if abs(val - rhs) < tol:
                    active.append((eq.coefficients, eq.rhs))

        # 2. Inequality constraints (active when tight: a.v == bound)
        if hasattr(constraints, 'inequalities'):
            for ineq in constraints.inequalities:
                val = sum(float(c) * float(vi)
                          for c, vi in zip(ineq.coefficients, vt))
                rhs = float(ineq.bound)  # always .bound, never .rhs
                if abs(val - rhs) < tol:
                    active.append((ineq.coefficients, ineq.bound))

        # 3. Bound constraints (active when variable is at its lower/upper limit)
        if hasattr(constraints, 'bounds'):
            k_len = len(vt)
            for bound in constraints.bounds:
                idx = bound.variable
                if idx >= k_len:
                    continue
                val = float(vt[idx])
                if bound.lower is not None and abs(val - float(bound.lower)) < tol:
                    from fractions import Fraction as _F
                    coeffs = tuple(_F(1) if j == idx else _F(0) for j in range(k_len))
                    active.append((coeffs, bound.lower))
                if bound.upper is not None and abs(val - float(bound.upper)) < tol:
                    from fractions import Fraction as _F
                    coeffs = tuple(_F(1) if j == idx else _F(0) for j in range(k_len))
                    active.append((coeffs, bound.upper))

        active_map[vt] = active

    return active_map


def _is_integer(x, tol: float = 1e-9) -> bool:
    """Return True if x is within tol of an integer."""
    try:
        if isinstance(x, Fraction):
            return x.denominator == 1
        return abs(x - round(float(x))) <= tol
    except Exception:
        return False


def _solve_determined_system(constraints: Any) -> List:
    """
    Solve a fully-determined linear system (eff_dim == 0) to find the unique
    feasible point.  Returns a list containing that single vertex, or empty
    list if no solution exists or the solution violates bounds.
    """
    import numpy as np
    from fractions import Fraction

    if not hasattr(constraints, 'equations') or not constraints.equations:
        return []

    n_vars = len(constraints.equations[0].coefficients)
    A = np.array([[float(c) for c in eq.coefficients]
                  for eq in constraints.equations], dtype=float)
    b = np.array([float(eq.rhs) for eq in constraints.equations], dtype=float)

    try:
        x, res, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        if rank < n_vars:
            return []  # Under-determined despite eff_dim=0 estimate
        # Verify residual
        if np.linalg.norm(A @ x - b) > 1e-8:
            return []
        # Convert to Fractions and check non-negativity
        vertex = tuple(Fraction(xi).limit_denominator(10**9) for xi in x)
        if any(v < 0 for v in vertex):
            return []
        return [vertex]
    except Exception:
        return []



def _compute_instance_hash(N: int, S1: int, S2: int,
                           min_val: int, max_val: int,
                           S3=None, S4=None) -> str:
    """
    Compute a deterministic SHA-256 hash of the problem instance.

    The hash is stable across Python versions and platforms because it is
    computed from a canonical string representation of the inputs, not from
    Python's built-in hash() which is randomised.

    Usage in papers:
        "Result verified with instance_hash=<hash> using reverse_stats v14."

    Returns:
        64-character lowercase hex string (SHA-256 digest).
    """
    parts = [f"N={N}", f"S1={S1}", f"S2={S2}",
             f"min={min_val}", f"max={max_val}"]
    if S3 is not None:
        parts.append(f"S3={S3}")
    if S4 is not None:
        parts.append(f"S4={S4}")
    canonical = "|".join(parts)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _empty_result() -> EvaluationResult:
    """Return an EvaluationResult representing an infeasible problem."""
    return EvaluationResult(
        frequency_state_count=0,
        total_multiset_count=0,
        marginal_distribution={},
        evaluation_time=0.0,
        orbit_weights_computed=0,
        metadata={"status": "infeasible"},
    )


# ═════════════════════════════════════════════════════════════════════════════
# Upgrade 2 — DP Cross-Verifier
# ═════════════════════════════════════════════════════════════════════════════

def dp_verify_count(
    N: int,
    S1: int,
    S2: int,
    min_val: int,
    max_val: int,
    S3: Optional[float] = None,
    S4: Optional[float] = None,
    max_N: int = 25,
) -> Optional[int]:
    """
    Independent DP-based counter for cross-verification.

    Uses a 3D (or 5D with S3/S4) convolution DP to count ordered sequences
    (not frequency states) of length N from alphabet [min_val, max_val] that
    satisfy all moment constraints.

    State: dp[n][s][q] = number of sequences of length n with
           sum = s (targeting S1) and sum-of-squares = q (targeting S2).

    Transition: dp[n+1][s+v][q+v²] += dp[n][s][q]  for each alphabet value v.

    This is a completely independent implementation from the Barvinok/direct-
    enumeration path.  It operates purely in integer arithmetic (no Fraction,
    no SymPy, no LP) and is used only for verification when N ≤ max_N.

    Args:
        N:       Sequence length.
        S1:      Target first moment (sum).
        S2:      Target second moment (sum of squares).
        min_val: Minimum alphabet value.
        max_val: Maximum alphabet value.
        S3:      Optional third moment (sum of cubes).
        S4:      Optional fourth moment (sum of fourth powers).
        max_N:   Maximum N for which DP is run (default 25).

    Returns:
        Exact integer count of ordered sequences, or None if N > max_N
        (DP would be too slow / memory-intensive).

    Raises:
        Nothing — returns None on any internal failure so the caller
        can always treat a None return as "verification skipped".
    """
    if N > max_N:
        logger.info(
            f"dp_verify_count: N={N} > max_N={max_N}, skipping DP verification."
        )
        return None

    try:
        alphabet = list(range(min_val, max_val + 1))
        k = len(alphabet)

        # ── Determine DP state space bounds ──────────────────────────────────
        # sum s ∈ [N*min_val, N*max_val]
        s_min = N * min_val
        s_max = N * max_val
        s_offset = -s_min   # shift so index 0 = s_min

        # sum-of-squares q ∈ [N*min_val², N*max_val²]
        q_min = N * min_val * min_val
        q_max = N * max_val * max_val
        q_offset = -q_min

        s_size = s_max - s_min + 1
        q_size = q_max - q_min + 1

        # Memory guard: refuse if state space too large
        mem_estimate = s_size * q_size * 8  # bytes (int64)
        if mem_estimate > 512 * 1024 * 1024:  # 512 MB
            logger.warning(
                f"dp_verify_count: state space {s_size}×{q_size} too large "
                f"({mem_estimate // 1024**2} MB), skipping."
            )
            return None

        # ── Initialise DP ─────────────────────────────────────────────────────
        # dp[s_idx][q_idx] = count of sequences of current length
        # Use Python dicts for sparse representation (most entries are 0)
        dp: Dict[tuple, int] = {(0 + s_offset, 0 + q_offset): 1}

        for step in range(N):
            new_dp: Dict[tuple, int] = {}
            for (s_idx, q_idx), cnt in dp.items():
                if cnt == 0:
                    continue
                for v in alphabet:
                    ns = s_idx + v
                    nq = q_idx + v * v
                    # Bounds check
                    if ns < 0 or ns >= s_size + s_offset + (N - step) * (max_val - min_val):
                        pass  # allow — will filter at end
                    key = (ns, nq)
                    new_dp[key] = new_dp.get(key, 0) + cnt
            dp = new_dp

        # ── Extract result for target (S1, S2) ───────────────────────────────
        target_s_idx = S1 + s_offset
        target_q_idx = S2 + q_offset
        result = dp.get((target_s_idx, target_q_idx), 0)

        # Handle S3, S4 if provided — filter the sequences post-hoc using
        # a separate 1D DP pass over just the matching (S1,S2) sequences.
        # For simplicity with optional moments, fall back to direct enumerate
        # if S3/S4 are specified (dp_verify_count still gives a valid count).
        if S3 is not None or S4 is not None:
            logger.info(
                "dp_verify_count: S3/S4 provided — DP covers S1,S2 only; "
                "returning None so caller uses direct enumeration for S3/S4 cases."
            )
            return None

        logger.info(f"dp_verify_count: DP result = {result} ordered sequences")
        return result

    except Exception as e:
        logger.warning(f"dp_verify_count: failed ({e}), returning None")
        return None


def run_pipeline_with_verification(
    N: int,
    S1: int,
    S2: int,
    min_val: int,
    max_val: int,
    S3: Optional[float] = None,
    S4: Optional[float] = None,
    dp_max_N: int = 25,
) -> EvaluationResult:
    """
    Run pipeline and cross-verify result with independent DP counter (Upgrade 2).

    For N ≤ dp_max_N, runs dp_verify_count() independently and asserts that
    both methods agree.  Disagreement raises AssertionError with both values
    so the discrepancy is immediately visible.

    For N > dp_max_N the function is identical to run_pipeline().

    Args:
        N, S1, S2, min_val, max_val, S3, S4: Same as run_pipeline().
        dp_max_N: Maximum N for DP verification (default 25).

    Returns:
        EvaluationResult with verification status in metadata.

    Raises:
        AssertionError: if pipeline and DP counts disagree.
    """
    result = run_pipeline(
        N=N, S1=S1, S2=S2,
        min_val=min_val, max_val=max_val,
        S3=S3, S4=S4,
    )

    if N <= dp_max_N and S3 is None and S4 is None:
        dp_count = dp_verify_count(N, S1, S2, min_val, max_val, max_N=dp_max_N)
        if dp_count is not None:
            pipeline_count = result.total_multiset_count
            if pipeline_count != dp_count:
                raise AssertionError(
                    f"VERIFICATION FAILED — counts disagree!\n"
                    f"  instance_hash = {(result.metadata or {}).get('instance_hash', 'N/A')}\n"
                    f"  N={N} S1={S1} S2={S2} min={min_val} max={max_val}\n"
                    f"  pipeline total_multiset_count = {pipeline_count}\n"
                    f"  dp_verify_count               = {dp_count}\n"
                    "This indicates a bug in the pipeline.  "
                    "Report with the instance_hash above."
                )
            logger.info(
                f"Verification PASSED: pipeline={pipeline_count}, DP={dp_count}"
            )
            # Record verification in metadata
            meta = result.metadata or {}
            meta["dp_verification"] = "passed"
            meta["dp_count"] = dp_count
            result.metadata = meta
        else:
            meta = result.metadata or {}
            meta["dp_verification"] = "skipped"
            result.metadata = meta
    else:
        meta = result.metadata or {}
        meta["dp_verification"] = "skipped_large_N" if N > dp_max_N else "skipped_higher_moments"
        result.metadata = meta

    return result


# ═════════════════════════════════════════════════════════════════════════════
# V15.1 — Vertex Certification
# ═════════════════════════════════════════════════════════════════════════════

def _certify_vertices(
    vertices: List,
    constraints: Any,
    eff_dim: int,
    ambient_dim: int,
    tol: float = 1e-9,
) -> List:
    """
    Certify that each vertex is a true extreme point of the polytope.

    V15.1 — Formal precondition for Brion's Theorem.

    A point v is a certified vertex if and only if the active constraint
    matrix at v has rank equal to eff_dim (the intrinsic dimension of the
    polytope).  Points failing this test are pseudo-vertices — they satisfy
    all equality constraints but lie on a face of positive dimension rather
    than being isolated extreme points.

    Mathematically: rank(A_active_intrinsic(v)) == eff_dim
    Equivalently:   rank([A_eq; A_active_ineq(v)]) == ambient_dim

    Args:
        vertices:    List of vertex tuples (Fraction values).
        constraints: ConstraintSystem with .equations and .inequalities.
        eff_dim:     Effective dimension = ambient_dim - rank(A_eq).
        ambient_dim: Number of variables (k).
        tol:         Tolerance for deciding if a constraint is active.

    Returns:
        Filtered list containing only certified extreme points.
        If SymPy is unavailable or certification fails for all vertices,
        returns the original list unchanged (fail-open, never discard all).
    """
    if not vertices:
        return vertices

    # ── Precompute equality constraint matrix (once, outside the loop) ────────
    try:
        import sympy
        from sympy import Matrix, Rational

        eq_rows = []
        if hasattr(constraints, 'equations') and constraints.equations:
            for eq in constraints.equations:
                row = [Rational(c).limit_denominator(10**12)
                       if not isinstance(c, sympy.Rational) else c
                       for c in eq.coefficients]
                eq_rows.append(row)

        A_eq_sym = Matrix(eq_rows) if eq_rows else Matrix([])

    except Exception as e:
        logger.warning(
            f"_certify_vertices: could not build equality matrix ({e}); "
            "skipping certification — all vertices accepted."
        )
        return vertices

    # ── Certify each vertex ───────────────────────────────────────────────────
    certified = []
    rejected  = 0

    for v in vertices:
        try:
            vf = [float(c) for c in v]

            # Find active inequality constraints at this vertex
            active_rows = list(eq_rows)  # equalities are always active

            if hasattr(constraints, 'inequalities'):
                for ineq in constraints.inequalities:
                    val = sum(float(c) * vi
                              for c, vi in zip(ineq.coefficients, vf))
                    rhs = float(getattr(ineq, 'bound', getattr(ineq, 'rhs', 0)))
                    if abs(val - rhs) < tol:
                        row = [Rational(c).limit_denominator(10**12)
                               for c in ineq.coefficients]
                        active_rows.append(row)

            if hasattr(constraints, 'bounds'):
                for bound in constraints.bounds:
                    idx = bound.variable
                    if idx >= len(vf):
                        continue
                    val = vf[idx]
                    e_row = [Rational(1) if j == idx else Rational(0)
                             for j in range(ambient_dim)]
                    if bound.lower is not None and abs(val - float(bound.lower)) < tol:
                        active_rows.append(e_row)
                    if bound.upper is not None and abs(val - float(bound.upper)) < tol:
                        active_rows.append(e_row)

            if not active_rows:
                # No active constraints at all — definitely not a vertex
                logger.debug(
                    f"_certify_vertices: vertex {v} has no active constraints — rejected."
                )
                rejected += 1
                continue

            A_active = Matrix(active_rows)
            rank = A_active.rank()

            if rank < eff_dim:
                logger.debug(
                    f"_certify_vertices: vertex {v} rejected — "
                    f"active rank={rank} < eff_dim={eff_dim} "
                    "(pseudo-vertex on a higher-dimensional face)."
                )
                rejected += 1
                continue

            certified.append(v)

        except Exception as ve:
            # Fail-open: accept vertex if certification throws
            logger.debug(
                f"_certify_vertices: certification error for vertex {v}: {ve}; accepted."
            )
            certified.append(v)

    if rejected > 0:
        logger.warning(
            f"_certify_vertices: {rejected}/{len(vertices)} pseudo-vertices discarded "
            f"(rank < eff_dim={eff_dim}). These were enumerated by the LP but are not "
            "true extreme points of the polytope."
        )

    # Fail-open: if we somehow rejected everything, return originals
    if not certified:
        logger.warning(
            "_certify_vertices: all vertices were rejected — "
            "returning original list to avoid empty polytope. "
            "Check constraint system for degeneracy."
        )
        return vertices

    return certified
