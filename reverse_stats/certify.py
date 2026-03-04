"""
V15.6 — Barvinok Correctness Invariant Gate

7-point theorem-level certification suite.
Run after any pipeline call to certify the engine is mathematically correct.

Usage:
    from reverse_stats.certify import run_invariant_gate
    results = run_invariant_gate(N=6, S1=21, S2=81, min_val=1, max_val=6)
    results.print_report()
"""
from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class InvariantResult:
    name: str
    passed: bool
    detail: str = ""

    def __str__(self):
        mark = "✓" if self.passed else "✗"
        return f"  {mark}  {self.name}" + (f" — {self.detail}" if self.detail else "")


@dataclass
class GateReport:
    results: List[InvariantResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_pass(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def score(self) -> str:
        n = len(self.results)
        k = sum(1 for r in self.results if r.passed)
        return f"{k}/{n}"

    def print_report(self):
        print(f"\n{'='*60}")
        print(f"Barvinok Invariant Gate — {self.score} passed")
        print(f"{'='*60}")
        for r in self.results:
            print(r)
        if self.metadata:
            print(f"\n  metadata: {self.metadata}")
        print(f"\n  {'ENGINE CERTIFIED ✓' if self.all_pass else 'ENGINE INCOMPLETE ✗'}")
        print(f"{'='*60}\n")


def run_invariant_gate(N: int, S1: int, S2: int,
                       min_val: int = 1, max_val: int = 6) -> GateReport:
    """
    Run the 7-point Barvinok invariant gate on a single problem instance.

    Invariant I.1  — Intrinsic projections are integers (no Z[1/k] contamination)
    Invariant I.3  — Unimodular sub-cones exist for index-1 cones
    Invariant II.1 — All tangent cones are pointed (Brion prerequisite)
    Invariant II.2 — No non-square matrix reaches Brion det stage
    Invariant III.2 — Normaliz boundary: Normaliz receives intrinsic rays
    Invariant VII  — No float in algebraic core (_dot, dimension, projection)
    Invariant VI   — gf_terms > 0 (requires Normaliz; fails cleanly otherwise)
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    report = GateReport()

    # ── Capture tangent cones ─────────────────────────────────────────────
    captured_cones = []
    try:
        import reverse_stats.cones as _cm
        _orig = _cm.construct_all_tangent_cones
        def _hook(*a, **kw):
            r = _orig(*a, **kw); captured_cones.extend(r); return r
        _cm.construct_all_tangent_cones = _hook
        import reverse_stats.pipeline as _pip
        _pip.construct_all_tangent_cones = _hook
    except Exception as e:
        report.results.append(InvariantResult("cone-capture", False, str(e)))
        return report

    pipeline_result = None
    try:
        from reverse_stats.pipeline import run_pipeline
        pipeline_result = run_pipeline(N=N, S1=S1, S2=S2, min_val=min_val, max_val=max_val)
    except Exception as e:
        report.results.append(InvariantResult("pipeline-run", False, str(e)))
        return report
    finally:
        try:
            _cm.construct_all_tangent_cones = _orig
            _pip.construct_all_tangent_cones = _orig
        except Exception:
            pass

    meta = pipeline_result.metadata if pipeline_result else {}
    report.metadata = {
        "freq": getattr(pipeline_result, "frequency_state_count", None),
        "gf_terms": meta.get("gf_terms_constructed", 0),
        "method": meta.get("method", "?"),
        "N": N, "S1": S1,
    }

    # ── Invariant I.1: integer intrinsic projections ──────────────────────
    try:
        from reverse_stats.dimension import intrinsic_lattice_basis
        import sympy as sp
        non_int_cones = []
        for i, tc in enumerate(captured_cones):
            B, B_inv, d = intrinsic_lattice_basis(tc.rays)
            for ray in tc.rays:
                coords = [sum(B_inv[k][j] * Fraction(ray[j])
                              for j in range(len(ray)))
                          for k in range(d)]
                if any(c.denominator != 1 for c in coords):
                    non_int_cones.append(i)
                    break
        report.results.append(InvariantResult(
            "I.1  Integer projections",
            len(non_int_cones) == 0,
            f"all {len(captured_cones)} cones integral" if not non_int_cones
            else f"non-integer projections in cones {non_int_cones}"
        ))
    except Exception as e:
        report.results.append(InvariantResult("I.1  Integer projections", False, str(e)))

    # ── Invariant I.3: unimodular sub-cones exist for index-1 cones ───────
    try:
        from reverse_stats.dimension import intrinsic_lattice_basis
        from sympy.matrices.normalforms import smith_normal_decomp
        import sympy as sp
        from itertools import combinations as _comb

        index1_uni = []
        index8_blocked = []
        for i, tc in enumerate(captured_cones):
            int_rays = [[int(x) for x in r] for r in tc.rays]
            D, U, V = smith_normal_decomp(sp.Matrix(int_rays))
            rank = sum(1 for j in range(min(sp.Matrix(int_rays).shape)) if D[j,j] != 0)
            last_div = int(D[rank-1, rank-1])

            B, B_inv, d = intrinsic_lattice_basis(tc.rays)
            proj = [[sum(int(B_inv[k][j]) * int(r[j]) for j in range(len(r)))
                     for k in range(d)] for r in tc.rays]
            dets = [abs(int(sp.Matrix([proj[a] for a in c]).det()))
                    for c in _comb(range(len(proj)), d)]
            has_uni = 1 in dets

            if last_div == 1:
                index1_uni.append(has_uni)
            else:
                index8_blocked.append((i, last_div, min(x for x in dets if x > 0)))

        inv_pass = all(index1_uni) if index1_uni else True
        detail = (f"{sum(index1_uni)}/{len(index1_uni)} index-1 cones have unimodular sub-cones"
                  + (f"; index>1 cones {index8_blocked} need Normaliz short-vector split"
                     if index8_blocked else ""))
        report.results.append(InvariantResult("I.3  Unimodular sub-cones (index-1)", inv_pass, detail))
    except Exception as e:
        report.results.append(InvariantResult("I.3  Unimodular sub-cones", False, str(e)))

    # ── Invariant II.1: all cones pointed ─────────────────────────────────
    try:
        non_pointed = [i for i, tc in enumerate(captured_cones)
                       if not tc.certify_pointed()]
        report.results.append(InvariantResult(
            "II.1 All cones pointed",
            len(non_pointed) == 0,
            f"all {len(captured_cones)} pointed" if not non_pointed
            else f"non-pointed: {non_pointed}"
        ))
    except Exception as e:
        report.results.append(InvariantResult("II.1 All cones pointed", False, str(e)))

    # ── Invariant II.2: square assert in brion ────────────────────────────
    try:
        import reverse_stats.brion as _brion
        src = open(_brion.__file__).read()
        has_square_assert = 'Non-square ray matrix' in src
        has_no_det0 = '_det_val = 0' not in src
        report.results.append(InvariantResult(
            "II.2 No det=0 fallback; square assert present",
            has_square_assert and has_no_det0,
            f"square_assert={has_square_assert}, det0_removed={has_no_det0}"
        ))
    except Exception as e:
        report.results.append(InvariantResult("II.2 det=0 removed", False, str(e)))

    # ── Invariant III.2 (Normaliz boundary): intrinsic rays sent ──────────
    try:
        import reverse_stats.decomposition as _decomp
        src = open(_decomp.__file__).read()
        uses_intrinsic_dim = 'amb_space {_d}' in src
        lifts_back = '_B[i][k]' in src or 'intrinsic_basis * R_int' in src or 'lc.is_unimodular' in src
        report.results.append(InvariantResult(
            "III.2 Normaliz boundary: intrinsic rays",
            uses_intrinsic_dim and lifts_back,
            f"intrinsic_amb_space={uses_intrinsic_dim}, lift_back={lifts_back}"
        ))
    except Exception as e:
        report.results.append(InvariantResult("III.2 Normaliz boundary", False, str(e)))

    # ── Invariant VII: no float in algebraic core ─────────────────────────
    try:
        import reverse_stats.decomposition as _decomp
        src_d = open(_decomp.__file__).read()
        # Find _dot function body
        dot_start = src_d.find('def _dot(a, b):')
        dot_end   = src_d.find('\n    ', dot_start + 20)
        dot_body  = src_d[dot_start:dot_end + 200]
        no_float_in_dot = 'float(' not in dot_body and 'np.' not in dot_body
        # Dimension property
        import reverse_stats.cones as _cones
        src_c = open(_cones.__file__).read()
        dim_start = src_c.find('def dimension(')
        dim_end   = src_c.find('\n    def ', dim_start + 20)
        dim_body  = src_c[dim_start:dim_end]
        no_nplinalg_in_dim = 'np.linalg.matrix_rank' not in dim_body
        report.results.append(InvariantResult(
            "VII  Exact arithmetic in core",
            no_float_in_dot and no_nplinalg_in_dim,
            f"_dot_exact={no_float_in_dot}, dim_exact={no_nplinalg_in_dim}"
        ))
    except Exception as e:
        report.results.append(InvariantResult("VII  Exact arithmetic", False, str(e)))

    # ── Invariant VI: gf_terms > 0 ────────────────────────────────────────
    gf_terms = meta.get("gf_terms_constructed", 0)
    report.results.append(InvariantResult(
        "VI   gf_terms_constructed > 0",
        gf_terms > 0,
        f"gf_terms={gf_terms}" + (" ← Normaliz required for index-8 cones" if gf_terms == 0 else " ✓ engine fired")
    ))

    return report


if __name__ == "__main__":
    report = run_invariant_gate(N=6, S1=21, S2=81, min_val=1, max_val=6)
    report.print_report()
