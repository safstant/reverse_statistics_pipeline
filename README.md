# Reverse Statistics Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Exact counting of ordered sequences from observed integer statistics.**

Given sample size N and observed moments (sum, sum-of-squares, …), answers:
*"How many ordered sequences of integers from a given alphabet are consistent with these statistics?"*

Uses Barvinok's algorithm for exact lattice-point counting — the answer is always a precise
integer, never an approximation.

> **What the count means:** `total_multiset_count` is the number of **ordered sequences**
> (arrangements) consistent with the constraints — equivalently, the sum of multinomial
> coefficients over all valid frequency vectors. For frequency vector f, the weight is
> N! / (f₁! · f₂! · … · fₖ!).

---

## Installation

### From GitHub

```bash
pip install git+https://github.com/safstant/reverse_integer_statistics.git
```

### From a cloned repo

```bash
git clone https://github.com/safstant/reverse_integer_statistics.git
cd reverse_integer_statistics
pip install .
```

### With dev tools

```bash
pip install ".[dev]"
```

### Requirements

| Package | Min version |
|---------|-------------|
| Python  | 3.8         |
| numpy   | 1.21        |
| sympy   | 1.9         |
| scipy   | 1.7         |

---

## Quick Start

```python
from reverse_stats import run_pipeline

# How many ordered sequences of N=10 dice rolls (values 1–6)
# have sum=35 and sum-of-squares=140?
result = run_pipeline(
    N=10,
    S1=35,
    S2=140,
    min_val=1,
    max_val=6,
)

print(result.total_multiset_count)   # exact integer (ordered sequences)
print(result.frequency_state_count)  # number of distinct frequency vectors
```

### With higher moments

```python
result = run_pipeline(
    N=15,
    S1=45,
    S2=150,
    S3=520.0,   # third moment (optional)
    S4=1900.0,  # fourth moment (optional)
    min_val=1,
    max_val=6,
)
```

### Certify the Barvinok engine (v15.6+)

```python
from reverse_stats.certify import run_invariant_gate

# Runs the 7-point mathematical invariant gate.
# 6/7 pass without Normaliz; 7/7 with Normaliz installed.
report = run_invariant_gate(N=6, S1=21, S2=81, min_val=1, max_val=6)
report.print_report()
```

---

## API Reference

### `run_pipeline`

```python
from reverse_stats import run_pipeline

result = run_pipeline(
    N,                       # int   — sequence length
    S1,                      # int   — first moment   (Σ aᵢ fᵢ = S1)
    S2,                      # int   — second moment  (Σ aᵢ² fᵢ = S2)
    min_val=1,               # int   — alphabet lower bound (default 1)
    max_val=6,               # int   — alphabet upper bound (default 6)
    S3=None,                 # float — third moment  (optional)
    S4=None,                 # float — fourth moment (optional)
    verify=False,            # bool  — brute-force cross-check (slow, for testing)
    parallel=False,          # bool  — parallel orbit processing
    strict_barvinok=False,   # bool  — assert gf_terms>0; requires Normaliz (v15.5+)
)
```

**Note:** `alphabet_values` is not a parameter. The alphabet is always the integer range
`[min_val, min_val+1, …, max_val]`.

#### `strict_barvinok` flag

When `True`, the pipeline raises `AssertionError` if the Barvinok engine did not fire
(i.e. `gf_terms_constructed == 0`). Use this in production to guarantee that the
algebraic path was taken rather than the direct-enumeration fallback. Requires Normaliz
to be installed and on `PATH` for non-unimodular cones.

### `EvaluationResult`

| Field | Type | Description |
|-------|------|-------------|
| `total_multiset_count` | `int` | Number of ordered sequences satisfying constraints |
| `frequency_state_count` | `int` | Number of distinct frequency vectors |
| `marginal_distribution` | `dict` | Per-bin marginal fractions |
| `evaluation_time` | `float` | Seconds |
| `orbit_weights_computed` | `int` | Frequency states for which weights were computed |
| `metadata` | `dict` | Pipeline diagnostics (see below) |

#### `metadata` keys

| Key | Description |
|-----|-------------|
| `method` | `"barvinok_gf"` or `"direct_enumeration"` |
| `gf_terms_constructed` | Number of cones for which a GF was built (0 = Barvinok did not fire) |
| `num_cones` | Total tangent cones processed |
| `unimodular_cones` | Cones classified as unimodular |
| `num_vertices` | Polytope vertices found |
| `dimension` | Effective dimension of the frequency polytope |
| `version` | Pipeline version string |

---

## How It Works

```
Input: N, S1, S2 (and optionally S3, S4), alphabet [min_val..max_val]
        ↓
Build frequency polytope  P = {f ∈ Zᵏ : Cf = d,  0 ≤ fᵢ ≤ N}
        ↓
Enumerate vertices of P   (Normaliz if available, else internal fallback)
        ↓
Attach SNF intrinsic lattice basis to each tangent cone  (v15.4+)
        ↓
Certify all cones are pointed; classify unimodular / non-unimodular  (v15.5+)
        ↓
Barvinok / Brion signed decomposition into unimodular vertex cones
  • Normaliz path: projects to intrinsic ℤ^d, triangulates, lifts back  (v15.6+)
  • LLL path: Barvinok replacement algorithm with correct ±1 signs  (v15.7+)
        ↓
Sum vertex generating functions → rational function F(z)
        ↓
Evaluate F(1)  =  number of lattice points  =  frequency state count
        ↓
For each frequency vector f: add weight N! / (f₁! · … · fₖ!)
        ↓
Total = number of ordered sequences satisfying all constraints
```

All internal arithmetic uses `fractions.Fraction` and SymPy — no floating-point accumulation.

---

## External Tool: Normaliz (optional but recommended)

The pipeline uses [Normaliz](https://www.normaliz.uni-osnabrueck.de) for triangulation
of non-unimodular tangent cones. Without it, the LLL fallback path is used (v15.7:
correct Barvinok replacement algorithm with proper ±1 signs).

```bash
# macOS
brew install normaliz

# Ubuntu / Debian
sudo apt install normaliz

# Windows — download binary from:
# https://github.com/Normaliz/Normaliz/releases
# Add to PATH then verify:
normaliz --version
```

When Normaliz is present, `metadata["gf_terms_constructed"]` will be > 0, confirming
the full algebraic path was taken.

---

## Running Tests

```bash
# Clone and install with dev tools
git clone https://github.com/safstant/reverse_integer_statistics.git
cd reverse_integer_statistics
pip install -e ".[dev]"

# Run all tests
pytest

# Verbose output
pytest -v

# Skip slow tests
pytest -m "not slow"

# Step 18 LLL bug regression (v15.7)
python test_step18_lll.py
```

---

## Changelog

### v15.7
- **Fix (Step 18, Stage B):** Replaced incorrect pos/neg splitter with Barvinok's
  replacement algorithm. Finds the interior lattice point `w` of the fundamental
  parallelepiped via `M⁻¹·w ∈ (0,1)^d`, then decomposes as
  `C = Σᵢ sign(det(...)) · C(r₁,...,w,...,rᵈ)`.
- **Fix (Step 18, Stage A):** Fan sub-cone signs now assigned from `sign(det(fan_rays))`
  instead of all `+1`. Degenerate sub-cones (`det=0`) excluded.

### v15.6
- **Fix (Normaliz boundary):** `to_normaliz_format()` now projects to intrinsic `ℤ^d`
  before writing (`amb_space d`, not `amb_space n`). Results lifted back to ambient
  after unimodularity check in intrinsic space.
- **New:** `reverse_stats.certify.run_invariant_gate()` — 7-point mathematical
  certification suite.
- **Fix:** `_dot()` uses pure `Fraction` arithmetic (no float fallback).
- **Fix:** `cones.py` dimension property uses `sp.Matrix.rank()`.
- **Fix:** Square-matrix assert added in `brion.py` before det computation.

### v15.5
- SNF-based `intrinsic_lattice_basis()` replaces rref (exact integer projections).
- `certify_pointed()` uses `nullspace(R.T)` (exact SymPy).
- `strict_barvinok` parameter added to `run_pipeline`.

---

## License

MIT — see [LICENSE](LICENSE).
