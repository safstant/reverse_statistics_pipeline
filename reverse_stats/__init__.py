"""
Reverse Statistics Pipeline
============================
Implements Barvinok's algorithm for exact multiset counting with
statistical constraints.

Quick start
-----------
    from reverse_stats import run_pipeline
    result = run_pipeline(N=100, S1=350, S2=1300, min_val=1, max_val=6)
    print(result.total_multiset_count)
"""

__version__ = "10.0.0"

import logging
import sys
import warnings
from typing import Optional, Dict, Any, Union, List

# ── Core types ────────────────────────────────────────────────────────────────
from .pipeline_types import (
    ObservedStatistics,
    AlphabetInfo,
    ConstraintSystem,
    PipelineConfig,
    ReverseStatsError,
    OptimizationDirection,
)

# ── Configuration ─────────────────────────────────────────────────────────────
from .config import (
    get_config,
    ReverseStatsConfig,
    get_pipeline_config,
    load_config,
    setup_environment,
)

# ── Main pipeline entry point ─────────────────────────────────────────────────
from .pipeline import run_pipeline, run_pipeline_with_verification, dp_verify_count

# ── Evaluation result type ────────────────────────────────────────────────────
from .evaluation import EvaluationResult, CountingResult

def check_python_version() -> None:
    """Check Python version compatibility. Warn if below 3.8."""
    if sys.version_info < (3, 8):
        warnings.warn(
            f"Python {sys.version_info.major}.{sys.version_info.minor} detected. "
            "Reverse Statistics requires Python 3.8 or higher for optimal performance.",
            RuntimeWarning,
            stacklevel=2
        )

check_python_version()

logger = logging.getLogger(__name__)

__all__ = [
    '__version__',
    # Main entry point
    'run_pipeline',
    # Result type
    'EvaluationResult',
    # Core types
    'ObservedStatistics',
    'AlphabetInfo',
    'ConstraintSystem',
    'PipelineConfig',
    'ReverseStatsError',
    'OptimizationDirection',
    # Configuration
    'get_config',
    'ReverseStatsConfig',
    'get_pipeline_config',
    'load_config',
    'setup_environment',
]

if __name__ == "__main__":
    print("=" * 60)
    print(f"Reverse Statistics Pipeline v{__version__}")
    print("=" * 60)
    print("\nUsage:")
    print("  from reverse_stats import run_pipeline")
    print("  result = run_pipeline(N=100, S1=350, S2=1300, min_val=1, max_val=6)")
    print("=" * 60)