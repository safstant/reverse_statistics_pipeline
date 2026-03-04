"""
Core exception hierarchy for Reverse Statistics Pipeline.
"""

class ReverseStatsError(Exception):
    """Base exception for Reverse Statistics."""
    pass

class ConfigurationError(ReverseStatsError):
    """Raised when configuration is invalid."""
    pass

class MathError(ReverseStatsError):
    """Base exception for mathematical computation errors."""
    pass

class GeometryError(MathError):
    """Raised for geometric/polytope/cone errors."""
    pass

class SymbolicError(MathError):
    """Raised for symbolic computation errors."""
    pass

class AlgorithmError(ReverseStatsError):
    """Raised for algorithmic/logical failures."""
    pass
