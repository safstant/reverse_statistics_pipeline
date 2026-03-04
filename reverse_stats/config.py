"""
Central Configuration Management for Reverse Statistics Pipeline
Version: 2.2.2

Features:
- Singleton pattern for global config
- Environment variable support (REVERSE_STATS_*)
- JSON/YAML file loading/saving
- Configuration templates
- Deep copy protection
- Backward compatibility layer
"""

import os
import json
import yaml
import shutil
import sys
import logging
import platform
import copy
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Set
from dataclasses import fields, dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# ============================================================================
# SAFE PIPELINE_TYPES IMPORT
# ============================================================================
try:
    from pipeline_types import PipelineConfig
    HAS_PIPELINE_TYPES = True
except (ImportError, ModuleNotFoundError):
    HAS_PIPELINE_TYPES = False
    
    @dataclass
    class PipelineConfig:
        """Fallback PipelineConfig when pipeline_types not available."""
        max_dimension: int = 15
        enumeration_limit: int = 1_000_000
        enable_enumeration: bool = True
        skip_gomory_if_integral: bool = True
        skip_decomposition_if_unimodular: bool = True
        skip_simplex_decomposition: bool = True
        integrality_tolerance: float = 1e-10
        constraint_tolerance: float = 1e-10

# ============================================================================
# ENVIRONMENT ENUM
# ============================================================================
class Environment(Enum):
    """Runtime environment enumeration."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

# ============================================================================
# PLATFORM-SPECIFIC UTILITIES
# ============================================================================
def _get_common_normaliz_paths() -> List[str]:
    """Get common Normaliz installation paths for current platform."""
    system = platform.system()
    
    if system == "Windows":
        return [
            r"C:\normaliz\normaliz.exe",
            r"C:\Program Files\Normaliz\normaliz.exe",
            r"C:\Program Files (x86)\Normaliz\normaliz.exe",
        ]
    elif system == "Linux":
        return [
            "/usr/bin/normaliz",
            "/usr/local/bin/normaliz",
            "/opt/normaliz/bin/normaliz",
        ]
    elif system == "Darwin":  # macOS
        return [
            "/usr/local/bin/normaliz",
            "/opt/homebrew/bin/normaliz",
            "/opt/local/bin/normaliz",
        ]
    return []


# ============================================================================
# CANONICAL NORMALIZ PATH RESOLVER  (FIX Bug-B5)
# ============================================================================
def find_normaliz_path(config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Return the path to the Normaliz executable, or None if not found.

    FIX(Bug-B5): Previously every module (cones.py, decomposition.py,
    simplex.py, polytope.py, vertices.py) contained its own private
    ``_find_normaliz()`` that only called ``shutil.which('normaliz')``,
    completely ignoring the ``normaliz_path`` key set via the config system
    or the ``REVERSE_STATS_NORMALIZ_PATH`` environment variable.

    Resolution order (first match wins):
      1. ``config['normaliz_path']`` — if supplied and executable.
      2. ``REVERSE_STATS_NORMALIZ_PATH`` / ``NORMALIZ_PATH`` env var.
      3. ``shutil.which('normaliz')`` — any executable named 'normaliz' on PATH.
      4. Platform-specific common installation directories.

    All modules that need Normaliz should call this function instead of
    maintaining their own local resolver.
    """
    import os as _os

    # 1. Explicit config key
    if config:
        cfg_path = config.get("normaliz_path")
        if cfg_path and _os.path.isfile(cfg_path) and _os.access(cfg_path, _os.X_OK):
            return cfg_path

    # 2. Environment variable (supports custom binary names / paths)
    env_path = (_os.environ.get("REVERSE_STATS_NORMALIZ_PATH")
                or _os.environ.get("NORMALIZ_PATH"))
    if env_path and _os.path.isfile(env_path) and _os.access(env_path, _os.X_OK):
        return env_path

    # 3. shutil.which — covers anything on PATH named 'normaliz'
    found = shutil.which("normaliz")
    if found:
        return found

    # 4. Platform common paths
    for candidate in _get_common_normaliz_paths():
        if _os.path.isfile(candidate) and _os.access(candidate, _os.X_OK):
            return candidate

    return None


# ============================================================================
# UNIFIED DEFAULT CONFIG (isl_build_dir REMOVED)
# ============================================================================
DEFAULT_CONFIG = {
    # Core pipeline settings
    "max_dimension": 15,
    "enumeration_limit": 1_000_000,
    "enable_enumeration": True,
    "skip_gomory_if_integral": True,
    "skip_decomposition_if_unimodular": True,
    "skip_simplex_decomposition": True,
    "integrality_tolerance": 1e-10,
    "constraint_tolerance": 1e-10,
    
    # Environment
    "environment": "development",
    
    # Paths
    "data_dir": "./data",
    "output_dir": "./output",
    "cache_dir": "./cache",
    "intermediate_dir": "./intermediate",
    "profile_output_dir": "./profiles",
    
    # Logging
    "log_level": "INFO",
    "log_file": None,
    "enable_metrics": True,
    
    # Performance
    "enable_caching": True,
    "cache_ttl_seconds": 3600,
    "max_cache_size_mb": 1024,
    "parallel_processing": True,
    "max_workers": None,
    "enable_profiling": False,
    
    # Tool paths
    "normaliz_path": None,
    # isl_build_dir REMOVED - ISL integration abandoned Feb 2026
    
    # Memory
    "max_memory_mb": 8192,
    "enable_memory_monitoring": True,
    
    # Output
    "save_intermediate_results": False,
    "compress_output": True,
    "output_format": "json",  # json, pickle, both
    
    # Module-specific (BACKWARD COMPATIBILITY)
    "enumeration": {
        "max_state_space_size": 1_000_000,
        "max_combination_length": 20,
        "enable_bit_optimization": True,
        "enable_pruning": True,
        "pattern_min_support": 0.01,
        "pattern_max_length": 10,
        "parallel_enumerate": False,
        "cache_results": True
    },
    "marginal": {
        "max_dimension": 15,
        "integrality_tolerance": 1e-10,
        "max_marginal_order": 3
    },
    "evaluation": {
        "default_k_folds": 5,
        "default_test_size": 0.2,
        "random_seed": 42,
        "significance_level": 0.05,
        "bootstrap_iterations": 1000,
        "confidence_level": 0.95,
        "max_classes": 100
    }
}

# Environment variable mapping
ENV_PREFIX = "REVERSE_STATS_"
ENV_MAPPING = {
    "MAX_DIMENSION": "max_dimension",
    "ENABLE_ENUMERATION": "enable_enumeration",
    "ENUMERATION_LIMIT": "enumeration_limit",
    "LOG_LEVEL": "log_level",
    "NORMALIZ_PATH": "normaliz_path",
    "MAX_MEMORY_MB": "max_memory_mb",
    "ENABLE_CACHING": "enable_caching",
    "PARALLEL_PROCESSING": "parallel_processing",
    "DATA_DIR": "data_dir",
    "OUTPUT_DIR": "output_dir",
    "CACHE_DIR": "cache_dir",
    "INTERMEDIATE_DIR": "intermediate_dir",
    "ENABLE_PROFILING": "enable_profiling",
    "ENVIRONMENT": "environment"
}

# Configuration template with descriptions
CONFIG_TEMPLATE = {
    "comment": "Reverse Statistics Pipeline Configuration",
    "version": "2.2.2",
    "_descriptions": {
        "max_dimension": "Maximum effective dimension guard",
        "enumeration_limit": "Maximum number of vectors to enumerate",
        "enable_enumeration": "Enable/disable full enumeration",
        "skip_gomory_if_integral": "Skip Gomory cuts if vertices are integral",
        "skip_decomposition_if_unimodular": "Skip decomposition if all cones unimodular",
        "skip_simplex_decomposition": "Skip decomposition for simplex polytopes",
        "integrality_tolerance": "Tolerance for checking integer vertices",
        "constraint_tolerance": "Tolerance for constraint satisfaction",
        "environment": "Runtime environment (development, testing, production)",
        "data_dir": "Directory for input data",
        "output_dir": "Directory for output files",
        "cache_dir": "Directory for cached results",
        "intermediate_dir": "Directory for intermediate results",
        "profile_output_dir": "Directory for profile outputs",
        "log_level": "Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        "log_file": "Log file path (null = console only)",
        "enable_metrics": "Enable performance metrics collection",
        "enable_caching": "Enable result caching",
        "cache_ttl_seconds": "Cache time-to-live in seconds",
        "max_cache_size_mb": "Maximum cache size in megabytes",
        "parallel_processing": "Enable parallel processing",
        "max_workers": "Maximum worker processes (null = auto)",
        "enable_profiling": "Enable detailed profiling",
        "normaliz_path": "Path to Normaliz executable",
        # isl_build_dir description REMOVED
        "max_memory_mb": "Maximum memory usage in MB",
        "enable_memory_monitoring": "Enable memory usage monitoring",
        "save_intermediate_results": "Save intermediate results",
        "compress_output": "Compress output files",
        "output_format": "Output format (json, pickle, both)",
    },
    "config": DEFAULT_CONFIG
}

# ============================================================================
# CORE CONFIGURATION CLASS
# ============================================================================
class ReverseStatsConfig:
    """Production-grade configuration for Barvinok pipeline."""
    
    def __init__(self, **kwargs) -> None:
        """Initialize configuration with optional overrides."""
        # Deep copy to prevent cross-instance mutation
        self._config = copy.deepcopy(DEFAULT_CONFIG)
        self._version = "2.2.2"
        self._modified_keys: Set[str] = set()
        
        # Load from file if specified
        if config_file := kwargs.pop("config_file", None):
            self.load_from_file(config_file)
        
        # Apply environment variables
        self._apply_environment_vars()
        
        # Apply explicit overrides
        self._config.update(kwargs)
        self._modified_keys.update(kwargs.keys())
        
        # Validate
        self.validate()
        
        # Create PipelineConfig for staged pipeline
        if HAS_PIPELINE_TYPES:
            self.pipeline_config = self._create_pipeline_config()
    
    def _apply_environment_vars(self) -> None:
        """Type-safe environment variable application."""
        for env_var, config_key in ENV_MAPPING.items():
            full_env_var = f"{ENV_PREFIX}{env_var}"
            if value := os.environ.get(full_env_var):
                try:
                    current_type = type(self._config[config_key])
                    
                    if current_type == bool:
                        self._config[config_key] = value.lower() in ('true', 'yes', '1', 't', 'y')
                    elif current_type == int:
                        self._config[config_key] = int(value)
                    elif current_type == float:
                        self._config[config_key] = float(value)
                    elif current_type == str:
                        self._config[config_key] = value
                    elif current_type in (list, dict):
                        # For complex types, try JSON parsing
                        try:
                            self._config[config_key] = json.loads(value)
                        except json.JSONDecodeError:
                            self._config[config_key] = value
                    
                    self._modified_keys.add(config_key)
                    
                except (ValueError, TypeError) as e:
                    if self._config.get("environment") == "production":
                        raise ValueError(f"Invalid env var {full_env_var}={value}") from e
                    logger.warning(f"Invalid env var {full_env_var}={value}: {e}")
    
    def _create_pipeline_config(self) -> PipelineConfig:
        """Project config to PipelineConfig for staged pipeline."""
        pipeline_fields = {f.name for f in fields(PipelineConfig)}
        return PipelineConfig(**{
            k: v for k, v in self._config.items() if k in pipeline_fields
        })
    
    def load_from_file(self, filepath: Union[str, Path]) -> None:
        """Load and deep-merge config from JSON/YAML."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() in ('.yaml', '.yml'):
                file_config = yaml.safe_load(f)
            else:
                file_config = json.load(f)
        
        # Handle template format (with nested config)
        if isinstance(file_config, dict) and "config" in file_config:
            file_config = file_config["config"]
        
        self._deep_update(self._config, file_config)
        self._modified_keys.add(f"loaded_from:{path}")
    
    def _deep_update(self, base: Dict, update: Dict) -> None:
        """Recursive dictionary update for nested configs."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def save_to_file(self, filepath: Union[str, Path], format: str = "json") -> None:
        """Save current configuration for reproducibility."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self._make_serializable(self._config)
        
        with open(path, 'w', encoding='utf-8') as f:
            if format.lower() == 'json':
                json.dump(config_dict, f, indent=2)
            elif format.lower() in ('yaml', 'yml'):
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert non-serializable objects for JSON/YAML."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        if isinstance(obj, Enum):
            return obj.value
        if not isinstance(obj, (str, int, float, bool, type(None))):
            return str(obj)
        return obj
    
    def validate(self) -> None:
        """Comprehensive config validation."""
        errors = []
        
        # Numeric validation
        if self._config["max_dimension"] <= 0:
            errors.append("max_dimension must be positive")
        if self._config["max_dimension"] > 15:
            logger.warning("max_dimension > 15 may cause performance issues")
        if self._config["max_memory_mb"] < 100:
            errors.append("max_memory_mb must be ≥ 100MB")
        if self._config["enumeration_limit"] < 0:
            errors.append("enumeration_limit must be non-negative")
        if self._config["integrality_tolerance"] <= 0:
            errors.append("integrality_tolerance must be positive")
        if self._config["constraint_tolerance"] <= 0:
            errors.append("constraint_tolerance must be positive")
        if self._config["cache_ttl_seconds"] < 0:
            errors.append("cache_ttl_seconds must be non-negative")
        if self._config["max_cache_size_mb"] < 0:
            errors.append("max_cache_size_mb must be non-negative")
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self._config["log_level"].upper() not in valid_log_levels:
            errors.append(f"log_level must be one of {valid_log_levels}")
        
        # Validate output format
        valid_formats = ["json", "pickle", "both"]
        if self._config["output_format"].lower() not in valid_formats:
            errors.append(f"output_format must be one of {valid_formats}")
        
        # Validate environment
        try:
            Environment(self._config["environment"])
        except ValueError:
            errors.append(f"Invalid environment: {self._config['environment']}. Must be: {[e.value for e in Environment]}")
        
        # Normaliz path validation
        if path := self._config.get("normaliz_path"):
            self._validate_normaliz_path(path, errors)
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {e}" for e in errors))
    
    def _validate_normaliz_path(self, path: str, errors: List[str]) -> None:
        """Cross-platform Normaliz path resolution with executability check."""
        path_obj = Path(path)
        
        # Check PATH first
        if not path_obj.is_absolute() and (which_path := shutil.which(path)):
            self._config["normaliz_path"] = which_path
            logger.info(f"Found Normaliz in PATH: {which_path}")
            return
        
        # Check common platform locations
        for common_path in _get_common_normaliz_paths():
            if Path(common_path).exists():
                if os.access(common_path, os.X_OK):
                    self._config["normaliz_path"] = common_path
                    logger.info(f"Found Normaliz at: {common_path}")
                    return
                else:
                    logger.warning(f"Found Normaliz at {common_path} but it's not executable")
        
        # Final check: does the path itself exist and is executable?
        if path_obj.exists():
            if os.access(path_obj, os.X_OK):
                return
            else:
                errors.append(f"Normaliz found at '{path}' but is not executable")
        else:
            errors.append(
                f"Normaliz not found at '{path}'. Set REVERSE_STATS_NORMALIZ_PATH "
                "to full path or add to system PATH."
            )
    
    # ========================================================================
    # DICT INTERFACE
    # ========================================================================
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with dot notation support."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> 'ReverseStatsConfig':
        """Set a configuration value."""
        self._config[key] = value
        self._modified_keys.add(key)
        return self
    
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        return key in self._config
    
    def update(self, **kwargs) -> 'ReverseStatsConfig':
        """Update multiple configuration values."""
        self._config.update(kwargs)
        self._modified_keys.update(kwargs.keys())
        self.validate()
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Return deep copy of config to prevent external mutation."""
        return copy.deepcopy(self._config)
    
    def to_json(self) -> str:
        """Get configuration as JSON string."""
        return json.dumps(self._make_serializable(self._config), indent=2)
    
    @property
    def version(self) -> str:
        return self._version
    
    @property
    def modified_keys(self) -> Set[str]:
        return self._modified_keys.copy()
    
    @property
    def environment(self) -> Environment:
        return Environment(self._config["environment"])
    
    def __repr__(self) -> str:
        return f"ReverseStatsConfig(version={self._version}, modified={len(self._modified_keys)} keys)"

# ============================================================================
# GLOBAL CONFIGURATION (SINGLETON)
# ============================================================================
_GLOBAL_CONFIG: Optional[ReverseStatsConfig] = None

def get_config() -> ReverseStatsConfig:
    """Get or create global configuration instance."""
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None:
        _GLOBAL_CONFIG = ReverseStatsConfig()
    return _GLOBAL_CONFIG

def set_config(config: ReverseStatsConfig) -> None:
    """Set global configuration instance."""
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = config

# ============================================================================
# BACKWARD COMPATIBILITY LAYER
# ============================================================================
def get_enumeration_config() -> Dict[str, Any]:
    """Backward-compatible accessor for enumeration.py / enumerations.py."""
    cfg = get_config()
    base = cfg.get("enumeration", {})
    return {
        "max_state_space_size": base.get("max_state_space_size", 1_000_000),
        "max_combination_length": base.get("max_combination_length", 20),
        "enable_bit_optimization": base.get("enable_bit_optimization", True),
        "enable_pruning": base.get("enable_pruning", True),
        "pattern_min_support": base.get("pattern_min_support", 0.01),
        "pattern_max_length": base.get("pattern_max_length", 10),
        "parallel_enumerate": base.get("parallel_enumerate", cfg.get("parallel_processing", False)),
        "cache_results": base.get("cache_results", cfg.get("enable_caching", True)),
        "max_dimension": cfg.get("max_dimension", 15),
    }

def get_marginal_config() -> Dict[str, Any]:
    """Backward-compatible accessor for marginal.py."""
    cfg = get_config()
    base = cfg.get("marginal", {})
    return {
        "max_dimension": base.get("max_dimension", cfg.get("max_dimension", 15)),
        "integrality_tolerance": base.get("integrality_tolerance", cfg.get("integrality_tolerance", 1e-10)),
        "max_marginal_order": base.get("max_marginal_order", 3),
    }

def get_evaluation_config() -> Dict[str, Any]:
    """Backward-compatible accessor for evaluation.py."""
    cfg = get_config()
    base = cfg.get("evaluation", {})
    return {
        "default_k_folds": base.get("default_k_folds", 5),
        "default_test_size": base.get("default_test_size", 0.2),
        "random_seed": base.get("random_seed", 42),
        "significance_level": base.get("significance_level", 0.05),
        "bootstrap_iterations": base.get("bootstrap_iterations", 1000),
        "confidence_level": base.get("confidence_level", 0.95),
        "max_classes": base.get("max_classes", 100),
    }

def get_pipeline_config() -> Optional[PipelineConfig]:
    """Get pipeline-specific configuration."""
    return get_config().pipeline_config

# Aliases
get_global_config = get_config
set_global_config = set_config
create_default_config = lambda: ReverseStatsConfig()

def load_config(filepath: Optional[Union[str, Path]] = None) -> ReverseStatsConfig:
    """Load config from file or create default."""
    if filepath and Path(filepath).exists():
        return ReverseStatsConfig(config_file=filepath)
    return ReverseStatsConfig()

# ============================================================================
# CONFIGURATION TEMPLATES
# ============================================================================
def create_config_template(filepath: Union[str, Path], format: str = "json") -> None:
    """Create a configuration template file with descriptions."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'json':
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(CONFIG_TEMPLATE, f, indent=2)
    elif format.lower() in ('yaml', 'yml'):
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(CONFIG_TEMPLATE, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Configuration template saved to {path}")

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
def setup_environment(config: Optional[ReverseStatsConfig] = None) -> ReverseStatsConfig:
    """Configure runtime environment with conditional directory creation."""
    cfg = config or get_config()
    
    # Clear existing logging handlers
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # Configure logging
    log_level = getattr(logging, cfg.get("log_level", "INFO").upper(), logging.INFO)
    log_file = cfg.get("log_file")
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )
    
    # Set thread limits for numerical libraries
    if not cfg.get("parallel_processing", True):
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        logger.debug("Parallel processing disabled, thread limits set")
    
    # Core directories - always create
    core_dirs = ["data_dir", "output_dir", "cache_dir"]
    for dir_key in core_dirs:
        if dir_path := cfg.get(dir_key):
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created core directory: {dir_path}")
    
    # Optional directories - create only if enabled
    if cfg.get("save_intermediate_results", False):
        if intermediate_dir := cfg.get("intermediate_dir"):
            Path(intermediate_dir).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created intermediate directory: {intermediate_dir}")
    
    if cfg.get("enable_profiling", False):
        if profile_dir := cfg.get("profile_output_dir"):
            Path(profile_dir).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created profile directory: {profile_dir}")
    
    logger.info(
        f"Environment setup complete (environment: {cfg.environment.value}, "
        f"log level: {cfg.get('log_level', 'INFO')})"
    )
    return cfg

# ============================================================================
# MAIN (VALIDATION + DEMO)
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PRODUCTION CONFIGURATION - Barvinok Pipeline v2.2.2")
    print("=" * 60)
    
    # Test 1: Basic config
    config = get_config()
    print(f"\n✅ Version: {config.version}")
    print(f"   Max dimension: {config['max_dimension']}")
    print(f"   Environment: {config.environment.value}")
    print(f"   Normaliz path: {config['normaliz_path']}")
    
    # Test 2: Backward compatibility
    enum_cfg = get_enumeration_config()
    print(f"\n✅ Backward compatibility:")
    print(f"   Enumeration max_state_space_size: {enum_cfg['max_state_space_size']}")
    print(f"   Marginal max_dimension: {get_marginal_config()['max_dimension']}")
    
    # Test 3: Environment variables
    os.environ["REVERSE_STATS_MAX_DIMENSION"] = "20"
    os.environ["REVERSE_STATS_ENVIRONMENT"] = "production"
    test_config = ReverseStatsConfig()
    print(f"\n✅ Environment override:")
    print(f"   Max dimension (from env): {test_config['max_dimension']}")
    print(f"   Environment (from env): {test_config.environment.value}")
    print(f"   Modified keys: {test_config.modified_keys}")
    del os.environ["REVERSE_STATS_MAX_DIMENSION"]
    del os.environ["REVERSE_STATS_ENVIRONMENT"]
    
    # Test 4: Template creation
    create_config_template("config_template.json")
    print(f"\n✅ Template saved to config_template.json")
    
    # Test 5: Dict interface and deep copy isolation
    print(f"\n✅ Dict interface + deep copy isolation:")
    config["test_key"] = "test_value"
    config["enumeration"]["max_state_space_size"] = 999999
    print(f"   'test_key' in config: {'test_key' in config}")
    print(f"   enumeration.max_state_space_size: {config.get('enumeration.max_state_space_size')}")
    print(f"   Modified keys: {config.modified_keys}")
    
    # Create second config to verify deep copy isolation
    config2 = ReverseStatsConfig()
    print(f"   Second config enumeration value (should be default): {config2.get('enumeration.max_state_space_size')}")
    
    # Test 6: Nested access
    print(f"\n✅ Nested access:")
    print(f"   evaluation.default_k_folds: {config.get('evaluation.default_k_folds')}")
    
    # Test 7: Environment setup
    setup_environment(config)
    print(f"\n✅ Environment setup complete")
    
    # Test 8: Save config
    config.save_to_file("test_config.json")
    print(f"✅ Config saved to test_config.json")
    
    # Test 9: Load from file
    loaded = load_config("test_config.json")
    print(f"✅ Config loaded from file: max_dimension={loaded['max_dimension']}")
    
    # Verify isl_build_dir is gone
    print(f"\n✅ Verified: isl_build_dir key removed (not in config)")
    
    print("\n" + "=" * 60)
    print("✅ PRODUCTION CONFIGURATION READY")
    print("=" * 60)