"""
MetaMachine Environments Module

This module provides simulation and real robot environments for the MetaMachine framework.

Core Classes:
    - MetaMachine: Main simulation environment (env_sim.py)
    - RealMetaMachine: Real robot environment using capybarish (env_real.py)
    - RayVecMetaMachine: Vectorized environment using Ray for parallel execution (vec_env.py)
    - VecEnv: Abstract base class for vectorized environments (vec_env.py)

Factory Function:
    - make_env(cfg): Creates the appropriate environment based on config mode

Example:
    >>> from metamachine.environments import make_env
    >>> from metamachine.environments.configs.config_registry import ConfigRegistry
    >>> 
    >>> # Load config with mode: "sim" or "real"
    >>> cfg = ConfigRegistry.create_from_file("my_config.yaml")
    >>> 
    >>> # Factory automatically creates correct environment type
    >>> env = make_env(cfg)
    >>> 
    >>> # Or use specific classes directly
    >>> from metamachine.environments import MetaMachine, RealMetaMachine
    >>> sim_env = MetaMachine(cfg)
    >>> real_env = RealMetaMachine(cfg)

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>
"""

from typing import Optional

from .base import Base

# IMPORTANT:
# Importing MuJoCo-backed simulation envs at module import time makes *any*
# `import metamachine.environments...` crash on machines without a working MuJoCo
# GL setup (e.g. macOS with MUJOCO_GL=egl). We therefore lazy-import `MetaMachine`
# and only raise the underlying error if/when the user actually requests a sim env.
_SIM_IMPORT_ERROR: Optional[Exception] = None
try:
    from .env_sim import MetaMachine  # type: ignore
except Exception as e:  # MuJoCo can raise RuntimeError during import
    MetaMachine = None  # type: ignore[assignment]
    _SIM_IMPORT_ERROR = e

# Optional import for real robot environment (requires capybarish)
try:
    from .env_real import RealMetaMachine
except ImportError:
    RealMetaMachine = None

# Optional imports for vectorized environments (require Ray)
try:
    from .vec_env import RayVecMetaMachine, VecEnv, StateSnapshot
except ImportError:
    # Ray not available
    RayVecMetaMachine = None
    VecEnv = None
    StateSnapshot = None


def make_env(cfg, **kwargs):
    """Factory function to create the appropriate environment based on config.
    
    This function checks `cfg.environment.mode` and creates either:
    - MetaMachine (simulation) if mode == "sim" or not specified
    - RealMetaMachine (real robot) if mode == "real"
    
    Args:
        cfg: Configuration object (OmegaConf) with environment settings
        **kwargs: Additional arguments passed to the environment constructor
    
    Returns:
        Environment instance (MetaMachine or RealMetaMachine)
    
    Raises:
        ValueError: If mode is "real" but capybarish is not installed
        ValueError: If mode is unknown
    
    Example:
        >>> from metamachine.environments import make_env
        >>> from metamachine.environments.configs.config_registry import ConfigRegistry
        >>> 
        >>> # For simulation
        >>> cfg = ConfigRegistry.create_from_file("config.yaml")
        >>> cfg.environment.mode = "sim"
        >>> sim_env = make_env(cfg)
        >>> 
        >>> # For real robot
        >>> cfg.environment.mode = "real"
        >>> real_env = make_env(cfg)
    """
    # Get mode from config
    mode = cfg.environment.get("mode", "sim").lower()
    
    if mode == "sim" or mode == "simulation":
        if MetaMachine is None:
            raise RuntimeError(
                "Simulation environment (MetaMachine) could not be imported. "
                "This is usually due to a MuJoCo installation/GL backend issue. "
                "Original error:\n"
                f"{_SIM_IMPORT_ERROR}"
            )
        return MetaMachine(cfg, **kwargs)
    
    elif mode == "real":
        if RealMetaMachine is None:
            raise ValueError(
                "Real robot mode requires capybarish. "
                "Install it with: pip install capybarish"
            )
        return RealMetaMachine(cfg, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown environment mode: '{mode}'. "
            f"Use 'sim' for simulation or 'real' for real robot."
        )


__all__ = [
    # Core environments
    "MetaMachine",
    "RealMetaMachine",
    "Base",
    
    # Factory function
    "make_env",
    
    # Vectorized environments
    "RayVecMetaMachine",
    "VecEnv",
    "StateSnapshot",
]
