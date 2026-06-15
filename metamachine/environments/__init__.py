"""
MetaMachine Environments Module

This module provides simulation and real robot environments for the MetaMachine framework.

Core Classes:
    - MetaMachine: Main CPU-based simulation environment (env_sim.py)
    - MJXMetaMachine: GPU-accelerated MJX simulation environment (env_mjx.py)
    - RealMetaMachine: Real robot environment using capybarish (env_real.py)
    - CyberGearRealMetaMachine: Real robot environment using CyberGearDriver CAN transport
    - RayVecMetaMachine: Vectorized environment using Ray for parallel execution (vec_env.py)
    - VecEnv: Abstract base class for vectorized environments (vec_env.py)

Factory Function:
    - make_env(cfg): Creates the appropriate environment based on config mode

Example:
    >>> from metamachine.environments import make_env
    >>> from metamachine.environments.configs.config_registry import ConfigRegistry
    >>> 
    >>> # Load config with mode: "sim", "mjx", or "real"
    >>> cfg = ConfigRegistry.create_from_file("my_config.yaml")
    >>> 
    >>> # Factory automatically creates correct environment type
    >>> env = make_env(cfg)
    >>> 
    >>> # Or use specific classes directly
    >>> from metamachine.environments import MetaMachine, MJXMetaMachine, RealMetaMachine
    >>> sim_env = MetaMachine(cfg)  # CPU simulation
    >>> mjx_env = MJXMetaMachine(cfg)  # GPU-accelerated simulation
    >>> real_env = RealMetaMachine(cfg)

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>
"""

from .env_sim import MetaMachine
from .base import Base

# Optional import for real robot environment (requires capybarish)
try:
    from .env_real import RealMetaMachine
except ImportError:
    RealMetaMachine = None

try:
    from .env_real_cybergear import CyberGearRealMetaMachine
except ImportError:
    CyberGearRealMetaMachine = None

# Optional import for MJX environment (requires jax and mujoco-mjx)
try:
    from .env_mjx import MJXMetaMachine, MJXState
except ImportError:
    MJXMetaMachine = None
    MJXState = None

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
    - MetaMachine (CPU simulation) if mode == "sim" or not specified
    - MJXMetaMachine (GPU-accelerated simulation) if mode == "mjx"
    - RealMetaMachine (real robot) if mode == "real"
    
    Args:
        cfg: Configuration object (OmegaConf) with environment settings
        **kwargs: Additional arguments passed to the environment constructor
    
    Returns:
        Environment instance (MetaMachine, MJXMetaMachine, or RealMetaMachine)
    
    Raises:
        ValueError: If mode is "real" but capybarish is not installed
        ValueError: If mode is "mjx" but jax/mujoco-mjx is not installed
        ValueError: If mode is unknown
    
    Example:
        >>> from metamachine.environments import make_env
        >>> from metamachine.environments.configs.config_registry import ConfigRegistry
        >>> 
        >>> # For CPU simulation
        >>> cfg = ConfigRegistry.create_from_file("config.yaml")
        >>> cfg.environment.mode = "sim"
        >>> sim_env = make_env(cfg)
        >>> 
        >>> # For GPU-accelerated simulation
        >>> cfg.environment.mode = "mjx"
        >>> mjx_env = make_env(cfg)
        >>> 
        >>> # For real robot
        >>> cfg.environment.mode = "real"
        >>> real_env = make_env(cfg)
    """
    # Get mode from config
    mode = cfg.environment.get("mode", "sim").lower()
    
    if mode == "sim" or mode == "simulation":
        return MetaMachine(cfg, **kwargs)
    
    elif mode == "mjx":
        if MJXMetaMachine is None:
            raise ValueError(
                "MJX mode requires JAX and mujoco-mjx. "
                "Install with: pip install jax jaxlib mujoco-mjx"
            )
        return MJXMetaMachine(cfg, **kwargs)
    
    elif mode == "real":
        real_backend = cfg.get("real", {}).get("backend", "capybarish").lower()
        if real_backend == "cybergear":
            if CyberGearRealMetaMachine is None:
                raise ValueError(
                    "CyberGear real mode requires python-can and CyberGearDriver."
                )
            return CyberGearRealMetaMachine(cfg, **kwargs)
        if RealMetaMachine is None:
            raise ValueError(
                "Real robot mode requires capybarish. "
                "Install it with: pip install capybarish"
            )
        return RealMetaMachine(cfg, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown environment mode: '{mode}'. "
            f"Use 'sim' for CPU simulation, 'mjx' for GPU simulation, "
            f"or 'real' for real robot."
        )


__all__ = [
    # Core environments
    "MetaMachine",
    "MJXMetaMachine",
    "MJXState",
    "RealMetaMachine",
    "CyberGearRealMetaMachine",
    "Base",
    
    # Factory function
    "make_env",
    
    # Vectorized environments
    "RayVecMetaMachine",
    "VecEnv",
    "StateSnapshot",
]
