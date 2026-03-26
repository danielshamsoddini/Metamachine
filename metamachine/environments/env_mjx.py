"""
MJX (JAX-based) MetaMachine Environment for GPU-accelerated parallel simulation.

This module provides the MJXMetaMachine class which enables massively parallel
robot simulation using MuJoCo's MJX backend with JAX for GPU acceleration.

Key Features:
- JAX-based batched simulation for thousands of parallel environments
- Compatible with existing MetaMachine configs and morphology system
- GPU-accelerated physics and reward computation
- Seamless integration with JAX-based RL libraries (brax, rlax, etc.)

Architecture:
- Uses flax.struct.dataclass for immutable state management
- Functional programming style (no in-place mutations)
- Compatible with jax.vmap and jax.jit for vectorization

Usage:
    >>> from metamachine.environments.env_mjx import MJXMetaMachine
    >>> from metamachine.environments.configs.config_registry import ConfigRegistry
    >>> 
    >>> cfg = ConfigRegistry.create_from_name("basic_quadruped")
    >>> env = MJXMetaMachine(cfg)
    >>> 
    >>> # Single environment
    >>> rng = jax.random.PRNGKey(0)
    >>> state = env.reset(rng)
    >>> action = jax.numpy.zeros(env.action_size)
    >>> state = env.step(state, action)
    >>> 
    >>> # Batched environments (1024 parallel envs)
    >>> batch_reset = jax.vmap(env.reset)
    >>> batch_step = jax.vmap(env.step)
    >>> rngs = jax.random.split(rng, 1024)
    >>> states = batch_reset(rngs)

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
"""

from __future__ import annotations

import os

# Set MuJoCo rendering backend for headless environments (must be before mujoco import)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import tempfile
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from flax import struct
from mujoco import mjx
from omegaconf import OmegaConf

# Import robot factory for morphology management
from .. import robot_factory
from ..robot_factory.factory_registry import (
    get_default_draft_model_cfg,
    get_default_fine_model_cfg,
)
from ..utils.math_utils import (
    quaternion_from_vectors,
    quaternion_multiply_alt,
)

# Type aliases
Observation = Union[jax.Array, Mapping[str, jax.Array]]
ObservationSize = Union[int, Mapping[str, Union[Tuple[int, ...], int]]]


@struct.dataclass
class MJXState:
    """Immutable environment state for MJX-based simulation.
    
    This follows the flax.struct pattern for JAX-compatible state management.
    All fields are immutable and state updates create new instances.
    
    Attributes:
        data: MJX simulation data (qpos, qvel, ctrl, etc.)
        obs: Current observation (can be array or dict of arrays)
        reward: Scalar reward from last step
        done: Boolean indicating episode termination
        metrics: Dict of named metrics for logging
        info: Additional info dict (rng, last_act, step count, etc.)
    """
    data: mjx.Data
    obs: Observation
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array]
    info: Dict[str, Any]
    
    def tree_replace(
        self, params: Dict[str, Optional[jax.typing.ArrayLike]]
    ) -> "MJXState":
        """Replace nested attributes using dot notation."""
        new = self
        for k, v in params.items():
            new = _tree_replace(new, k.split("."), v)
        return new


def _tree_replace(
    base: Any,
    attr: Sequence[str],
    val: Optional[jax.typing.ArrayLike],
) -> Any:
    """Sets attributes in a struct.dataclass with values."""
    if not attr:
        return base
    if len(attr) == 1:
        return base.replace(**{attr[0]: val})
    return base.replace(
        **{attr[0]: _tree_replace(getattr(base, attr[0]), attr[1:], val)}
    )


class MJXMetaMachine:
    """MJX-based MetaMachine environment for GPU-accelerated parallel simulation.
    
    This environment uses MuJoCo's MJX backend with JAX for massively parallel
    simulation. It is designed to be compatible with the existing MetaMachine
    configuration system while providing GPU-accelerated physics.
    
    Key differences from CPU MetaMachine:
    - Functional API: reset() and step() are pure functions
    - Immutable state: All state is stored in MJXState dataclass
    - JAX arrays: All data uses jax.numpy arrays
    - Vectorizable: Use jax.vmap for batched simulation
    
    Args:
        cfg: OmegaConf configuration object (same format as MetaMachine)
        
    Example:
        >>> cfg = ConfigRegistry.create_from_name("basic_quadruped")
        >>> env = MJXMetaMachine(cfg)
        >>> 
        >>> # JIT compile for speed
        >>> jit_reset = jax.jit(env.reset)
        >>> jit_step = jax.jit(env.step)
        >>> 
        >>> state = jit_reset(jax.random.PRNGKey(0))
        >>> for _ in range(1000):
        ...     action = policy(state.obs)
        ...     state = jit_step(state, action)
    """
    
    def __init__(self, cfg: OmegaConf) -> None:
        """Initialize MJX environment from config.
        
        Args:
            cfg: Configuration object with same structure as MetaMachine config
        """
        self.cfg = cfg
        self._config = cfg  # Alias for compatibility
        
        # Validate config
        self._validate_config(cfg)
        
        # Setup logging directory (before loading model so we can save debug XML)
        self._setup_logging()
        
        # Setup simulation parameters
        self._setup_simulation_params(cfg)
        
        # Load robot model
        self._load_robot_model(cfg)
        
        # Setup MJX model
        self._setup_mjx_model()
        
        # Setup control parameters
        self._setup_control_params(cfg)
        
        # Setup observation specification
        self._setup_observation_spec(cfg)
        
        # Setup reward specification
        self._setup_reward_spec(cfg)
        
        # Cache joint indices for efficient access
        self._cache_joint_indices()
        
        # Setup domain randomization
        self._setup_domain_randomization(cfg)
        
        # Setup rendering infrastructure
        self._setup_rendering(cfg)
    
    def _validate_config(self, cfg: OmegaConf) -> None:
        """Validate configuration has required fields."""
        required_sections = ["environment", "control", "observation", "task"]
        missing = [s for s in required_sections if s not in cfg]
        if missing:
            raise ValueError(f"Missing required config sections: {missing}")
        
        if not hasattr(cfg, "simulation"):
            raise ValueError("Missing 'simulation' section in config")
    
    def _setup_logging(self) -> None:
        """Setup logging directories and files.
        
        Same logic as Base._setup_logging() for consistency.
        """
        self._log_dir = None
        logging_cfg = self.cfg.get("logging", {})
        
        # Priority 1: Direct log_dir specification
        if logging_cfg.get("log_dir"):
            self._log_dir = logging_cfg.log_dir
            os.makedirs(self._log_dir, exist_ok=True)
        # Priority 2: Create timestamped subdirectory
        elif logging_cfg.get("create_log_dir", False):
            base_log_dir = logging_cfg.get("data_dir", "./logs")
            base_log_dir = "./logs" if base_log_dir is None else base_log_dir
            exp_name = logging_cfg.get("experiment_name", None)
            self._log_dir = self._create_log_directory(base_log_dir, exp_name)
        # Priority 3: Use data_dir directly (if set)
        elif logging_cfg.get("data_dir", None):
            self._log_dir = logging_cfg.get("data_dir", None)
            os.makedirs(self._log_dir, exist_ok=True)
        else:
            self._log_dir = None
        
        # Save config to log directory
        if self._log_dir and os.path.isdir(self._log_dir):
            config_path = os.path.join(self._log_dir, "config.yaml")
            if not os.path.exists(config_path):
                with open(config_path, "w") as f:
                    OmegaConf.save(self.cfg, f)
    
    def _create_log_directory(self, log_dir: str, exp_name: Optional[str] = None) -> str:
        """Create a log directory with timestamp."""
        if exp_name and "-" in exp_name and any(c.isdigit() for c in exp_name):
            dir_name = exp_name
        else:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            robot_name = self.cfg.morphology.get("robot_type", "robot")
            dir_components = [date_str + "_mjx_" + robot_name[0]]
            if exp_name:
                dir_components.append(exp_name)
            dir_name = "_".join(dir_components)
        
        full_path = os.path.join(log_dir, dir_name)
        os.makedirs(full_path, exist_ok=True)
        return full_path
    
    def _setup_rendering(self, cfg: OmegaConf) -> None:
        """Setup rendering infrastructure for unified render API.
        
        Supports render_mode: "none", "mp4", "viewer"
        - "none": No rendering
        - "mp4": Frames are captured during step() and saved as video
        - "viewer": Real-time visualization using MuJoCo passive viewer
        """
        self._render_mode = cfg.simulation.get("render_mode", "none")
        self._render_size = cfg.simulation.get("render_size", [640, 480])
        self._video_path = cfg.simulation.get("video_path", self._log_dir)
        self._video_fps = cfg.simulation.get("video_fps", None)
        if self._video_fps is None:
            self._video_fps = int(1.0 / self._ctrl_dt)
        
        # Video recording state (managed externally for JAX compatibility)
        self._video_frames: List[np.ndarray] = []
        self._recording_active = False
        self._episode_counter = 0
        self._step_counter = 0
        
        # Renderer for mp4 mode (created lazily)
        self._renderer = None
        self._mj_data_for_render = None
        
        # Video recording interval (record every N episodes)
        self._video_record_interval = cfg.simulation.get("video_record_interval", 1)
        
        # Viewer mode state
        self._passive_viewer = None
        self._viewer_context_manager = None
    
    def _setup_simulation_params(self, cfg: OmegaConf) -> None:
        """Setup simulation timing parameters."""
        self.sim_cfg = cfg.simulation
        self._sim_dt = cfg.simulation.mj_dt
        self._ctrl_dt = cfg.control.dt
        
        # Validate timing
        assert self._ctrl_dt % self._sim_dt < 1e-9, \
            f"ctrl_dt ({self._ctrl_dt}) must be multiple of sim_dt ({self._sim_dt})"
        
        self._n_substeps = int(round(self._ctrl_dt / self._sim_dt))
        
        # Episode length
        self._episode_length = cfg.task.termination_conditions.get(
            "max_episode_steps", 1000
        )
    
    def _load_robot_model(self, cfg: OmegaConf) -> None:
        """Load robot model from morphology or asset file."""
        if hasattr(cfg, "morphology") and cfg.morphology.configuration is not None:
            self._load_from_morphology(
                cfg.morphology.robot_type,
                cfg.morphology.configuration
            )
        elif hasattr(cfg, "morphology") and cfg.morphology.asset_file is not None:
            self._load_from_asset(cfg.morphology.asset_file)
        else:
            raise ValueError("No robot asset file or morphology configuration provided")
    
    def _load_from_morphology(self, robot_type: str, morphology: Any) -> None:
        """Load robot from morphology using factory system.
        
        Uses the DRAFT model configuration with primitive collision shapes
        (spheres, capsules) instead of meshes. This is required for MJX
        compatibility since MJX doesn't support all mesh-plane collisions.
        
        Also handles restructuring for graph-based robots (like lego_legs)
        which need to convert flat body + weld constraints to nested hierarchy.
        """
        import uuid
        factory_kwargs = self._get_morphology_factory_kwargs()
        
        factory_init_kwargs = {
            **get_default_draft_model_cfg(robot_type),
            **factory_kwargs,
        }

        # Use DRAFT model config for MJX compatibility
        # Draft models use primitive shapes (SPHERE, CAPSULE) instead of meshes
        factory = robot_factory.get_robot_factory(
            robot_type,
            sim_cfg=self.cfg.simulation,
            **factory_init_kwargs,
        )
        if factory is None:
            raise ValueError(f"Unknown robot factory type: {robot_type}")
        
        # Create robot with draft configuration
        robot = factory.create_robot(morphology=morphology, log_dir=self._log_dir)
        
        # Validate
        is_valid, errors = robot.validate()
        if not is_valid:
            print(f"Warning: Robot validation failed: {errors}")
        
        # Store robot instance
        self._robot_instance = robot
        
        # Check if restructure is enabled (for lego_legs or other graph-based robots)
        restructure = getattr(self.cfg.morphology, "restructure", False)
        restructure_qpos = getattr(self.cfg.morphology, "restructure_qpos", None)
        
        if restructure and hasattr(robot, "save"):
            # Save to temp file with restructuring
            # Use _log_dir if available to avoid race conditions in parallel environments
            if self._log_dir is not None:
                temp_xml_path = os.path.join(self._log_dir, f"robot_temp_{uuid.uuid4().hex[:8]}.xml")
            else:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
                    temp_xml_path = f.name
            
            # Convert restructure_qpos from OmegaConf if needed
            if restructure_qpos is not None:
                from omegaconf import DictConfig, ListConfig
                if isinstance(restructure_qpos, (DictConfig, ListConfig)):
                    restructure_qpos = OmegaConf.to_container(restructure_qpos, resolve=True)
            
            # Save with restructure - converts flat body + weld to nested hierarchy
            saved_path = robot.save(
                temp_xml_path,
                restructure=True,
                restructure_qpos=restructure_qpos,
            )
            
            # Read the restructured XML
            with open(saved_path, 'r') as f:
                self._xml_string = f.read()
            
            print(f"Restructured robot saved to {saved_path}")
        else:
            # Get XML string directly (no restructure)
            self._xml_string = robot.get_xml_string()
        
        # Create MuJoCo model from XML
        self._mj_model = mujoco.MjModel.from_xml_string(self._xml_string)
        
        # Save debug XML to log directory if available
        if self._log_dir:
            debug_xml_path = os.path.join(self._log_dir, "robot.xml")
            with open(debug_xml_path, "w") as f:
                f.write(self._xml_string)
            print(f"Saved robot XML to {debug_xml_path}")

    def _get_morphology_factory_kwargs(self) -> dict[str, Any]:
        """Extract optional factory kwargs from morphology config."""
        morphology_cfg = getattr(self.cfg, "morphology", None)
        if morphology_cfg is None:
            return {}

        if hasattr(morphology_cfg, "get"):
            factory_kwargs = morphology_cfg.get("factory_kwargs", None)
        else:
            factory_kwargs = getattr(morphology_cfg, "factory_kwargs", None)

        if factory_kwargs is None:
            return {}

        if isinstance(factory_kwargs, dict):
            return factory_kwargs.copy()

        try:
            factory_kwargs = OmegaConf.to_container(factory_kwargs, resolve=True)
        except Exception:
            pass

        return dict(factory_kwargs) if isinstance(factory_kwargs, dict) else {}
    
    def _load_from_asset(self, asset_file: str) -> None:
        """Load robot from asset file."""
        xml_path = self._resolve_asset_xml_path(asset_file)
        
        with open(xml_path, 'r') as f:
            self._xml_string = f.read()
        
        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self._robot_instance = None

    def _resolve_asset_xml_path(self, asset_file: str) -> str:
        """Resolve XML path from absolute path, repo-relative path, or built-in asset name."""
        from .. import METAMACHINE_ROOT_DIR

        candidates = []
        if os.path.isabs(asset_file):
            candidates.append(asset_file)
        else:
            repo_root = os.path.dirname(METAMACHINE_ROOT_DIR)
            candidates.extend(
                [
                    asset_file,  # relative to current working directory
                    os.path.join(repo_root, asset_file),  # relative to repo root
                    os.path.join(METAMACHINE_ROOT_DIR, asset_file),  # package-relative
                    os.path.join(METAMACHINE_ROOT_DIR, "assets", "robots", asset_file),  # legacy behavior
                ]
            )

        for path in candidates:
            abs_path = os.path.abspath(os.path.expanduser(path))
            if os.path.exists(abs_path):
                return abs_path

        raise FileNotFoundError(
            f"Could not find XML asset '{asset_file}'. Tried: "
            + ", ".join(os.path.abspath(os.path.expanduser(p)) for p in candidates)
        )
    
    def _setup_mjx_model(self) -> None:
        """Setup MJX model from MuJoCo model."""
        # Set simulation timestep
        self._mj_model.opt.timestep = self._sim_dt
        
        # Increase offscreen framebuffer for high-res rendering
        self._mj_model.vis.global_.offwidth = 1920
        self._mj_model.vis.global_.offheight = 1080
        
        # Convert to MJX model
        self._mjx_model = mjx.put_model(self._mj_model)
        
        # Store reference for initialization
        self._init_qpos = jp.array(self._mj_model.qpos0.copy())
        self._init_qvel = jp.zeros(self._mj_model.nv)
    
    def _setup_control_params(self, cfg: OmegaConf) -> None:
        """Setup control parameters (PD gains, action limits, etc.)."""
        control_cfg = cfg.control
        
        # Number of actions
        if control_cfg.num_actions is None:
            self._num_actions = self._mj_model.nu
        else:
            self._num_actions = control_cfg.num_actions
        
        # PD gains
        kp = control_cfg.get("kp", 8.0)
        kd = control_cfg.get("kd", 0.2)
        self._kp = jp.array(float(kp) if not hasattr(kp, '__len__') else kp)
        self._kd = jp.array(float(kd) if not hasattr(kd, '__len__') else kd)
        
        # Broadcast to per-joint if scalar
        if self._kp.ndim == 0:
            self._kp = jp.full(self._num_actions, float(self._kp))
        if self._kd.ndim == 0:
            self._kd = jp.full(self._num_actions, float(self._kd))
        
        # Action limits
        self._action_scale = float(control_cfg.get("action_scale", 1.0))
        self._symmetric_limit = float(control_cfg.get("symmetric_limit", 0.8))
        
        # Default joint positions - handle OmegaConf ListConfig
        default_dof_pos = control_cfg.get("default_dof_pos", 0)
        
        # Convert OmegaConf types to Python native types
        if hasattr(default_dof_pos, '__iter__') and not isinstance(default_dof_pos, str):
            # It's a list/ListConfig
            self._default_dof_pos = jp.array(list(default_dof_pos), dtype=jp.float32)
        elif default_dof_pos is None:
            self._default_dof_pos = jp.zeros(self._num_actions, dtype=jp.float32)
        else:
            # It's a scalar
            self._default_dof_pos = jp.full(self._num_actions, float(default_dof_pos), dtype=jp.float32)
    
    def _setup_observation_spec(self, cfg: OmegaConf) -> None:
        """Setup observation specification from config.
        
        Supports two modes:
        1. Flat mode: observation.components = [list of component names]
        2. Modular mode: observation.modular.enabled = true with 
           per_module_components and global_components
        """
        obs_cfg = cfg.observation
        
        # Check for modular mode
        modular_cfg = obs_cfg.get("modular", {})
        self._modular_obs_enabled = modular_cfg.get("enabled", False)
        
        if self._modular_obs_enabled:
            self._setup_modular_observation(obs_cfg, modular_cfg)
        else:
            # Flat mode: use components list directly
            self._obs_components = obs_cfg.get("components", [])
            self._modular_components = []
            self._global_components = []
        
        self._include_history_steps = obs_cfg.get("include_history_steps", 1)
        self._clip_obs = obs_cfg.get("clip_observations", 100.0)
        
        # Reference vectors
        forward_vec = obs_cfg.get("forward_vec", [1, 0, 0])
        self._forward_vec = jp.array(forward_vec)
        gravity_vec = obs_cfg.get("gravity_vec", [0, 0, -1])
        self._gravity_vec = jp.array(gravity_vec)
        
        # Calculate observation size
        self._calculate_obs_size()
    
    def _setup_modular_observation(self, obs_cfg, modular_cfg) -> None:
        """Setup modular observation mode (per-module + global components)."""
        self._num_modules = modular_cfg.get("num_modules", self._num_actions)
        if self._num_modules == "auto":
            self._num_modules = self._num_actions
        
        # Main module index: if set, only use that module's data
        self._main_module_index = modular_cfg.get("main_module_index", None)
        
        # Per-module components (projected_gravities, gyros, etc.)
        self._modular_components = modular_cfg.get("per_module_components", [])
        
        # Global components (dof_pos, dof_vel, last_action, etc.)
        self._global_components = modular_cfg.get("global_components", [])
        
        # Build flat component list for compatibility
        # If main_module_index is set, we only use one module's data
        self._obs_components = []
        
        # Per-module components (with _modular suffix for identification)
        for comp in self._modular_components:
            name = comp if isinstance(comp, str) else comp.get("name", comp)
            self._obs_components.append({"name": name, "type": "modular"})
        
        # Global components
        for comp in self._global_components:
            name = comp if isinstance(comp, str) else comp.get("name", comp)
            self._obs_components.append({"name": name, "type": "global"})
    
    def _calculate_obs_size(self) -> None:
        """Calculate total observation size from components."""
        size = 0
        for comp in self._obs_components:
            name = comp if isinstance(comp, str) else comp.get("name", comp)
            comp_type = comp.get("type", "global") if isinstance(comp, dict) else "global"
            size += self._get_component_size(name, comp_type)
        
        self._single_obs_size = size
        self._total_obs_size = size * self._include_history_steps
    
    def _get_component_size(self, name: str, comp_type: str = "global") -> int:
        """Get size of observation component.
        
        Args:
            name: Component name
            comp_type: "modular" or "global"
        """
        # Base sizes for global/flat components
        component_sizes = {
            "projected_gravity": 3,
            "ang_vel_body": 3,
            "dof_pos": self._num_actions,
            "dof_vel": self._num_actions,
            "last_action": self._num_actions,
            "vel_body": 3,
            "commands": self._get_command_size(),
        }
        
        # Modular components (per-module data)
        # If main_module_index is set, we only use one module (size = element_size)
        # Otherwise, all modules (size = num_modules * element_size)
        modular_element_sizes = {
            "projected_gravities": 3,
            "gyros": 3,
            "accs": 3,
            "quats": 4,
            "vels": 3,
        }
        
        if comp_type == "modular" and name in modular_element_sizes:
            element_size = modular_element_sizes[name]
            if self._main_module_index is not None:
                return element_size  # Only one module
            else:
                return element_size * self._num_modules  # All modules
        
        return component_sizes.get(name, 0)
    
    def _get_command_size(self) -> int:
        """Get size of command vector."""
        if hasattr(self.cfg, "task") and hasattr(self.cfg.task, "commands"):
            dims = self.cfg.task.commands.get("dimensions", {})
            return len(dims)
        return 0
    
    def _setup_reward_spec(self, cfg: OmegaConf) -> None:
        """Setup reward specification from config."""
        task_cfg = cfg.task
        
        # Get reward components
        self._reward_components = task_cfg.get("reward_components", [])
        
        # Build reward weights dict
        self._reward_weights = {}
        for comp in self._reward_components:
            name = comp.get("name", "unnamed")
            weight = comp.get("weight", 1.0)
            self._reward_weights[name] = weight
    
    def _cache_joint_indices(self) -> None:
        """Cache joint indices for efficient access."""
        # Get joint IDs from actuator transmission
        self._joint_ids = []
        for i in range(self._mj_model.nu):
            joint_id = self._mj_model.actuator_trnid[i, 0]
            self._joint_ids.append(joint_id)
        
        self._joint_ids = np.array(self._joint_ids)
        
        # Get qpos and qvel addresses for joints
        self._jnt_qposadr = jp.array(
            [self._mj_model.jnt_qposadr[j] for j in self._joint_ids]
        )
        self._jnt_dofadr = jp.array(
            [self._mj_model.jnt_dofadr[j] for j in self._joint_ids]
        )
        
        # Find torso body (for observation extraction)
        torso_node_id = self.cfg.observation.get("torso_node_id", 0)
        try:
            self._torso_body_id = self._mj_model.body(f"l{torso_node_id}").id
        except KeyError:
            self._torso_body_id = 1  # Default to first non-world body
    
    def _setup_domain_randomization(self, cfg: OmegaConf) -> None:
        """Setup domain randomization configuration.
        
        This method parses the randomization config and prepares parameters
        for runtime randomization. Unlike CPU MuJoCo which can reload XML,
        MJX randomizes by directly modifying model parameters via tree_replace.
        
        Supported randomization (from cfg.randomization):
        - friction: Floor and contact friction
        - mass: Body masses (scaling)
        - damping: Joint damping
        - armature: Joint armature
        - frictionloss: Joint friction loss (static friction)
        - pd_controller: PD gains (kp, kd)
        - init_joint_pos: Initial joint positions
        - init_base_pos: Initial base position
        - init_base_vel: Initial base velocity
        """
        self._randomization_cfg = cfg.get("randomization", {})
        
        # Store whether any randomization is enabled
        self._domain_randomization_enabled = any([
            self._randomization_cfg.get("friction", {}).get("enabled", False),
            self._randomization_cfg.get("mass", {}).get("enabled", False),
            self._randomization_cfg.get("damping", {}).get("enabled", False),
            self._randomization_cfg.get("armature", {}).get("enabled", False),
            self._randomization_cfg.get("frictionloss", {}).get("enabled", False),
            self._randomization_cfg.get("pd_controller", {}).get("enabled", False),
            self._randomization_cfg.get("init_joint_pos", {}).get("enabled", False),
            self._randomization_cfg.get("com_offset", {}).get("enabled", False),
        ])
        
        # Cache original model parameters for randomization reference
        if self._domain_randomization_enabled:
            self._original_geom_friction = jp.array(self._mjx_model.geom_friction.copy())
            self._original_body_mass = jp.array(self._mjx_model.body_mass.copy())
            self._original_dof_damping = jp.array(self._mjx_model.dof_damping.copy())
            self._original_dof_armature = jp.array(self._mjx_model.dof_armature.copy())
            self._original_dof_frictionloss = jp.array(self._mjx_model.dof_frictionloss.copy())
            self._original_body_ipos = jp.array(self._mjx_model.body_ipos.copy())
            self._original_qpos0 = jp.array(self._mjx_model.qpos0.copy())
            
            # Store original PD gains
            self._original_kp = self._kp.copy()
            self._original_kd = self._kd.copy()
            
            # Find floor geom ID (usually 0, but let's be safe)
            self._floor_geom_id = 0
            try:
                self._floor_geom_id = self._mj_model.geom("floor").id
            except KeyError:
                pass  # Use default 0
            
            print(f"[MJX] Domain randomization enabled")
    
    def _apply_domain_randomization(self, rng: jax.Array) -> Tuple[mjx.Model, jax.Array, jax.Array]:
        """Apply domain randomization to the MJX model.
        
        This modifies model parameters directly (friction, mass, damping, etc.)
        following the pattern from mujoco_playground's domain randomization.
        
        Args:
            rng: JAX random key
            
        Returns:
            Tuple of (randomized_model, randomized_kp, randomized_kd)
        """
        if not self._domain_randomization_enabled:
            return self._mjx_model, self._kp, self._kd
        
        model = self._mjx_model
        kp = self._original_kp.copy()
        kd = self._original_kd.copy()
        
        rand_cfg = self._randomization_cfg
        
        # Friction randomization
        friction_cfg = rand_cfg.get("friction", {})
        if friction_cfg.get("enabled", False):
            rng, key = jax.random.split(rng)
            friction_range = friction_cfg.get("range", [0.4, 1.2])
            new_friction = jax.random.uniform(
                key, minval=friction_range[0], maxval=friction_range[1]
            )
            geom_friction = model.geom_friction.at[self._floor_geom_id, 0].set(new_friction)
            model = model.tree_replace({"geom_friction": geom_friction})
        
        # Mass randomization (scale all body masses)
        mass_cfg = rand_cfg.get("mass", {})
        if mass_cfg.get("enabled", False):
            rng, key = jax.random.split(rng)
            mass_scale_range = mass_cfg.get("scale_range", [0.9, 1.1])
            mass_scale = jax.random.uniform(
                key, shape=(model.nbody,),
                minval=mass_scale_range[0], maxval=mass_scale_range[1]
            )
            body_mass = self._original_body_mass * mass_scale
            model = model.tree_replace({"body_mass": body_mass})
        
        # Damping randomization
        damping_cfg = rand_cfg.get("damping", {})
        if damping_cfg.get("enabled", False):
            rng, key = jax.random.split(rng)
            damping_scale_range = damping_cfg.get("scale_range", [0.8, 1.2])
            damping_scale = jax.random.uniform(
                key, shape=(self._num_actions,),
                minval=damping_scale_range[0], maxval=damping_scale_range[1]
            )
            # Only modify actuated DOFs (skip free joint DOFs)
            dof_damping = model.dof_damping.at[6:6+self._num_actions].set(
                self._original_dof_damping[6:6+self._num_actions] * damping_scale
            )
            model = model.tree_replace({"dof_damping": dof_damping})
        
        # Armature randomization
        armature_cfg = rand_cfg.get("armature", {})
        if armature_cfg.get("enabled", False):
            rng, key = jax.random.split(rng)
            armature_scale_range = armature_cfg.get("scale_range", [1.0, 1.5])
            armature_scale = jax.random.uniform(
                key, shape=(self._num_actions,),
                minval=armature_scale_range[0], maxval=armature_scale_range[1]
            )
            dof_armature = model.dof_armature.at[6:6+self._num_actions].set(
                self._original_dof_armature[6:6+self._num_actions] * armature_scale
            )
            model = model.tree_replace({"dof_armature": dof_armature})
        
        # Friction loss (static friction) randomization
        frictionloss_cfg = rand_cfg.get("frictionloss", {})
        if frictionloss_cfg.get("enabled", False):
            rng, key = jax.random.split(rng)
            frictionloss_range = frictionloss_cfg.get("range", [0.0, 0.1])
            frictionloss = jax.random.uniform(
                key, shape=(self._num_actions,),
                minval=frictionloss_range[0], maxval=frictionloss_range[1]
            )
            dof_frictionloss = model.dof_frictionloss.at[6:6+self._num_actions].set(frictionloss)
            model = model.tree_replace({"dof_frictionloss": dof_frictionloss})
        
        # Center of mass offset randomization
        com_cfg = rand_cfg.get("com_offset", {})
        if com_cfg.get("enabled", False):
            rng, key = jax.random.split(rng)
            com_range = com_cfg.get("range", [-0.05, 0.05])
            dpos = jax.random.uniform(key, (3,), minval=com_range[0], maxval=com_range[1])
            body_ipos = model.body_ipos.at[self._torso_body_id].set(
                self._original_body_ipos[self._torso_body_id] + dpos
            )
            model = model.tree_replace({"body_ipos": body_ipos})
        
        # PD controller gain randomization
        pd_cfg = rand_cfg.get("pd_controller", {})
        if pd_cfg.get("enabled", False):
            rng, key1, key2 = jax.random.split(rng, 3)
            kp_range = pd_cfg.get("kp_range", [6.0, 10.0])
            kd_range = pd_cfg.get("kd_range", [0.15, 0.3])
            
            # Randomize as scale or absolute value
            if pd_cfg.get("scale_mode", False):
                kp_scale = jax.random.uniform(key1, minval=kp_range[0], maxval=kp_range[1])
                kd_scale = jax.random.uniform(key2, minval=kd_range[0], maxval=kd_range[1])
                kp = self._original_kp * kp_scale
                kd = self._original_kd * kd_scale
            else:
                kp_val = jax.random.uniform(key1, minval=kp_range[0], maxval=kp_range[1])
                kd_val = jax.random.uniform(key2, minval=kd_range[0], maxval=kd_range[1])
                kp = jp.full_like(self._original_kp, kp_val)
                kd = jp.full_like(self._original_kd, kd_val)
        
        return model, kp, kd
    
    def _get_randomized_initial_qpos(self, rng: jax.Array, qpos: jax.Array) -> jax.Array:
        """Apply randomization to initial joint positions.
        
        Args:
            rng: JAX random key
            qpos: Base qpos array to randomize
            
        Returns:
            Randomized qpos
        """
        rand_cfg = self._randomization_cfg
        
        # Joint position randomization
        joint_cfg = rand_cfg.get("init_joint_pos", {})
        if joint_cfg.get("enabled", False):
            rng, key = jax.random.split(rng)
            noise_range = joint_cfg.get("range", [-0.2, 0.2])
            noise = jax.random.uniform(
                key, shape=(self._num_actions,),
                minval=noise_range[0], maxval=noise_range[1]
            )
            qpos = qpos.at[self._jnt_qposadr].add(noise)
        
        # Base position randomization
        base_pos_cfg = rand_cfg.get("init_base_pos", {})
        if base_pos_cfg.get("enabled", False):
            rng, key = jax.random.split(rng)
            xy_range = base_pos_cfg.get("xy_range", [-0.5, 0.5])
            z_range = base_pos_cfg.get("z_range", [0.0, 0.1])
            xy_noise = jax.random.uniform(key, (2,), minval=xy_range[0], maxval=xy_range[1])
            rng, key = jax.random.split(rng)
            z_noise = jax.random.uniform(key, minval=z_range[0], maxval=z_range[1])
            qpos = qpos.at[:2].add(xy_noise)
            qpos = qpos.at[2].add(z_noise)
        
        return qpos
    
    def _get_randomized_initial_qvel(self, rng: jax.Array, qvel: jax.Array) -> jax.Array:
        """Apply randomization to initial velocities.
        
        Args:
            rng: JAX random key
            qvel: Base qvel array to randomize
            
        Returns:
            Randomized qvel
        """
        rand_cfg = self._randomization_cfg
        
        # Base velocity randomization
        base_vel_cfg = rand_cfg.get("init_base_vel", {})
        if base_vel_cfg.get("enabled", False):
            rng, key = jax.random.split(rng)
            lin_range = base_vel_cfg.get("linear_range", [-0.5, 0.5])
            ang_range = base_vel_cfg.get("angular_range", [-0.5, 0.5])
            
            lin_noise = jax.random.uniform(key, (3,), minval=lin_range[0], maxval=lin_range[1])
            rng, key = jax.random.split(rng)
            ang_noise = jax.random.uniform(key, (3,), minval=ang_range[0], maxval=ang_range[1])
            
            qvel = qvel.at[:3].add(lin_noise)
            qvel = qvel.at[3:6].add(ang_noise)
        
        # Joint velocity randomization
        joint_vel_cfg = rand_cfg.get("init_joint_vel", {})
        if joint_vel_cfg.get("enabled", False):
            rng, key = jax.random.split(rng)
            vel_range = joint_vel_cfg.get("range", [-0.5, 0.5])
            vel_noise = jax.random.uniform(
                key, shape=(self._num_actions,),
                minval=vel_range[0], maxval=vel_range[1]
            )
            qvel = qvel.at[self._jnt_dofadr].add(vel_noise)
        
        return qvel
    
    # =========================================================================
    # Core Environment Interface
    # =========================================================================
    
    def reset(self, rng: jax.Array) -> MJXState:
        """Reset environment to initial state.
        
        Args:
            rng: JAX random key for stochastic initialization
            
        Returns:
            MJXState: Initial environment state
        """
        rng, init_rng, noise_rng, rand_rng = jax.random.split(rng, 4)
        
        # Apply domain randomization to model and gains
        randomized_model, randomized_kp, randomized_kd = self._apply_domain_randomization(rand_rng)
        
        # Get initial qpos/qvel with optional randomization
        qpos = self._get_initial_qpos(init_rng)
        qvel = self._get_initial_qvel(init_rng)
        
        # Apply additional initial state randomization
        rng, qpos_rng, qvel_rng = jax.random.split(rng, 3)
        qpos = self._get_randomized_initial_qpos(qpos_rng, qpos)
        qvel = self._get_randomized_initial_qvel(qvel_rng, qvel)
        
        # Create MJX data with randomized model
        data = self._init_mjx_data_with_model(randomized_model, qpos, qvel)
        
        # Initialize info dict with randomized parameters
        info = {
            "step": jp.array(0),
            "rng": rng,
            "last_act": jp.zeros(self._num_actions),
            "last_last_act": jp.zeros(self._num_actions),
            "commands": self._sample_commands(rng),
            # Store randomized parameters for use in step()
            "kp": randomized_kp,
            "kd": randomized_kd,
            "model": randomized_model,  # Store randomized model for step()
        }
        
        # Initialize metrics
        metrics = {}
        for comp in self._reward_components:
            name = comp.get("name", "unnamed")
            metrics[f"reward/{name}"] = jp.zeros(())
        
        # Initialize observation history
        obs_history = jp.zeros(self._total_obs_size)
        
        # Get initial observation
        obs = self._get_obs(data, info, obs_history, noise_rng)
        
        reward, done = jp.zeros(2)
        
        return MJXState(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )
    
    def step(self, state: MJXState, action: jax.Array) -> MJXState:
        """Execute one environment step.
        
        Args:
            state: Current environment state
            action: Action to execute (in [-symmetric_limit, symmetric_limit])
            
        Returns:
            MJXState: New environment state
        """
        rng, noise_rng = jax.random.split(state.info["rng"], 2)
        
        # Process action
        action = jp.clip(action, -self._symmetric_limit, self._symmetric_limit)
        motor_targets = self._default_dof_pos + action * self._action_scale
        
        # Get randomized parameters from state info (or use defaults)
        kp = state.info.get("kp", self._kp)
        kd = state.info.get("kd", self._kd)
        model = state.info.get("model", self._mjx_model)
        
        # Step physics with PD control using randomized parameters
        data = self._step_pd_with_model(
            model,
            state.data,
            motor_targets,
            kp,
            kd,
            self._n_substeps,
        )
        
        # Get observation with history
        obs_history = state.obs["state"] if isinstance(state.obs, dict) else state.obs
        obs = self._get_obs(data, state.info, obs_history, noise_rng)
        
        # Check termination
        done = self._get_termination(data, state.info)
        
        # Calculate rewards
        rewards = self._get_rewards(data, action, state.info, done)
        total_reward = sum(
            v * self._reward_weights.get(k, 1.0) 
            for k, v in rewards.items()
        )
        reward = jp.clip(total_reward * self._ctrl_dt, -10.0, 10.0)
        
        # Update info (preserve randomized parameters across steps)
        new_info = {
            "step": state.info["step"] + 1,
            "rng": rng,
            "last_act": action,
            "last_last_act": state.info["last_act"],
            "commands": state.info["commands"],
            # Preserve randomized parameters for the episode
            "kp": kp,
            "kd": kd,
            "model": model,
        }
        
        # Update metrics
        metrics = {}
        for k, v in rewards.items():
            metrics[f"reward/{k}"] = v
        
        return MJXState(
            data=data,
            obs=obs,
            reward=reward,
            done=done.astype(reward.dtype),
            metrics=metrics,
            info=new_info,
        )
    
    # =========================================================================
    # Physics Simulation
    # =========================================================================
    
    def _init_mjx_data(
        self,
        qpos: jax.Array,
        qvel: jax.Array,
    ) -> mjx.Data:
        """Initialize MJX data with given state (using default model)."""
        return self._init_mjx_data_with_model(self._mjx_model, qpos, qvel)
    
    def _init_mjx_data_with_model(
        self,
        model: mjx.Model,
        qpos: jax.Array,
        qvel: jax.Array,
    ) -> mjx.Data:
        """Initialize MJX data with given state and specific model.
        
        This method is used for domain randomization where each environment
        may have a different randomized model.
        
        Args:
            model: MJX model (possibly randomized)
            qpos: Initial joint positions
            qvel: Initial joint velocities
            
        Returns:
            Initialized MJX data
        """
        data = mjx.make_data(model)
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(model, data)
        return data
    
    def _step_pd(
        self,
        data: mjx.Data,
        motor_targets: jax.Array,
        kp: jax.Array,
        kd: jax.Array,
        n_substeps: int,
    ) -> mjx.Data:
        """Step physics with PD control (using default model)."""
        return self._step_pd_with_model(
            self._mjx_model, data, motor_targets, kp, kd, n_substeps
        )
    
    def _step_pd_with_model(
        self,
        model: mjx.Model,
        data: mjx.Data,
        motor_targets: jax.Array,
        kp: jax.Array,
        kd: jax.Array,
        n_substeps: int,
    ) -> mjx.Data:
        """Step physics with PD control using specific model (Cybergear motor model).
        
        Uses the Cybergear torque-velocity characteristic curve for realistic
        motor simulation. This version accepts a model parameter for domain
        randomization support.
        
        Args:
            model: MJX model (possibly randomized)
            data: Current simulation data
            motor_targets: Target joint positions
            kp: Proportional gains (possibly randomized)
            kd: Derivative gains (possibly randomized)
            n_substeps: Number of simulation substeps
            
        Returns:
            Updated simulation data
        """
        # Get current motor state
        motor_pos = data.qpos[self._jnt_qposadr]
        motor_vel = data.qvel[self._jnt_dofadr]
        
        # Compute raw torque from PD control
        raw_torque = kp * (motor_targets - motor_pos) - kd * motor_vel
        
        # Apply Cybergear torque-velocity limits
        # Piecewise: constant 12 Nm below 11.5 rad/s, then linear decay
        vel_abs = jp.abs(motor_vel)
        const_limit = 12.0
        threshold = 11.5
        decay_slope = -0.656
        decay_offset = 19.541
        
        linear_part = decay_slope * vel_abs + decay_offset
        decayed_limit = jp.clip(linear_part, a_min=0.0)
        torque_limit = jp.where(vel_abs < threshold, const_limit, decayed_limit)
        torque = jp.clip(raw_torque, -torque_limit, torque_limit)
        
        # Step simulation with the provided model
        def single_step(data, _):
            data = data.replace(ctrl=torque)
            data = mjx.step(model, data)
            return data, None
        
        data, _ = jax.lax.scan(single_step, data, (), n_substeps)
        return data
    
    # =========================================================================
    # Observation Computation
    # =========================================================================
    
    def _get_obs(
        self,
        data: mjx.Data,
        info: Dict[str, Any],
        obs_history: jax.Array,
        noise_rng: jax.Array,
    ) -> Union[jax.Array, Dict[str, jax.Array]]:
        """Compute observation from simulation data.
        
        Supports both flat and modular observation modes.
        
        Args:
            data: MJX simulation data
            info: Info dict with last_act, commands, etc.
            obs_history: Previous observation history buffer
            noise_rng: Random key for observation noise
            
        Returns:
            Observation (dict with 'state' and 'privileged_state' keys)
        """
        # Extract observation components
        obs_parts = []
        
        for comp in self._obs_components:
            name = comp if isinstance(comp, str) else comp.get("name", comp)
            comp_type = comp.get("type", "global") if isinstance(comp, dict) else "global"
            transform = comp.get("transform", None) if isinstance(comp, dict) else None
            
            value = self._get_obs_component(data, info, name, comp_type, noise_rng)
            
            # Apply transform
            if transform == "cos":
                value = jp.cos(value)
            elif transform == "sin":
                value = jp.sin(value)
            
            if value.size > 0:  # Only add non-empty arrays
                obs_parts.append(value.flatten())
        
        # Handle empty observation case
        if not obs_parts:
            current_obs = jp.zeros(self._single_obs_size)
        else:
            # Concatenate current observation
            current_obs = jp.concatenate(obs_parts)
        
        # Update history buffer (roll and append)
        if self._include_history_steps > 1:
            obs = jp.roll(obs_history, -self._single_obs_size)
            obs = obs.at[-self._single_obs_size:].set(current_obs)
        else:
            obs = current_obs
        
        # Clip to prevent extreme values
        obs = jp.clip(obs, -self._clip_obs, self._clip_obs)
        
        # Return flat array (compatible with SB3) instead of dict
        # For training with brax-style pipelines that need privileged info,
        # subclass and override to return {"state": obs, "privileged_state": ...}
        return obs
    
    def _get_obs_component(
        self,
        data: mjx.Data,
        info: Dict[str, Any],
        name: str,
        comp_type: str,
        noise_rng: jax.Array,
    ) -> jax.Array:
        """Get single observation component.
        
        Args:
            data: MJX simulation data
            info: Info dict
            name: Component name
            comp_type: "modular" or "global"
            noise_rng: Random key for noise
            
        Returns:
            Observation component as JAX array
        """
        # Global/flat components
        if name == "projected_gravity":
            return self._get_projected_gravity(data)
        elif name == "ang_vel_body":
            return self._get_gyro(data)
        elif name == "dof_pos":
            return data.qpos[self._jnt_qposadr] - self._default_dof_pos
        elif name == "dof_vel":
            return data.qvel[self._jnt_dofadr]
        elif name == "last_action":
            return info["last_act"]
        elif name == "vel_body":
            return self._get_local_linvel(data)
        elif name == "commands":
            return info.get("commands", jp.zeros(self._get_command_size()))
        
        # Modular components (per-module data)
        elif name == "projected_gravities":
            return self._get_modular_projected_gravities(data)
        elif name == "gyros":
            return self._get_modular_gyros(data)
        elif name == "accs":
            return self._get_modular_accelerations(data)
        elif name == "quats":
            return self._get_modular_quaternions(data)
        elif name == "vels":
            return self._get_modular_velocities(data)
        
        else:
            # Unknown component - return zeros with correct size
            size = self._get_component_size(name, comp_type)
            return jp.zeros(size)
    
    def _get_modular_projected_gravities(self, data: mjx.Data) -> jax.Array:
        """Get projected gravity for each module (or main module only)."""
        # For now, use torso gravity for all modules (simplified)
        # TODO: Per-module IMU support would require sensor setup
        gravity = self._get_projected_gravity(data)
        
        if self._main_module_index is not None:
            # Return only main module's gravity (same as torso for now)
            return gravity
        else:
            # Return for all modules (tiled)
            return jp.tile(gravity, self._num_modules)
    
    def _get_modular_gyros(self, data: mjx.Data) -> jax.Array:
        """Get angular velocity for each module (or main module only)."""
        # Simplified: use torso gyro for all modules
        gyro = self._get_gyro(data)
        
        if self._main_module_index is not None:
            return gyro
        else:
            return jp.tile(gyro, self._num_modules)
    
    def _get_modular_accelerations(self, data: mjx.Data) -> jax.Array:
        """Get accelerations for each module."""
        # Simplified: return zeros (would need accelerometer sensors)
        if self._main_module_index is not None:
            return jp.zeros(3)
        else:
            return jp.zeros(3 * self._num_modules)
    
    def _get_modular_quaternions(self, data: mjx.Data) -> jax.Array:
        """Get quaternions for each module."""
        # Simplified: use torso quaternion for all modules
        quat = data.xquat[self._torso_body_id]
        
        if self._main_module_index is not None:
            return quat
        else:
            return jp.tile(quat, self._num_modules)
    
    def _get_modular_velocities(self, data: mjx.Data) -> jax.Array:
        """Get velocities for each module."""
        vel = self._get_local_linvel(data)
        
        if self._main_module_index is not None:
            return vel
        else:
            return jp.tile(vel, self._num_modules)
    
    def _get_projected_gravity(self, data: mjx.Data) -> jax.Array:
        """Get gravity vector projected into body frame."""
        # Get body rotation matrix
        body_xmat = data.xmat[self._torso_body_id].reshape(3, 3)
        # Project world gravity into body frame
        return body_xmat.T @ jp.array([0.0, 0.0, -1.0])
    
    def _get_gyro(self, data: mjx.Data) -> jax.Array:
        """Get angular velocity in body frame."""
        return data.cvel[self._torso_body_id][:3]
    
    def _get_local_linvel(self, data: mjx.Data) -> jax.Array:
        """Get linear velocity in body frame."""
        return data.cvel[self._torso_body_id][3:]
    
    # =========================================================================
    # Reward Computation
    # =========================================================================
    
    def _get_rewards(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: Dict[str, Any],
        done: jax.Array,
    ) -> Dict[str, jax.Array]:
        """Compute all reward components."""
        rewards = {}
        
        for comp in self._reward_components:
            name = comp.get("name", "unnamed")
            reward_type = comp.get("type", "custom")
            params = comp.get("params", {})
            
            reward = self._compute_reward_component(
                data, action, info, reward_type, params
            )
            rewards[name] = reward
        
        return rewards
    
    def _compute_reward_component(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: Dict[str, Any],
        reward_type: str,
        params: Dict[str, Any],
    ) -> jax.Array:
        """Compute single reward component."""
        if reward_type == "linear_velocity_tracking":
            return self._reward_linear_velocity_tracking(data, info, params)
        elif reward_type == "angular_velocity_tracking":
            return self._reward_angular_velocity_tracking(data, info, params)
        elif reward_type == "action_rate":
            return self._reward_action_rate(action, info)
        elif reward_type in ("action_rate_rate", "action_acceleration"):
            return self._reward_action_rate_rate(action, info)
        elif reward_type == "upright":
            return self._reward_upright(data)
        else:
            return jp.zeros(())
    
    def _reward_linear_velocity_tracking(
        self,
        data: mjx.Data,
        info: Dict[str, Any],
        params: Dict[str, Any],
    ) -> jax.Array:
        """Reward for tracking target linear velocity."""
        target_vel = params.get("target_velocity", 0.6)
        sigma = params.get("tracking_sigma", 0.25)
        
        # Get current forward velocity (in body x direction)
        vel_body = self._get_local_linvel(data)
        forward_vel = vel_body[0]  # Assume x is forward
        
        # Exponential tracking reward
        error = jp.square(target_vel - forward_vel)
        return jp.exp(-error / sigma)
    
    def _reward_angular_velocity_tracking(
        self,
        data: mjx.Data,
        info: Dict[str, Any],
        params: Dict[str, Any],
    ) -> jax.Array:
        """Reward for tracking target angular velocity (yaw rate)."""
        target_ang_vel = params.get("target_angular_velocity", 0.0)
        sigma = params.get("tracking_sigma", 0.25)
        
        # Get yaw rate (angular velocity around z-axis)
        ang_vel = self._get_gyro(data)
        yaw_rate = ang_vel[2]  # Assume z is up
        
        error = jp.square(target_ang_vel - yaw_rate)
        return jp.exp(-error / sigma)
    
    def _reward_action_rate(
        self,
        action: jax.Array,
        info: Dict[str, Any],
    ) -> jax.Array:
        """Penalty for action rate (smoothness)."""
        last_action = info["last_act"]
        return -jp.sum(jp.square(action - last_action))

    def _reward_action_rate_rate(
        self,
        action: jax.Array,
        info: Dict[str, Any],
    ) -> jax.Array:
        """Penalty for changes in action rate (second-order smoothness)."""
        last_action = info["last_act"]
        last_last_action = info["last_last_act"]
        second_diff = action - 2.0 * last_action + last_last_action
        return -jp.sum(jp.square(second_diff))
    
    def _reward_upright(self, data: mjx.Data) -> jax.Array:
        """Reward for staying upright."""
        projected_gravity = self._get_projected_gravity(data)
        # Reward is higher when gravity points straight down in body frame
        return projected_gravity[2]  # Should be close to -1 when upright
    
    # =========================================================================
    # Termination
    # =========================================================================
    
    def _get_termination(
        self,
        data: mjx.Data,
        info: Dict[str, Any],
    ) -> jax.Array:
        """Check termination conditions."""
        # Fall termination (upside down)
        projected_gravity = self._get_projected_gravity(data)
        fall_termination = projected_gravity[2] > 0  # Upside down
        
        # Step limit termination
        step_termination = info["step"] >= self._episode_length
        
        return fall_termination | step_termination
    
    # =========================================================================
    # Initialization Helpers
    # =========================================================================
    
    def _get_initial_qpos(self, rng: jax.Array) -> jax.Array:
        """Get initial qpos with optional randomization."""
        qpos = self._init_qpos.copy()
        
        # Apply config-specified initial position if available
        init_cfg = self.cfg.get("initialization", {})
        
        # Helper to convert OmegaConf ListConfig to regular Python list
        def to_list(val):
            if val is None:
                return None
            # Handle OmegaConf ListConfig
            if hasattr(val, '_iter_ex') or hasattr(val, '_content'):
                return OmegaConf.to_container(val) if OmegaConf.is_config(val) else list(val)
            return val
        
        # Initial position
        init_pos = to_list(init_cfg.get("init_pos", None))
        if init_pos is not None:
            qpos = qpos.at[:3].set(jp.array(init_pos))
        
        # Initial quaternion - MuJoCo uses WXYZ format [w, x, y, z]
        # Config should provide quaternion in WXYZ format for MuJoCo compatibility
        init_quat = to_list(init_cfg.get("init_quat", None))
        if init_quat is not None and len(init_quat) == 4:
            qpos = qpos.at[3:7].set(jp.array(init_quat))
        
        # Initial joint positions
        init_joint_pos = to_list(init_cfg.get("init_joint_pos", None))
        if init_joint_pos is None:
            init_joint_pos = self._default_dof_pos
        
        if init_joint_pos is not None:
            init_joint_pos = jp.array(init_joint_pos) if not isinstance(
                init_joint_pos, jax.Array
            ) else init_joint_pos
            qpos = qpos.at[self._jnt_qposadr].set(init_joint_pos)
        
        # Add noise if configured
        if init_cfg.get("noisy_init", False):
            rng, noise_rng = jax.random.split(rng)
            noise = jax.random.uniform(
                noise_rng, shape=qpos.shape, minval=-0.1, maxval=0.1
            )
            qpos = qpos + noise
        
        return qpos
    
    def _get_initial_qvel(self, rng: jax.Array) -> jax.Array:
        """Get initial qvel with optional randomization."""
        qvel = self._init_qvel.copy()
        
        init_cfg = self.cfg.get("initialization", {})
        
        if init_cfg.get("randomize_ini_vel", False):
            rng, vel_rng = jax.random.split(rng)
            # Randomize base velocity
            vel_noise = jax.random.uniform(
                vel_rng, shape=(6,), minval=-1.0, maxval=1.0
            )
            qvel = qvel.at[:6].set(vel_noise)
        
        return qvel
    
    def _sample_commands(self, rng: jax.Array) -> jax.Array:
        """Sample command vector from config."""
        if not hasattr(self.cfg, "task") or not hasattr(self.cfg.task, "commands"):
            return jp.zeros(0)
        
        cmd_cfg = self.cfg.task.commands
        dimensions = cmd_cfg.get("dimensions", {})
        
        commands = []
        for name, spec in dimensions.items():
            rng, sample_rng = jax.random.split(rng)
            
            cmd_type = spec.get("type", "uniform")
            cmd_range = spec.get("range", [0, 1])
            initial_value = spec.get("initial_value", None)
            
            if initial_value is not None:
                value = jp.array(initial_value)
            elif cmd_type == "uniform":
                value = jax.random.uniform(
                    sample_rng, minval=cmd_range[0], maxval=cmd_range[1]
                )
            else:
                value = jp.array(0.0)
            
            commands.append(value)
        
        return jp.array(commands) if commands else jp.zeros(0)
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def action_size(self) -> int:
        """Size of action space."""
        return self._num_actions
    
    @property
    def observation_size(self) -> ObservationSize:
        """Size of observation space."""
        # Use abstract state to determine observation shape
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        obs = abstract_state.obs
        if isinstance(obs, Mapping):
            return jax.tree_util.tree_map(lambda x: x.shape, obs)
        return obs.shape[-1]
    
    @property
    def dt(self) -> float:
        """Control timestep."""
        return self._ctrl_dt
    
    @property
    def sim_dt(self) -> float:
        """Simulation timestep."""
        return self._sim_dt
    
    @property
    def n_substeps(self) -> int:
        """Number of simulation substeps per control step."""
        return self._n_substeps
    
    @property
    def mj_model(self) -> mujoco.MjModel:
        """MuJoCo model (for rendering)."""
        return self._mj_model
    
    @property
    def mjx_model(self) -> mjx.Model:
        """MJX model (for simulation)."""
        return self._mjx_model
    
    @property
    def xml_string(self) -> str:
        """Robot XML string."""
        return self._xml_string
    
    # =========================================================================
    # Rendering
    # =========================================================================
    
    def render(
        self,
        trajectory: Optional[Union[List[MJXState], MJXState]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        camera: Optional[str] = None,
    ) -> Union[np.ndarray, List[np.ndarray], None]:
        """Render trajectory, single state, or capture current frame.
        
        Three usage modes:
        1. render(trajectory=states) - Render list of states (batch rendering)
        2. render(trajectory=state) - Render single state
        3. render() - Return None, but capture frame if render_mode="mp4"
        
        Note: This runs on CPU as MJX doesn't support direct rendering.
        The MJX data is converted back to MuJoCo data for visualization.
        
        Args:
            trajectory: Single state, list of states, or None for current capture
            height: Image height (default from config)
            width: Image width (default from config)
            camera: Camera name (None for default)
            
        Returns:
            Image, list of images, or None
        """
        if height is None:
            height = self._render_size[1]
        if width is None:
            width = self._render_size[0]
        
        # If trajectory is provided, render it
        if trajectory is not None:
            return self._render_trajectory(trajectory, height, width, camera)
        
        # If no trajectory, return None (frame capture happens in capture_frame)
        return None
    
    def _render_trajectory(
        self,
        trajectory: Union[List[MJXState], MJXState],
        height: int,
        width: int,
        camera: Optional[str] = None,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Render a trajectory or single state to images."""
        renderer = mujoco.Renderer(self._mj_model, height=height, width=width)
        camera_id = -1 if camera is None else self._mj_model.camera(camera).id
        
        def get_image(state: MJXState) -> np.ndarray:
            d = mujoco.MjData(self._mj_model)
            d.qpos[:] = np.array(state.data.qpos)
            d.qvel[:] = np.array(state.data.qvel)
            mujoco.mj_forward(self._mj_model, d)
            renderer.update_scene(d, camera=camera_id)
            return renderer.render().copy()
        
        if isinstance(trajectory, list):
            images = [get_image(state) for state in trajectory]
            renderer.close()
            return images
        else:
            image = get_image(trajectory)
            renderer.close()
            return image
    
    def capture_frame(self, state: MJXState) -> Optional[np.ndarray]:
        """Capture a single frame from MJX state for video recording.
        
        This method converts MJX data back to MuJoCo data for CPU rendering,
        following the official MJX tutorial pattern.
        
        Args:
            state: MJX state to render
            
        Returns:
            Rendered frame as numpy array, or None if not recording
        """
        if self._render_mode != "mp4" or not self._recording_active:
            return None
        
        # Lazy initialize renderer
        if self._renderer is None:
            width, height = self._render_size
            self._renderer = mujoco.Renderer(self._mj_model, height=height, width=width)
            self._mj_data_for_render = mujoco.MjData(self._mj_model)
            self._preferred_camera_id = self._get_preferred_camera()
        
        # Convert MJX data to MuJoCo data using mjx.get_data (more complete)
        try:
            self._mj_data_for_render = mjx.get_data(self._mj_model, state.data)
        except Exception:
            # Fallback to manual copy
            self._mj_data_for_render.qpos[:] = np.array(state.data.qpos)
            self._mj_data_for_render.qvel[:] = np.array(state.data.qvel)
            mujoco.mj_forward(self._mj_model, self._mj_data_for_render)
        
        # Render
        self._renderer.update_scene(self._mj_data_for_render, camera=self._preferred_camera_id)
        pixels = self._renderer.render()
        frame = pixels.copy()
        
        # Store frame
        self._video_frames.append(frame)
        
        return frame
    
    def _get_preferred_camera(self) -> int:
        """Get preferred camera ID for rendering."""
        configured_camera = self.sim_cfg.get("render_camera", None)
        
        if configured_camera is not None:
            if isinstance(configured_camera, str):
                try:
                    return self._mj_model.camera(configured_camera).id
                except Exception:
                    pass
            elif isinstance(configured_camera, int):
                return configured_camera
        
        # Try to find common camera names
        preferred_names = ["follow_camera", "main_camera", "tracking_camera"]
        for i in range(self._mj_model.ncam):
            cam_name = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if cam_name and any(pn in cam_name.lower() for pn in preferred_names):
                return i
        
        # Default to free camera
        return -1
    
    def start_video_recording(self) -> None:
        """Start video recording (call before episode)."""
        if self._render_mode == "mp4":
            self._video_frames = []
            self._recording_active = True
            print(f"MJX video recording started for episode {self._episode_counter}")
    
    def stop_video_recording(self, suffix: str = "") -> bool:
        """Stop video recording and save video.
        
        Args:
            suffix: Optional suffix for video filename
            
        Returns:
            True if video was saved successfully
        """
        if self._render_mode != "mp4" or not self._recording_active:
            return False
        
        self._recording_active = False
        
        if not self._video_frames:
            print("No frames to save")
            return False
        
        try:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            
            # Generate filename
            video_dir = self._video_path or self._log_dir or "."
            os.makedirs(video_dir, exist_ok=True)
            
            filename = f"episode_{self._episode_counter}{suffix}.mp4"
            filepath = os.path.join(video_dir, filename)
            
            clip = ImageSequenceClip(self._video_frames, fps=self._video_fps)
            clip.write_videofile(
                filepath,
                codec="libx264",
                fps=self._video_fps,
                audio=False,
                logger=None,
            )
            clip.close()
            
            print(f"✓ MJX video saved: {filepath} ({len(self._video_frames)} frames)")
            self._video_frames = []
            return True
        except Exception as e:
            print(f"Failed to save video: {e}")
            return False
    
    def _should_record_episode(self) -> bool:
        """Check if current episode should be recorded."""
        if self._video_record_interval <= 0:
            return False
        return (self._episode_counter % self._video_record_interval) == 0
    
    def _cleanup_renderer(self) -> None:
        """Clean up rendering resources."""
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
            self._mj_data_for_render = None
    
    @property
    def render_mode(self) -> str:
        """Current render mode."""
        return self._render_mode
    
    @render_mode.setter
    def render_mode(self, mode: str) -> None:
        """Set render mode ('none', 'mp4', or 'viewer')."""
        if mode not in ("none", "mp4", "viewer"):
            raise ValueError(f"Invalid render_mode: {mode}. Must be 'none', 'mp4', or 'viewer'")
        self._render_mode = mode
    
    @property
    def log_dir(self) -> Optional[str]:
        """Logging directory path."""
        return self._log_dir
    
    def sync_viewer(self, state: MJXState) -> None:
        """Sync the passive viewer with the current MJX state.
        
        This method converts MJX data back to MuJoCo data and updates
        the passive viewer for real-time visualization.
        
        Call this method after each step() when render_mode="viewer".
        
        Args:
            state: Current MJX state to visualize
            
        Example:
            >>> env.render_mode = "viewer"
            >>> state = env.reset(rng)
            >>> env.sync_viewer(state)
            >>> for step in range(100):
            ...     state = env.step(state, action)
            ...     env.sync_viewer(state)
        """
        if self._render_mode != "viewer":
            return
        
        # Import viewer module (lazy import to avoid issues with headless systems)
        import mujoco.viewer as mujoco_viewer
        
        # Lazy initialize the passive viewer
        if self._passive_viewer is None:
            # Create MjData for viewer (need it before launching viewer)
            if self._mj_data_for_render is None:
                self._mj_data_for_render = mujoco.MjData(self._mj_model)
            
            # Launch passive viewer
            self._viewer_context_manager = mujoco_viewer.launch_passive(
                self._mj_model, self._mj_data_for_render
            )
            self._passive_viewer = self._viewer_context_manager.__enter__()
            print("MJX passive viewer initialized")
        
        # Copy MJX state to the existing MjData (must update in-place, not replace!)
        # The passive viewer is bound to self._mj_data_for_render, so we can't replace it
        self._mj_data_for_render.qpos[:] = np.array(state.data.qpos)
        self._mj_data_for_render.qvel[:] = np.array(state.data.qvel)
        self._mj_data_for_render.ctrl[:] = np.array(state.data.ctrl)
        
        # Run forward kinematics to update body positions, orientations, etc.
        mujoco.mj_forward(self._mj_model, self._mj_data_for_render)
        
        # Sync the viewer
        self._passive_viewer.sync()
    
    def close(self) -> None:
        """Close environment and cleanup resources."""
        # Stop any active video recording
        if self._render_mode == "mp4" and self._recording_active:
            self.stop_video_recording()
        
        # Cleanup passive viewer
        if self._viewer_context_manager is not None:
            try:
                self._viewer_context_manager.__exit__(None, None, None)
            except Exception:
                pass
            finally:
                self._passive_viewer = None
                self._viewer_context_manager = None
        
        # Cleanup renderer
        self._cleanup_renderer()
    
    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup


# =========================================================================
# Utility Functions
# =========================================================================

def create_mjx_env(config_name: str, **overrides) -> MJXMetaMachine:
    """Create MJX environment from config name.
    
    Args:
        config_name: Name of config in registry (e.g., "basic_quadruped")
        **overrides: Config overrides
        
    Returns:
        MJXMetaMachine environment
    """
    from .configs.config_registry import ConfigRegistry
    
    cfg = ConfigRegistry.create_from_name(config_name, **overrides)
    return MJXMetaMachine(cfg)
