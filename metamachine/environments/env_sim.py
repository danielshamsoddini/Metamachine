"""
Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import copy
import os
from collections import defaultdict

# Set MuJoCo rendering backend with fallback for headless environments
try:
    # Try EGL first (best for headless with GPU)
    os.environ["MUJOCO_GL"] = "egl"
    # Also set PyOpenGL platform for consistency
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
except Exception:
    # Fallback to osmesa for CPU-only rendering
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

# Note: optimize_pose is imported conditionally when needed to avoid JAX dependency issues
import pdb
from typing import Any, Optional, Union

# Import debug utilities if available
try:
    from .debug_utils import (
        debug_section,
        debug_trace,
        set_current_operation,
        _DEBUG_ENABLED,
    )
    _HAS_DEBUG_UTILS = True
except ImportError:
    _HAS_DEBUG_UTILS = False
    _DEBUG_ENABLED = False
    def debug_section(name):
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    def debug_trace(name):
        def decorator(func):
            return func
        return decorator
    def set_current_operation(name):
        pass

# from gymnasium.envs.mujoco import MujocoEnv
import cv2
import mujoco
import mujoco.viewer
import numpy as np
from omegaconf import OmegaConf

from metamachine.robot_factory.factory_registry import (
    get_default_draft_model_cfg,
    get_default_fine_model_cfg,
)

from .. import robot_factory
from ..robot_factory.core.xml_compiler import XMLCompiler
from ..utils.math_utils import (
    AverageFilter,
    construct_quaternion,
    quat_rotate,
    quat_rotate_inverse,
    quaternion_from_vectors,
    quaternion_multiply_alt,
    quaternion_to_euler,
    rotate_vector2D,
    wxyz_to_xyzw,
)
from ..utils.rendering import add_ground_disc_marker, add_ground_line_marker
from ..utils.validation import is_list_like, is_number
from .base import Base
from .gym.mujoco_env import MujocoEnv

# Try to get CONTROLLER_ROOT_DIR from environment or use default
try:
    from metamachine import METAMACHINE_ROOT_DIR

    ROOT_DIR = METAMACHINE_ROOT_DIR
except ImportError:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class MetaMachine(Base, MujocoEnv):
    """
    MetaMachine Simulation Environment

    A comprehensive robotic simulation environment that provides:
    - Modular robot morphology support with pose optimization
    - Advanced action processing with filtering and multiple control modes
    - Configurable reward systems with multiple components
    - Real-time visualization and video recording capabilities
    - Command-based high-level control interface
    - Comprehensive observation space with temporal stacking

    This environment serves as the main interface for reinforcement learning
    research, evolutionary robotics research, and robotic control experiments
    in the MetaMachine framework.

    Features high-fidelity physics simulation with comprehensive domain
    randomization, sensor simulation, and advanced control features. Currently
    implemented with MuJoCo physics engine.

    Args:
        cfg: OmegaConf configuration object containing all environment parameters

    Example:
        >>> from metamachine.environments.configs.config_registry import ConfigRegistry
        >>> from metamachine.environments.env_sim import MetaMachine
        >>> cfg = ConfigRegistry.create_from_name("basic_quadruped")
        >>> env = MetaMachine(cfg)
        >>> obs, info = env.reset()
        >>> obs, reward, done, truncated, info = env.step(action)
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, cfg: OmegaConf) -> None:
        """Initialize simulation environment.

        Args:
            cfg: Modern configuration object with simulation parameters
        """
        self.cfg = cfg

        # Validate simulation config
        self._validate_simulation_config(cfg)
        self._setup_logging()
        self._update_pose_cfg(cfg)

        # Initialize base class
        super().__init__(cfg)

        # Initialize simulation components
        self._initialize_environment(cfg)

        # Setup simulation-specific state
        self._setup_simulation_state()

    def _update_pose_cfg(self, cfg: Any) -> None:

        self._pose_setter = {
            "init_pos": lambda v: setattr(cfg.initialization, "init_pos", v),
            "init_quat": lambda v: setattr(cfg.initialization, "init_quat", v),
            "default_dof_pos": lambda v: setattr(cfg.control, "default_dof_pos", v),
            "forward_vec": lambda v: setattr(cfg.observation, "forward_vec", v),
            "projected_forward": lambda v: setattr(
                cfg.observation, "projected_forward_vec", v
            ),
            "projected_upward": lambda v: setattr(
                cfg.observation, "projected_upward_vec", v
            ),
        }

        if hasattr(cfg, "pose_optimization") and cfg.pose_optimization.enabled:
            # Perform pose optimization if enabled
            if cfg.pose_optimization.load_pose is None:
                self.pose_dict = self._optimize_pose()

                # Save optimized pose parameters
                pose_cfg = OmegaConf.create(self.pose_dict)
                if self._log_dir:  # Check that _log_dir is not None
                    pose_cfg_file = os.path.join(self._log_dir, "optimized_pose.yaml")
                    with open(pose_cfg_file, "w") as fp:
                        OmegaConf.save(config=pose_cfg, f=fp.name)
            else:
                pose_cfg_file = cfg.pose_optimization.load_pose
                self.pose_dict = OmegaConf.load(pose_cfg_file)
                print(f"Loaded pose parameters from {pose_cfg_file}")

            # Ensure pose_dict is a dict-like object
            if hasattr(self.pose_dict, "items"):
                pose_items = self.pose_dict.items()
            else:
                # If pose_dict is a tuple/other type, handle appropriately
                pose_items = []

            for key, value in pose_items:
                if key in self._pose_setter:
                    self._pose_setter[key](value)
                else:
                    print(f"Warning: Unrecognized pose parameter '{key}'")

    def _validate_simulation_config(self, cfg: Any) -> None:
        """Validate simulation configuration."""
        if not hasattr(cfg, "simulation"):
            raise ValueError("Missing 'simulation' section in config")

        required_sim_fields = ["mj_dt"]
        missing = [f for f in required_sim_fields if not hasattr(cfg.simulation, f)]
        if missing:
            raise ValueError(f"Missing simulation config fields: {missing}")

    def _initialize_environment(self, cfg: Any) -> None:
        """Initialize MuJoCo simulation components."""
        self.sim_cfg = cfg.simulation

        if hasattr(cfg, "morphology") and cfg.morphology.asset_file is not None:
            # Load and compile robot asset
            self._load_robot_asset()
        elif hasattr(cfg, "morphology") and cfg.morphology.configuration is not None:
            self._base_morphology_configuration = self._to_plain_container(
                cfg.morphology.configuration
            )
            self._validate_morphology_randomization_config()
            self._load_robot_asset_from_morphology(
                cfg.morphology.robot_type, self._base_morphology_configuration
            )
        else:
            raise ValueError("No robot asset file or morphology provided")

        self._initialize_actuation_model()

        # Setup MuJoCo environment
        self._setup_mujoco()

        # Initialize terrain system
        self._setup_terrain()

    def _optimize_pose(self) -> Any:
        """Optimize robot pose using the specified optimization method."""
        if (
            not hasattr(self.cfg, "pose_optimization")
            or not self.cfg.pose_optimization.enabled
        ):
            return {}
        else:
            # Import optimize_pose only when needed to avoid JAX dependency issues
            try:
                from metamachine.robot_factory.pose_optimizer import optimize_pose
            except ImportError as e:
                raise ImportError(
                    f"JAX and MuJoCo-MJX are required for pose optimization functionality. "
                    f"Install with: pip install 'metamachine[jax]' or pip install jax jaxlib mujoco-mjx\n"
                    f"Original error: {e}"
                ) from e

            print("Optimizing robot pose...")
            # Implement optimization logic here
            # This could involve running a physics simulation, adjusting joint angles, etc.
            self._load_draft_robot_asset_from_morphology(
                self.cfg.morphology.robot_type,  # type: ignore
                self.cfg.morphology.configuration,  # type: ignore
            )

            pose_dict = optimize_pose(
                self._draft_robot_instance,
                drop_steps=self.cfg.pose_optimization.drop_steps,
                move_steps=self.cfg.pose_optimization.move_steps,
                optimization_type=self.cfg.pose_optimization.optimization_type,
                enable_progress_bar=True,
                spine_assumption=False,
                seed=0,
                log_dir=self._log_dir,
            )
            return pose_dict

    def _load_robot_asset_from_morphology(
        self, robot_type: Any, morphology: Any
    ) -> None:
        """Load robot asset from morphology using the new factory system."""
        import tempfile
        import os

        factory_kwargs = self._get_morphology_factory_kwargs()
        
        factory_init_kwargs = {
            **get_default_fine_model_cfg(robot_type),
            **factory_kwargs,
        }

        # Get the factory using the new registry system
        factory = robot_factory.get_robot_factory(
            robot_type,
            sim_cfg=self.cfg.simulation,  # type: ignore
            log_dir=self._log_dir,  # Pass the environment's log directory
            **factory_init_kwargs,
        )
        if factory is None:
            raise ValueError(f"Unknown robot factory type: {robot_type}")

        # Create robot using the new factory interface
        robot = factory.create_robot(morphology=morphology, log_dir=self._log_dir)

        # Validate the robot
        is_valid, errors = robot.validate()
        if not is_valid:
            print(f"Warning: Robot validation failed: {errors}")

        # Store robot instance for potential future use
        self._robot_instance = robot
        
        # Morphology-based robots also initialize an XMLCompiler here so
        # per-episode XML randomization can reuse the generated model.
        # Check if restructure is enabled (for lego_legs or other graph-based robots)
        restructure = getattr(self.cfg.morphology, "restructure", False)
        restructure_qpos = getattr(self.cfg.morphology, "restructure_qpos", None)
        
        if restructure and hasattr(robot, "save"):
            # Save to temp file with restructuring
            # Use _log_dir if available to avoid race conditions in parallel environments
            if self._log_dir is not None:
                import uuid
                os.makedirs(os.path.join(self._log_dir, "tmp"), exist_ok=True)
                temp_xml_path = os.path.join(self._log_dir, "tmp", f"robot_temp_{uuid.uuid4().hex[:8]}.xml")
            else:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
                    temp_xml_path = f.name
            
            # Convert restructure_qpos from OmegaConf if needed
            if restructure_qpos is not None:
                from omegaconf import DictConfig, ListConfig, OmegaConf
                if isinstance(restructure_qpos, (DictConfig, ListConfig)):
                    restructure_qpos = OmegaConf.to_container(restructure_qpos, resolve=True)
            
            # Save with restructure
            saved_path = robot.save(
                temp_xml_path,
                restructure=True,
                restructure_qpos=restructure_qpos,
            )
            
            # Read the restructured XML
            with open(saved_path, 'r') as f:
                self.xml_string = f.read()
            
            # Create XMLCompiler from the restructured file
            self.xml_compiler = XMLCompiler(saved_path)
        else:
            # Get the XML string directly (no restructure)
            self.xml_string = robot.get_xml_string()
            
            # Create XMLCompiler from the generated XML for randomization support
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
                f.write(self.xml_string)
                temp_xml_path = f.name
            
            try:
                self.xml_compiler = XMLCompiler(temp_xml_path)
            finally:
                # Clean up temporary file
                if os.path.exists(temp_xml_path):
                    os.unlink(temp_xml_path)

        # Save debug XML only if log directory is available
        if self._log_dir is not None:
            self.xml_compiler.save(os.path.join(self._log_dir, "robot_debug.xml"))
        
        # Setup mass range if mass randomization is enabled
        randomization_cfg = getattr(self.cfg, "randomization", {})
        mass_cfg = randomization_cfg.get("mass", {})
        mass_enabled = mass_cfg.get("enabled", False)
        
        # Fallback to old style for backward compatibility
        if not mass_enabled:
            mass_enabled = self.sim_cfg.get("randomize_mass", False)
        
        if mass_enabled:
            # Get percentage from new or old config
            mass_percentage = mass_cfg.get("percentage", None)
            if mass_percentage is None:
                mass_percentage = self.sim_cfg.get("random_mass_percentage", 0.1)
            
            self.mass_range = self.xml_compiler.get_mass_range(mass_percentage)

    def _load_draft_robot_asset_from_morphology(
        self, robot_type: Any, morphology: Any
    ) -> None:
        """Load robot asset from morphology using the new factory system."""
        factory_kwargs = self._get_morphology_factory_kwargs()

        factory_init_kwargs = {
            **get_default_draft_model_cfg(robot_type),
            **factory_kwargs,
        }

        # Get the factory using the new registry system
        draft_factory = robot_factory.get_robot_factory(
            robot_type,
            sim_cfg=self.cfg.simulation,  # type: ignore
            log_dir=self._log_dir,  # Pass the environment's log directory
            **factory_init_kwargs,
        )

        if draft_factory is None:
            raise ValueError(f"Unknown robot factory type: {robot_type}")

        # # Prepare configuration for robot creation
        # robot_config = {
        #     'sim_cfg': OmegaConf.to_container(self.sim_cfg) if self.sim_cfg else None
        # }

        # Create robot using the new factory interface
        draft_robot = draft_factory.create_robot(morphology=morphology, log_dir=self._log_dir)

        # Validate the robot
        is_valid, errors = draft_robot.validate()
        if not is_valid:
            print(f"Warning: Robot validation failed: {errors}")

        # Get the XML string
        self.draft_xml_string = draft_robot.get_xml_string()

        # Store robot instance for potential future use
        self._draft_robot_instance = draft_robot

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
            from omegaconf import OmegaConf
            factory_kwargs = OmegaConf.to_container(factory_kwargs, resolve=True)
        except Exception:
            pass

        return dict(factory_kwargs) if isinstance(factory_kwargs, dict) else {}

    def _get_robot_actuation_defaults(self) -> dict[str, Any]:
        """Read robot-specific actuation defaults from the generated robot, if available."""
        robot = getattr(self, "_robot_instance", None)
        if robot is None or not hasattr(robot, "get_actuation_config"):
            return {}

        try:
            actuation_cfg = robot.get_actuation_config()
        except Exception:
            return {}

        if actuation_cfg is None:
            return {}
        return self._to_plain_container(actuation_cfg)

    def _initialize_actuation_model(self) -> None:
        """
        Resolve the actuation model used by tn_constraint clipping.

        Priority:
        1. Explicit simulation.actuation in the environment config
        2. Robot-specific defaults exposed by the factory/plugin
        3. Legacy hardcoded smart motor curve
        """
        robot_defaults = self._get_robot_actuation_defaults()

        if hasattr(self.sim_cfg, "get"):
            explicit_cfg = self.sim_cfg.get("actuation", None)
        else:
            explicit_cfg = getattr(self.sim_cfg, "actuation", None)

        explicit_cfg = self._to_plain_container(explicit_cfg) if explicit_cfg is not None else {}
        if not isinstance(explicit_cfg, dict):
            explicit_cfg = {}

        self._actuation_cfg_per_joint = None
        if "per_joint" in robot_defaults and "per_joint" not in explicit_cfg:
            per_joint = robot_defaults.get("per_joint", [])
            if isinstance(per_joint, list) and per_joint:
                self._actuation_cfg_per_joint = [
                    self._to_plain_container(cfg) for cfg in per_joint
                ]

        self._actuation_cfg = {
            **{k: v for k, v in robot_defaults.items() if k != "per_joint"},
            **explicit_cfg,
        }

    def _to_plain_container(self, value: Any) -> Any:
        """Convert OmegaConf containers to plain Python containers."""
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        return copy.deepcopy(value)

    def _get_morphology_randomization_cfg(self) -> dict[str, Any]:
        """Return morphology randomization config as a plain dictionary."""
        randomization_cfg = getattr(self.cfg, "randomization", {}) or {}
        morphology_cfg = randomization_cfg.get("morphology", {}) or {}
        plain_cfg = self._to_plain_container(morphology_cfg)
        return plain_cfg if isinstance(plain_cfg, dict) else {}

    def _has_morphology_randomization(self) -> bool:
        """Check whether morphology parameter randomization is enabled."""
        morphology_cfg = self._get_morphology_randomization_cfg()
        rules = morphology_cfg.get("component_params", []) or []
        return bool(
            morphology_cfg.get("enabled", False)
            and rules
            and hasattr(self, "_base_morphology_configuration")
        )

    def _validate_morphology_randomization_config(self) -> None:
        """Validate morphology randomization rules against the base morphology."""
        if not self._has_morphology_randomization():
            return

        base_morphology = getattr(self, "_base_morphology_configuration", None)
        if not isinstance(base_morphology, dict):
            raise ValueError(
                "Morphology randomization requires a dict-like morphology.configuration"
            )

        components = base_morphology.get("components", []) or []
        if not isinstance(components, list):
            raise ValueError("morphology.configuration.components must be a list")

        for rule in self._get_morphology_randomization_cfg().get("component_params", []):
            component_type = rule.get("component_type")
            param_rules = rule.get("params", {}) or {}
            if not component_type:
                raise ValueError(
                    "Each randomization.morphology.component_params entry needs component_type"
                )
            if not isinstance(param_rules, dict) or not param_rules:
                raise ValueError(
                    f"Morphology randomization rule for '{component_type}' needs a params mapping"
                )

            matching_components = [
                component
                for component in components
                if str(component.get("component_type")) == str(component_type)
            ]
            if not matching_components:
                raise ValueError(
                    f"No components found for morphology randomization type '{component_type}'"
                )

            for param_name, param_cfg in param_rules.items():
                if not isinstance(param_cfg, dict):
                    raise ValueError(
                        f"Morphology randomization config for '{component_type}.{param_name}' "
                        "must be a mapping"
                    )

                param_mode = param_cfg.get("mode", "absolute")
                param_range = param_cfg.get("range")
                if (
                    not is_list_like(param_range)
                    or len(param_range) != 2
                    or not all(is_number(v) for v in param_range)
                ):
                    raise ValueError(
                        f"Morphology randomization for '{component_type}.{param_name}' "
                        "requires numeric range: [min, max]"
                    )
                if param_mode not in {"absolute", "percentage", "additive"}:
                    raise ValueError(
                        f"Unsupported morphology randomization mode '{param_mode}' "
                        f"for '{component_type}.{param_name}'"
                    )

                matched_param = False
                for component in matching_components:
                    component_params = component.get("params", {}) or {}
                    if param_name not in component_params:
                        continue
                    matched_param = True
                    if not is_number(component_params[param_name]):
                        raise ValueError(
                            f"Morphology randomization only supports numeric params, got "
                            f"'{component_type}.{param_name}'={component_params[param_name]!r}"
                        )

                if not matched_param:
                    raise ValueError(
                        f"No '{component_type}' components define param '{param_name}'"
                    )

    def _sample_morphology_param(
        self, base_value: float, param_cfg: dict[str, Any]
    ) -> float:
        """Sample one randomized morphology parameter from its base value."""
        param_mode = param_cfg.get("mode", "absolute")
        low, high = param_cfg["range"]

        if param_mode == "percentage":
            return float(base_value * (1.0 + self.np_random.uniform(low, high)))
        if param_mode == "additive":
            return float(base_value + self.np_random.uniform(low, high))
        return float(self.np_random.uniform(low, high))

    def _get_randomized_morphology_configuration(self) -> dict[str, Any]:
        """Create a randomized morphology config from the pristine base graph."""
        randomized = copy.deepcopy(self._base_morphology_configuration)
        rules = self._get_morphology_randomization_cfg().get("component_params", []) or []

        for component in randomized.get("components", []) or []:
            component_type = str(component.get("component_type"))
            component_params = component.setdefault("params", {})

            for rule in rules:
                if component_type != str(rule.get("component_type")):
                    continue

                for param_name, param_cfg in (rule.get("params", {}) or {}).items():
                    if param_name not in component_params:
                        continue
                    component_params[param_name] = self._sample_morphology_param(
                        float(component_params[param_name]), param_cfg
                    )

        return randomized

    def get_robot_instance(self) -> Any:
        """Get the robot instance if available (new factory system only)."""
        return getattr(self, "_robot_instance", None)

    def _load_robot_asset(self) -> None:
        """Load and compile robot asset."""
        asset_files = self.cfg.morphology.asset_file  # type: ignore
        self.randomize_asset = is_list_like(asset_files)

        asset_file = (
            np.random.choice(asset_files) if self.randomize_asset else asset_files
        )

        xml_path = self._resolve_asset_xml_path(asset_file)

        # Initialize XML compiler with modifications
        self.xml_compiler = XMLCompiler(xml_path)
        self.xml_compiler.torque_control()
        self.xml_compiler.update_timestep(self.sim_cfg.mj_dt)

        # Apply asset modifications
        if self.sim_cfg.get("pyramidal_cone", False):
            self.xml_compiler.pyramidal_cone()

        # Check randomization config (new style: cfg.randomization.mass)
        randomization_cfg = getattr(self.cfg, "randomization", {})
        mass_cfg = randomization_cfg.get("mass", {})
        mass_enabled = mass_cfg.get("enabled", False)
        
        # Fallback to old style for backward compatibility
        if not mass_enabled:
            mass_enabled = self.sim_cfg.get("randomize_mass", False)
        
        if mass_enabled:
            # Get percentage from new or old config
            mass_percentage = mass_cfg.get("percentage", None)
            if mass_percentage is None:
                mass_percentage = self.sim_cfg.get("random_mass_percentage", 0.1)
            
            self.mass_range = self.xml_compiler.get_mass_range(mass_percentage)

        self.xml_string = self.xml_compiler.get_string()

    def _resolve_asset_xml_path(self, asset_file: Any) -> str:
        """Resolve XML path from absolute path, repo-relative path, or built-in asset name."""
        if asset_file is None:
            raise ValueError("morphology.asset_file cannot be None when loading from asset")

        asset_file_str = str(asset_file)
        candidates = []

        if os.path.isabs(asset_file_str):
            candidates.append(asset_file_str)
        else:
            repo_root = os.path.dirname(ROOT_DIR)
            candidates.extend(
                [
                    asset_file_str,  # relative to current working directory
                    os.path.join(repo_root, asset_file_str),  # relative to repo root
                    os.path.join(ROOT_DIR, asset_file_str),  # relative to metamachine package root
                    os.path.join(ROOT_DIR, "assets", "robots", asset_file_str),  # legacy behavior
                ]
            )

        for path in candidates:
            abs_path = os.path.abspath(os.path.expanduser(path))
            if os.path.exists(abs_path):
                return abs_path

        raise FileNotFoundError(
            f"Could not find XML asset '{asset_file_str}'. Tried: "
            + ", ".join(os.path.abspath(os.path.expanduser(p)) for p in candidates)
        )

    def _setup_mujoco(self) -> None:
        """Setup MuJoCo environment."""
        # Calculate simulation parameters
        self.mj_dt = self.sim_cfg.mj_dt
        assert (
            self.cfg.control.dt % self.mj_dt < 1e-9  # type: ignore
        ), f"dt ({self.cfg.control.dt}) must be multiple of mj_dt ({self.mj_dt})"  # type: ignore
        self.frame_skip = int(self.cfg.control.dt / self.mj_dt)  # type: ignore

        # Rendering configuration
        self.render_size = self.sim_cfg.get("render_size", [426, 240])
        # self.render_on = self.sim_cfg.get('render', False)

        # New rendering mode configuration
        self.render_mode = self.sim_cfg.get(
            "render_mode", "none"
        )  # 'none', 'viewer', 'mp4'
        self.video_path = self.sim_cfg.get("video_path", self._log_dir)
        self.video_path = self._log_dir if self.video_path is None else self.video_path

        # Video recording configuration
        self.video_record_interval = self.sim_cfg.get(
            "video_record_interval", 1
        )  # Record every N episodes
        self.video_name_pattern = self.sim_cfg.get(
            "video_name_pattern", "episode_{episode}"
        )  # Naming pattern
        self.video_base_name = self.sim_cfg.get(
            "video_base_name", "robot_video"
        )  # Base name without extension

        # Setup EGL environment if needed
        if self.render_mode == "mp4":
            self._setup_egl_environment()
            self._initialize_video_recording()

        # Initialize MuJoCo
        MujocoEnv.__init__(
            self,
            self.xml_string,
            self.frame_skip,
            observation_space=None,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            width=self.render_size[0],
            height=self.render_size[1],
        )

        self._setup_spaces()

        # Restore our render settings after MuJoCo init (it might overwrite them)
        self.render_mode = self.sim_cfg.get(
            "render_mode", "none"
        )  # 'none', 'viewer', 'mp4'
        self.video_fps = self.sim_cfg.get("video_fps", None)
        if self.video_fps is None:
            self.video_fps = 1 / self.cfg.control.dt  # type: ignore

        # Set up termination checker with model for body contact termination
        self.termination_checker.set_model(self.model)

    def _setup_terrain(self) -> None:
        """Setup terrain generation system."""
        # TODO: Implement terrain system when available
        # Currently no 'Terrain' class exists, only TerrainBuilder which requires arguments
        self.terrain_resetter = None
        if hasattr(self.sim_cfg, "get") and self.sim_cfg.get("terrain"):
            print("Warning: Terrain requested but module unavailable")

    def _setup_egl_environment(self) -> None:
        """Setup EGL environment for headless rendering."""
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
        os.environ["PYOPENGL_PLATFORM"] = "egl"

        print("EGL environment configured for headless rendering")

    def _initialize_video_recording(self) -> None:
        """Initialize video recording system."""
        self.video_frames: list[np.ndarray] = []
        self.egl_renderer: Optional[Union[str, Any]] = None
        self.recording_active = False
        
        # Callback lists for video customization
        self._pre_render_callbacks: list = []  # Called after update_scene, before render
        self._post_render_callbacks: list = []  # Called after frame capture, before overlay
        self._frame_overlay_callbacks: list = []  # Called after metrics overlay
        
        print(
            f"Video recording initialized. Recording every {self.video_record_interval} episodes."
        )
    
    def register_pre_render_callback(self, callback) -> None:
        """Register a callback to be called before rendering each frame.
        
        The callback receives (renderer, data) and can add markers to the scene.
        Called after update_scene() but before render().
        
        Example:
            def add_target_marker(renderer, data):
                from metamachine.utils.rendering import add_sphere_marker
                add_sphere_marker(renderer.scene, pos=[1, 0, 0.1], rgba=[1, 0, 0, 0.8])
            
            env.register_pre_render_callback(add_target_marker)
        """
        if not hasattr(self, '_pre_render_callbacks'):
            self._pre_render_callbacks = []
        self._pre_render_callbacks.append(callback)
    
    def register_post_render_callback(self, callback) -> None:
        """Register a callback to be called after rendering each frame.
        
        The callback receives (frame) and should return the modified frame.
        Called after frame capture but before metrics overlay.
        
        Example:
            def add_custom_drawing(frame):
                cv2.circle(frame, (100, 100), 20, (0, 255, 0), -1)
                return frame
            
            env.register_post_render_callback(add_custom_drawing)
        """
        if not hasattr(self, '_post_render_callbacks'):
            self._post_render_callbacks = []
        self._post_render_callbacks.append(callback)
    
    def register_frame_overlay_callback(self, callback) -> None:
        """Register a callback to add overlays after the default metrics.
        
        The callback receives (frame) and should return the modified frame.
        Called after _add_metrics_overlay().
        
        Example:
            def add_bearing_info(frame):
                cv2.putText(frame, "Bearing: 45°", (500, 30), ...)
                return frame
            
            env.register_frame_overlay_callback(add_bearing_info)
        """
        if not hasattr(self, '_frame_overlay_callbacks'):
            self._frame_overlay_callbacks = []
        self._frame_overlay_callbacks.append(callback)
    
    def clear_video_callbacks(self) -> None:
        """Clear all registered video callbacks."""
        self._pre_render_callbacks = []
        self._post_render_callbacks = []
        self._frame_overlay_callbacks = []
    
    def get_current_renderer(self) -> Any:
        """Get the current EGL renderer (creates one if needed).
        
        Returns:
            The MuJoCo renderer object, or "synthetic" if EGL is unavailable.
        """
        return self._create_egl_renderer()

    def _create_egl_renderer(self) -> Any:
        """Create EGL renderer for direct rendering."""
        if self.egl_renderer is None:
            try:
                width, height = self.render_size
                self.egl_renderer = mujoco.Renderer(
                    self.model, height=height, width=width
                )
                print(f"EGL renderer created ({width}x{height})")
                self.preferred_camera_id = self._get_preferred_camera()
            except Exception as e:
                print(f"Failed to create EGL renderer: {e}")
                print("Using synthetic rendering mode")
                self.egl_renderer = "synthetic"
        return self.egl_renderer

    def _cleanup_egl_renderer(self) -> None:
        """Clean up EGL renderer."""
        if hasattr(self, "egl_renderer") and self.egl_renderer is not None:
            if self.egl_renderer != "synthetic":
                try:
                    if hasattr(self.egl_renderer, "close"):
                        self.egl_renderer.close()
                except Exception as e:
                    print(f"Warning: Error closing EGL renderer: {e}")
            self.egl_renderer = None

    def _create_synthetic_frame(self, width: int, height: int) -> np.ndarray:
        """Create a synthetic frame with robot state visualization."""
        from io import BytesIO

        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 3)
        ax.set_aspect("equal")

        # Draw ground
        ground = patches.Rectangle((-2, -1), 4, 0.1, color="brown")
        ax.add_patch(ground)

        # Get robot position
        if hasattr(self, "data") and self.data is not None:
            pos = self.data.qpos[:3]
            self.data.qpos[3:7]

            # Draw robot as a simple circle
            robot = patches.Circle((pos[0], pos[2]), 0.2, color="blue", alpha=0.7)
            ax.add_patch(robot)

            # Draw velocity vector
            if hasattr(self, "data"):
                vel = self.data.qvel[:3] * 0.5  # Scale for visibility
                ax.arrow(
                    pos[0],
                    pos[2],
                    vel[0],
                    vel[2],
                    head_width=0.05,
                    head_length=0.05,
                    fc="red",
                    ec="red",
                )

            # Add text info
            ax.text(-1.8, 2.7, f"Step: {getattr(self, 'step_count', 0)}", fontsize=8)
            ax.text(-1.8, 2.5, f"Pos: ({pos[0]:.2f}, {pos[2]:.2f})", fontsize=8)

        ax.set_title("Robot Simulation (Synthetic View)", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Convert to numpy array
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        buf.seek(0)

        from PIL import Image

        img = Image.open(buf)
        frame = np.array(img)[:, :, :3]  # Remove alpha channel if present

        plt.close(fig)
        buf.close()

        return frame

    def _capture_frame_egl(self) -> Optional[np.ndarray]:
        """Capture frame using EGL renderer.
        
        This method supports registered callbacks for customization:
        - pre_render_callbacks: Called after update_scene, before render (for 3D markers)
        - post_render_callbacks: Called after capture, before overlay (for frame processing)
        - frame_overlay_callbacks: Called after metrics overlay (for additional text/graphics)
        """
        if self.render_mode != "mp4" or not self.recording_active:
            return None

        renderer = self._create_egl_renderer()

        if renderer == "synthetic":
            width, height = self.render_size
            frame = self._create_synthetic_frame(width, height)
        else:
            camera_id = getattr(self, "preferred_camera_id", -1)
            renderer.update_scene(self.data, camera=camera_id)
            self._add_goal_markers_to_scene(renderer.scene)
            
            # Call pre-render callbacks (for adding 3D markers to scene)
            for callback in getattr(self, '_pre_render_callbacks', []):
                try:
                    callback(renderer, self.data)
                except Exception as e:
                    pass  # Silently ignore callback errors
            
            pixels = renderer.render()
            frame = 1 - pixels
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Call post-render callbacks (for frame processing)
        for callback in getattr(self, '_post_render_callbacks', []):
            try:
                frame = callback(frame)
            except Exception as e:
                pass  # Silently ignore callback errors

        frame = self._add_metrics_overlay(frame)
        
        # Call frame overlay callbacks (for additional overlays)
        for callback in getattr(self, '_frame_overlay_callbacks', []):
            try:
                frame = callback(frame)
            except Exception as e:
                pass  # Silently ignore callback errors
        
        self.video_frames.append(frame)
        return frame

    def _save_video(self) -> bool:
        """Save collected frames to MP4 video using moviepy."""
        if not self.video_frames:
            print("No frames to save")
            return False

        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        filename = self._generate_video_filename()

        clip = ImageSequenceClip(list(self.video_frames), fps=self.video_fps)
        clip.write_videofile(
            filename,
            codec="libx264",
            fps=self.video_fps,
            audio=False,
            logger=None,
        )
        clip.close()

        file_size = os.path.getsize(filename)
        print(f"✓ Video saved: {filename} ({file_size:,} bytes, {len(self.video_frames)} frames)")
        return True

    def _generate_video_filename(self) -> str:
        """Generate video filename based on pattern and episode counter."""
        # Extract base name and extension
        video_path = self.video_path or "output"  # Default if None
        base_name, ext = os.path.splitext(video_path)
        if not ext:
            ext = ".mp4"
            video_dir = video_path
        else:
            video_dir = os.path.dirname(video_path)

        # Generate filename using pattern
        if "{episode}" in self.video_name_pattern:
            filename = str(
                self.video_name_pattern.format(episode=self.episode_counter + 1)
            )
        else:
            # Fallback to pattern as is if no {episode} placeholder
            filename = str(self.video_name_pattern)

        # Ensure it has .mp4 extension
        if not filename.endswith(".mp4"):
            filename += ext

        # Use directory from original video_path if provided
        if video_dir:
            filename = os.path.join(video_dir, filename)

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        return filename

    def _should_record_episode(self) -> bool:
        """Check if current episode should be recorded based on interval."""
        if self.video_record_interval is None or self.video_record_interval <= 0:
            return False
        return (
            self.episode_counter + 1
        ) % self.video_record_interval == 0 or self.episode_counter == 0

    def start_video_recording(self) -> None:
        """Start video recording."""
        if self.render_mode == "mp4":
            self.video_frames = []
            self.recording_active = True
            print(f"Video recording started for episode {self.episode_counter}")
        else:
            print("Video recording not available (render_mode must be 'mp4')")

    def stop_video_recording(self) -> bool:
        """Stop video recording and save video."""
        if self.render_mode == "mp4" and self.recording_active:
            self.recording_active = False
            success = self._save_video()
            self._cleanup_egl_renderer()
            return success
        return False

    def _setup_simulation_state(self) -> None:
        """Setup simulation-specific state and tracking."""
        self.state.set_module_orientation_offset_sim_enabled(True)

        # Robot configuration
        self._parse_robot_parameters()

        # Joint and body setup
        self._setup_robot_model()

        # Control tracking
        self._initialize_control_state()

        # Sensor and data tracking
        self._initialize_sensors()

        # External forces
        self._setup_external_forces()

        # Per-module randomization events (latency, torque, control attenuation)
        self._setup_asymmetric_randomization()

        # Initialization parameters
        self._setup_initialization_parameters()

        # Initialize viewer for viewer mode
        self._passive_viewer = None
        self._viewer_context_manager = None
        self._setup_goal_task_state()

    def _setup_goal_task_state(self) -> None:
        """Initialize optional forward goal-reaching task state."""
        goal_cfg = getattr(self.cfg.task, "goal", None)
        self.goal_cfg = goal_cfg
        self.goal_task_enabled = bool(goal_cfg and goal_cfg.get("enabled", False))
        self.goal_position_world: Optional[np.ndarray] = None
        self.goal_distance = -1.0
        self.goal_distance_delta = 0.0
        self._previous_goal_distance: Optional[float] = None

    def _get_goal_forward_basis(self) -> tuple[np.ndarray, np.ndarray]:
        """Return forward and lateral unit vectors on the ground plane."""
        forward_xy = getattr(self, "adjusted_forward_vec", self.forward_vec)
        if forward_xy is None:
            forward_xy = np.array([1.0, 0.0], dtype=np.float32)
        else:
            forward_xy = np.asarray(forward_xy[:2], dtype=np.float32)

        norm = float(np.linalg.norm(forward_xy))
        if norm < 1e-6:
            forward_xy = np.array([1.0, 0.0], dtype=np.float32)
        else:
            forward_xy = forward_xy / norm

        lateral_xy = np.array([-forward_xy[1], forward_xy[0]], dtype=np.float32)
        return forward_xy, lateral_xy

    def _reset_goal_task(self) -> None:
        """Sample a new goal in front of the robot."""
        if not self.goal_task_enabled:
            return

        goal_cfg = self.goal_cfg
        robot_xy = np.asarray(self.data.xpos[self.torso_body_id][:2], dtype=np.float32)
        forward_xy, lateral_xy = self._get_goal_forward_basis()

        distance_range = np.asarray(
            goal_cfg.get("distance_range", [0.35, 0.75]),
            dtype=np.float32,
        )
        lateral_range = np.asarray(
            goal_cfg.get("lateral_range", [0.0, 0.0]),
            dtype=np.float32,
        )
        height = float(goal_cfg.get("height", 0.03))

        if distance_range.size == 1:
            forward_distance = float(distance_range[0])
        else:
            forward_distance = float(
                self.np_random.uniform(float(distance_range[0]), float(distance_range[1]))
            )

        if lateral_range.size == 1:
            lateral_offset = float(lateral_range[0])
        else:
            lateral_offset = float(
                self.np_random.uniform(float(lateral_range[0]), float(lateral_range[1]))
            )

        goal_xy = robot_xy + forward_distance * forward_xy + lateral_offset * lateral_xy
        self.goal_position_world = np.array(
            [goal_xy[0], goal_xy[1], height],
            dtype=np.float32,
        )
        self.goal_distance = -1.0
        self.goal_distance_delta = 0.0
        self._previous_goal_distance = None

    def _get_goal_observable_data(self, pos_world: np.ndarray) -> dict[str, Any]:
        """Compute goal distance signals for observations and rewards."""
        if not self.goal_task_enabled or self.goal_position_world is None:
            return {}

        goal_vector_xy = np.asarray(
            self.goal_position_world[:2] - pos_world[:2],
            dtype=np.float32,
        )
        current_distance = float(np.linalg.norm(goal_vector_xy))
        if self._previous_goal_distance is None:
            distance_delta = 0.0
        else:
            distance_delta = float(self._previous_goal_distance - current_distance)

        self.goal_distance = current_distance
        self.goal_distance_delta = distance_delta
        self._previous_goal_distance = current_distance

        return {
            "goal_distance": current_distance,
            "goal_distance_delta": distance_delta,
            "goal_distances": np.full(self.num_act, current_distance, dtype=np.float32),
            "goal_position_world": self.goal_position_world.copy(),
            "goal_vector_world": np.array(
                [
                    self.goal_position_world[0] - pos_world[0],
                    self.goal_position_world[1] - pos_world[1],
                    self.goal_position_world[2] - pos_world[2],
                ],
                dtype=np.float32,
            ),
        }

    def _add_goal_markers_to_scene(self, scene: Any) -> None:
        """Visualize the goal and robot-to-goal line in recorded videos."""
        if (
            not self.goal_task_enabled
            or self.goal_position_world is None
            or not bool(self.goal_cfg.get("visualize", True))
        ):
            return

        goal_xy = np.asarray(self.goal_position_world[:2], dtype=np.float32)
        robot_xy = np.asarray(self.data.xpos[self.torso_body_id][:2], dtype=np.float32)
        marker_radius = float(self.goal_cfg.get("marker_radius", 0.06))
        line_radius = float(self.goal_cfg.get("line_radius", 0.01))
        z_offset = float(self.goal_position_world[2])

        add_ground_disc_marker(
            scene,
            pos_xy=goal_xy,
            radius=marker_radius,
            height=float(self.goal_cfg.get("marker_height", 0.01)),
            color=(0.15, 0.95, 0.25, 0.85),
            z_offset=z_offset,
        )
        add_ground_line_marker(
            scene,
            start_xy=robot_xy,
            end_xy=goal_xy,
            radius=line_radius,
            color=(0.95, 0.35, 0.2, 0.75),
            z_offset=max(z_offset, 0.02),
        )

    def _parse_robot_parameters(self) -> None:
        """Parse robot-specific parameters."""
        # Get parameters from their correct config locations
        self.theta = getattr(self.cfg.environment, "theta", 0.610865)  # type: ignore
        self.kp = getattr(self.cfg.control, "kp", 8.0)  # type: ignore
        self.kd = getattr(self.cfg.control, "kd", 0.2)  # type: ignore

        control_cfg = self.cfg.control  # type: ignore

        if control_cfg.num_actions is None:
            self.num_act = self.model.nu
        else:
            self.num_act = control_cfg.num_actions
        self.num_envs = self.cfg.environment.num_envs  # type: ignore

    def _setup_robot_model(self) -> None:
        """Setup robot model parameters after MuJoCo initialization."""
        self.num_joint = self.model.nu

        # Get actuator-driven joints (in actuator order for generic XML support)
        actuator_joint_ids = [int(j) for j in self.model.actuator_trnid[:, 0]]
        actuator_joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in actuator_joint_ids
        ]

        # Validate action space
        expected_joints = self.num_act * self.num_envs
        if expected_joints != self.num_joint:
            raise ValueError(
                f"Action space mismatch: expected {expected_joints}, got {self.num_joint}"
            )

        # Try modular naming first (joint{module_id}), then fall back to generic joints.
        self._uses_modular_joint_naming = False
        parsed_module_ids = []
        try:
            for name in actuator_joint_names:
                if name is None or not name.startswith("joint"):
                    raise ValueError("Non-modular joint name detected")
                parsed_module_ids.append(int(name.replace("joint", "")))

            self.jointed_module_ids = sorted(parsed_module_ids)
            self.joint_idx = [
                self.model.joint(f"joint{i}").id for i in self.jointed_module_ids
            ]
            self.joint_geom_idx = [
                self.model.geom(f"left{i}").id for i in self.jointed_module_ids
            ] + [self.model.geom(f"right{i}").id for i in self.jointed_module_ids]
            self.joint_body_idx = [
                int(self.model.geom(f"left{i}").bodyid.item())
                for i in self.jointed_module_ids
            ]
            self._uses_modular_joint_naming = True
        except (ValueError, KeyError):
            self.jointed_module_ids = []
            self.joint_idx = actuator_joint_ids
            joint_body_ids = [int(self.model.jnt_bodyid[j]) for j in self.joint_idx]
            # Preserve order while de-duplicating
            self.joint_body_idx = list(dict.fromkeys(joint_body_ids))
            joint_body_set = set(self.joint_body_idx)
            self.joint_geom_idx = [
                geom_id
                for geom_id in range(self.model.ngeom)
                if int(self.model.geom_bodyid[geom_id]) in joint_body_set
            ]

        self.torso_body_id = self._resolve_torso_body_id()
        self.floor_geom_ids = self._resolve_floor_geom_ids()

        # Setup PD gains with per-joint control support
        self._setup_pd_gains()

    def _resolve_torso_body_id(self) -> int:
        """Resolve torso body ID for both modular and generic XML models."""
        torso_body_name = self.cfg.observation.get("torso_body_name", None)  # type: ignore
        if torso_body_name is not None:
            try:
                return self.model.body(str(torso_body_name)).id
            except KeyError:
                print(
                    f"Warning: torso_body_name '{torso_body_name}' not found. Falling back to auto torso body."
                )

        torso_node_id = self.cfg.observation.get("torso_node_id", 0)  # type: ignore
        modular_name = f"l{torso_node_id}"
        try:
            return self.model.body(modular_name).id
        except KeyError:
            pass

        if isinstance(torso_node_id, str):
            try:
                return self.model.body(torso_node_id).id
            except KeyError:
                pass

        if getattr(self, "joint_body_idx", None):
            return int(self.joint_body_idx[0])

        # Fallback to first non-world body when available
        return 1 if self.model.nbody > 1 else 0

    def _resolve_floor_geom_ids(self) -> list[int]:
        """Resolve floor geom IDs from config, with robust defaults."""
        term_cfg = self.cfg.task.get("termination_conditions", {})  # type: ignore
        configured_floor_names = term_cfg.get("floor_geom_names", None)

        if configured_floor_names is None:
            floor_names = ["floor"]
            user_configured = False
        elif isinstance(configured_floor_names, str):
            floor_names = [configured_floor_names]
            user_configured = True
        else:
            floor_names = list(configured_floor_names)
            user_configured = True

        floor_ids: set[int] = set()
        for floor_name in floor_names:
            try:
                floor_ids.add(self.model.geom(str(floor_name)).id)
            except KeyError:
                if user_configured:
                    print(f"Warning: floor geom '{floor_name}' not found in model.")

        # Auto-detect plane geoms when default "floor" is not present.
        if not floor_ids and not user_configured:
            for geom_id in range(self.model.ngeom):
                if self.model.geom(geom_id).type == mujoco.mjtGeom.mjGEOM_PLANE:
                    floor_ids.add(geom_id)

        # Legacy fallback
        if not floor_ids and self.model.ngeom > 0:
            floor_ids.add(0)
            print("Warning: Falling back to geom id 0 as floor.")

        return sorted(floor_ids)

    def _setup_pd_gains(self) -> None:
        """Setup PD gains with optional per-joint configuration.
        
        This method supports:
        1. Uniform gains (default): All joints use the same kp/kd
        2. Wheel joints: Specific joints configured as velocity-controlled (kp=0, higher kd)
        3. Fine-grained: Explicit per-joint kp/kd arrays
        
        For velocity-controlled joints (wheels), the action is interpreted as velocity target
        rather than position offset.
        """
        per_joint_cfg = getattr(self.cfg.control, 'per_joint_control', None)
        
        if per_joint_cfg and per_joint_cfg.get('enabled', False):
            # Initialize with global values
            self.kps = np.full(self.num_act, self.kp, dtype=np.float32)
            self.kds = np.full(self.num_act, self.kd, dtype=np.float32)
            self.joint_control_modes = ['position'] * self.num_act
            self.default_dof_vel = np.zeros(self.num_act, dtype=np.float32)
            self.wheel_action_scale = 1.0
            
            # Apply wheel joints configuration
            wheel_cfg = per_joint_cfg.get('wheel_joints', {})
            wheel_indices = wheel_cfg.get('indices', [])
            if wheel_indices:
                wheel_kd = wheel_cfg.get('kd', 2.0)
                default_vel = wheel_cfg.get('default_velocity', 0.0)
                self.wheel_action_scale = wheel_cfg.get('action_scale', 1.0)
                
                for idx in wheel_indices:
                    if idx >= self.num_act:
                        raise ValueError(
                            f"Wheel joint index {idx} >= num_actions {self.num_act}"
                        )
                    self.kps[idx] = 0.0  # No position control for wheels
                    self.kds[idx] = wheel_kd
                    self.joint_control_modes[idx] = 'velocity'
                    self.default_dof_vel[idx] = default_vel
            
            # Override with explicit per-joint config if provided
            if per_joint_cfg.get('per_joint_kp') is not None:
                self.kps = np.array(per_joint_cfg.per_joint_kp, dtype=np.float32)
            if per_joint_cfg.get('per_joint_kd') is not None:
                self.kds = np.array(per_joint_cfg.per_joint_kd, dtype=np.float32)
            if per_joint_cfg.get('per_joint_mode') is not None:
                self.joint_control_modes = list(per_joint_cfg.per_joint_mode)
                # Update default_dof_vel for velocity-controlled joints
                for i, mode in enumerate(self.joint_control_modes):
                    if mode == 'velocity' and i not in wheel_indices:
                        self.default_dof_vel[i] = 0.0
            
            # Store wheel indices for later use
            self.wheel_joint_indices = [
                i for i, mode in enumerate(self.joint_control_modes) 
                if mode == 'velocity'
            ]
        else:
            # Original behavior: uniform gains for all joints
            self.kps = np.full(self.num_act * self.num_envs, self.kp, dtype=np.float32)
            self.kds = np.full(self.num_act * self.num_envs, self.kd, dtype=np.float32)
            self.joint_control_modes = None
            self.default_dof_vel = None
            self.wheel_joint_indices = []
            self.wheel_action_scale = 1.0


    def _initialize_control_state(self) -> None:
        """Initialize control state tracking."""
        # Default DOF positions
        default_dof_pos = self.cfg.control.get("default_dof_pos", 0)  # type: ignore
        if isinstance(default_dof_pos, (list, np.ndarray)):
            self.default_dof_pos = np.array(default_dof_pos)
        else:
            self.default_dof_pos = np.full(self.num_joint, default_dof_pos)

        # Latency simulation tracking
        self.last_pos_sim = self.default_dof_pos.copy()
        self.last_last_pos_sim = self.default_dof_pos.copy()
        self.last_vel_sim = np.zeros(self.num_joint)
        self.last_last_vel_sim = np.zeros(self.num_joint)

        # Other tracking
        self.last_com_pos = np.zeros(3)
        self.episode_counter = 0

    def _initialize_sensors(self) -> None:
        """Initialize sensor data structures."""
        self.sensors: defaultdict[str, Any] = defaultdict(list)

    def _setup_external_forces(self) -> None:
        """Setup external force system."""
        self.external_forces_enabled = self.sim_cfg.get("random_external_force", False)
        if self.external_forces_enabled:
            self.external_force_config = {
                "ranges": self.sim_cfg.random_external_force_ranges,
                "bodies": self.sim_cfg.random_external_force_bodies,
                "positions": self.sim_cfg.random_external_force_positions,
                "directions": self.sim_cfg.random_external_force_directions,
                "durations": self.sim_cfg.random_external_force_durations,
                "interval": self.sim_cfg.random_external_force_interval,
            }
            self.external_force_counter = dict.fromkeys(
                range(len(self.external_force_config["bodies"])), 0
            )

    def _setup_asymmetric_randomization(self) -> None:
        """Parse and initialize per-module randomization state."""
        randomization_cfg = self._to_plain_container(
            getattr(self.cfg, "randomization", {}) or {}
        )
        if not isinstance(randomization_cfg, dict):
            randomization_cfg = {}

        self._module_id_to_joint_index = {
            int(module_id): idx
            for idx, module_id in enumerate(getattr(self, "jointed_module_ids", []))
        }

        self._module_latency_cfg = randomization_cfg.get("module_latency", {}) or {}
        if not isinstance(self._module_latency_cfg, dict):
            self._module_latency_cfg = {}
        self._module_latency_enabled = bool(
            self._module_latency_cfg.get("enabled", False)
        )
        self._module_latency_resample_interval = None
        self._module_latency_steps_until_resample = None

        self._external_torque_cfg = randomization_cfg.get("external_torque", {}) or {}
        if not isinstance(self._external_torque_cfg, dict):
            self._external_torque_cfg = {}
        self._module_torque_event_specs = self._parse_module_torque_event_specs(
            self._external_torque_cfg
        )
        self._module_torque_event_states: list[dict[str, Any]] = []

        self._module_action_scale_cfg = (
            randomization_cfg.get("module_action_scale", {}) or {}
        )
        if not isinstance(self._module_action_scale_cfg, dict):
            self._module_action_scale_cfg = {}
        self._module_action_scale_event_specs = (
            self._parse_module_action_scale_event_specs(self._module_action_scale_cfg)
        )
        self._module_action_scale_event_states: list[dict[str, Any]] = []

        base_latency_scheme = int(self.sim_cfg.get("latency_scheme", -1))
        self.current_latency_schemes = np.full(
            self.num_joint, base_latency_scheme, dtype=np.int32
        )
        self.current_module_torque_disturbance = np.zeros(
            self.num_joint, dtype=np.float32
        )
        self.current_module_action_scale = np.ones(self.num_joint, dtype=np.float32)

    def _resolve_selected_joint_indices(self, spec: dict[str, Any]) -> list[int]:
        """Resolve target joints from module indices or module ids."""
        raw_indices = spec.get("module_indices", spec.get("joint_indices"))
        if raw_indices is not None:
            if not isinstance(raw_indices, (list, tuple)):
                raw_indices = [raw_indices]
            selected = sorted({int(idx) for idx in raw_indices})
            invalid = [idx for idx in selected if idx < 0 or idx >= self.num_joint]
            if invalid:
                raise ValueError(
                    f"Invalid module_indices/joint_indices {invalid}; valid range is "
                    f"[0, {self.num_joint - 1}]"
                )
            return selected

        raw_module_ids = spec.get("module_ids")
        if raw_module_ids is not None:
            if not isinstance(raw_module_ids, (list, tuple)):
                raw_module_ids = [raw_module_ids]
            if not self._module_id_to_joint_index:
                raise ValueError(
                    "module_ids targeting requires modular joint naming in the MuJoCo model"
                )

            selected = []
            missing_ids = []
            for module_id in raw_module_ids:
                module_id = int(module_id)
                joint_index = self._module_id_to_joint_index.get(module_id)
                if joint_index is None:
                    missing_ids.append(module_id)
                else:
                    selected.append(joint_index)
            if missing_ids:
                raise ValueError(
                    f"Unknown module_ids {missing_ids}; available ids are "
                    f"{sorted(self._module_id_to_joint_index)}"
                )
            return sorted(set(selected))

        return list(range(self.num_joint))

    def _normalize_int_range(
        self, value: Any, *, field_name: str, minimum: int = 0
    ) -> tuple[int, int]:
        """Normalize an integer or [min, max] pair to an inclusive range."""
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError(f"{field_name} must be an int or a [min, max] pair")
            low, high = int(value[0]), int(value[1])
        else:
            low = high = int(value)

        if low > high:
            raise ValueError(f"{field_name} must satisfy min <= max")
        if low < minimum:
            raise ValueError(f"{field_name} must be >= {minimum}")
        return low, high

    def _normalize_float_range(
        self, value: Any, *, field_name: str
    ) -> tuple[float, float]:
        """Normalize a scalar or [min, max] pair to a float range."""
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError(f"{field_name} must be a scalar or a [min, max] pair")
            low, high = float(value[0]), float(value[1])
        else:
            low = high = float(value)

        if low > high:
            raise ValueError(f"{field_name} must satisfy min <= max")
        return low, high

    def _sample_int_from_range(self, value_range: tuple[int, int]) -> int:
        """Sample an integer from an inclusive [min, max] range."""
        low, high = value_range
        if low == high:
            return low
        return int(self.np_random.integers(low, high + 1))

    def _sample_float_from_range(self, value_range: tuple[float, float]) -> float:
        """Sample a float from a [min, max] range."""
        low, high = value_range
        if np.isclose(low, high):
            return float(low)
        return float(self.np_random.uniform(low, high))

    def _normalize_scheduled_module_event_spec(
        self,
        raw_spec: dict[str, Any],
        *,
        value_field: str,
        default_value: Any,
        neutral_value: float,
    ) -> dict[str, Any]:
        """Normalize a scheduled per-module event specification."""
        joint_pool = self._resolve_selected_joint_indices(raw_spec)
        if not joint_pool:
            raise ValueError(f"{value_field} event must target at least one module")

        sample_count = int(raw_spec.get("sample_count", len(joint_pool)))
        if sample_count <= 0:
            raise ValueError("sample_count must be >= 1")
        sample_count = min(sample_count, len(joint_pool))

        spec = {
            "joint_pool": joint_pool,
            "sample_count": sample_count,
            "interval_steps": self._normalize_int_range(
                raw_spec.get("interval_steps", 1),
                field_name="interval_steps",
                minimum=0,
            ),
            "duration_steps": self._normalize_int_range(
                raw_spec.get("duration_steps", 1),
                field_name="duration_steps",
                minimum=1,
            ),
            "activate_on_reset": bool(raw_spec.get("activate_on_reset", False)),
            "neutral_value": float(neutral_value),
            value_field: self._normalize_float_range(
                raw_spec.get(value_field, default_value), field_name=value_field
            ),
        }
        return spec

    def _parse_module_torque_event_specs(
        self, torque_cfg: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Parse per-module torque burst config."""
        if not torque_cfg.get("enabled", False):
            return []

        raw_events = torque_cfg.get("events", []) or []
        if raw_events:
            if not isinstance(raw_events, list):
                raise ValueError("randomization.external_torque.events must be a list")
            specs = []
            for raw_event in raw_events:
                spec = self._normalize_scheduled_module_event_spec(
                    raw_event,
                    value_field="torque_range",
                    default_value=torque_cfg.get(
                        "torque_range", torque_cfg.get("range", [0.0, 0.0])
                    ),
                    neutral_value=0.0,
                )
                spec["sign_mode"] = str(raw_event.get("sign_mode", "random")).lower()
                if spec["sign_mode"] not in {"random", "positive", "negative"}:
                    raise ValueError(
                        "randomization.external_torque sign_mode must be one of "
                        "'random', 'positive', or 'negative'"
                    )
                specs.append(spec)
            return specs

        legacy_spec = {
            "module_indices": torque_cfg.get("module_indices", torque_cfg.get("bodies")),
            "module_ids": torque_cfg.get("module_ids"),
            "torque_range": torque_cfg.get("torque_range", torque_cfg.get("range", [0.0, 0.0])),
            "interval_steps": torque_cfg.get("interval_steps", 1),
            "duration_steps": torque_cfg.get("duration_steps", 1),
            "sign_mode": torque_cfg.get("sign_mode", "random"),
            "activate_on_reset": torque_cfg.get("activate_on_reset", False),
        }
        spec = self._normalize_scheduled_module_event_spec(
            legacy_spec,
            value_field="torque_range",
            default_value=[0.0, 0.0],
            neutral_value=0.0,
        )
        spec["sign_mode"] = str(legacy_spec.get("sign_mode", "random")).lower()
        return [spec]

    def _parse_module_action_scale_event_specs(
        self, scale_cfg: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Parse temporary per-module command attenuation config."""
        if not scale_cfg.get("enabled", False):
            return []

        raw_events = scale_cfg.get("events", []) or []
        if not raw_events:
            raise ValueError(
                "randomization.module_action_scale.enabled=true requires an events list"
            )
        if not isinstance(raw_events, list):
            raise ValueError("randomization.module_action_scale.events must be a list")

        specs = []
        for raw_event in raw_events:
            spec = self._normalize_scheduled_module_event_spec(
                raw_event,
                value_field="scale_range",
                default_value=scale_cfg.get("scale_range", [1.0, 1.0]),
                neutral_value=1.0,
            )
            if spec["scale_range"][0] < 0.0:
                raise ValueError("module_action_scale scale_range must be non-negative")
            specs.append(spec)
        return specs

    def _create_module_event_state(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Create runtime state for a scheduled per-module event."""
        neutral_value = float(spec["neutral_value"])
        return {
            "spec": spec,
            "steps_until_activation": (
                0
                if spec["activate_on_reset"]
                else self._sample_int_from_range(spec["interval_steps"])
            ),
            "remaining_steps": 0,
            "current_values": np.full(self.num_joint, neutral_value, dtype=np.float32),
        }

    def _sample_event_joint_indices(self, spec: dict[str, Any]) -> list[int]:
        """Sample the joint subset affected by an event activation."""
        joint_pool = np.asarray(spec["joint_pool"], dtype=np.int32)
        sample_count = int(spec["sample_count"])
        if sample_count >= len(joint_pool):
            return joint_pool.tolist()
        sampled = self.np_random.choice(joint_pool, size=sample_count, replace=False)
        return sorted(int(idx) for idx in sampled.tolist())

    def _activate_module_event(
        self, event_state: dict[str, Any], *, value_field: str
    ) -> None:
        """Activate one scheduled per-module event."""
        spec = event_state["spec"]
        joint_indices = self._sample_event_joint_indices(spec)
        values = np.full(
            self.num_joint, float(spec["neutral_value"]), dtype=np.float32
        )

        for joint_index in joint_indices:
            sampled_value = self._sample_float_from_range(spec[value_field])
            if value_field == "torque_range":
                sign_mode = spec.get("sign_mode", "random")
                if sign_mode == "negative":
                    sampled_value = -abs(sampled_value)
                elif sign_mode == "positive":
                    sampled_value = abs(sampled_value)
                else:
                    sampled_value = abs(sampled_value)
                    if bool(self.np_random.integers(0, 2)):
                        sampled_value = -sampled_value
            values[joint_index] = sampled_value

        event_state["current_values"] = values
        event_state["remaining_steps"] = self._sample_int_from_range(
            spec["duration_steps"]
        )
        event_state["steps_until_activation"] = 0

    def _advance_module_event_states(
        self,
        event_states: list[dict[str, Any]],
        *,
        value_field: str,
        neutral_value: float,
        combine_mode: str,
    ) -> np.ndarray:
        """Advance scheduled per-module events and return the current vector."""
        if combine_mode == "sum":
            combined = np.zeros(self.num_joint, dtype=np.float32)
        else:
            combined = np.full(self.num_joint, float(neutral_value), dtype=np.float32)

        for event_state in event_states:
            if event_state["remaining_steps"] <= 0 and event_state["steps_until_activation"] <= 0:
                self._activate_module_event(event_state, value_field=value_field)

            if event_state["remaining_steps"] > 0:
                current_values = event_state["current_values"]
                if combine_mode == "sum":
                    combined = combined + current_values
                else:
                    combined = combined * current_values

                event_state["remaining_steps"] -= 1
                if event_state["remaining_steps"] <= 0:
                    neutral = float(event_state["spec"]["neutral_value"])
                    event_state["current_values"] = np.full(
                        self.num_joint, neutral, dtype=np.float32
                    )
                    event_state["steps_until_activation"] = self._sample_int_from_range(
                        event_state["spec"]["interval_steps"]
                    )
            else:
                event_state["steps_until_activation"] = max(
                    0, int(event_state["steps_until_activation"]) - 1
                )

        return combined

    def _sample_module_latency_schemes(self) -> np.ndarray:
        """Sample per-joint latency schemes from config."""
        default_scheme = int(
            self._module_latency_cfg.get(
                "default_scheme", self.sim_cfg.get("latency_scheme", -1)
            )
        )
        if default_scheme not in {-1, 0, 1}:
            raise ValueError(
                "randomization.module_latency.default_scheme must be one of -1, 0, or 1"
            )

        schemes = np.full(self.num_joint, default_scheme, dtype=np.int32)
        for raw_module_cfg in self._module_latency_cfg.get("modules", []) or []:
            joint_indices = self._resolve_selected_joint_indices(raw_module_cfg)
            choices = raw_module_cfg.get("scheme_choices", raw_module_cfg.get("schemes"))
            if choices is None:
                choices = [raw_module_cfg.get("scheme", default_scheme)]
            if not isinstance(choices, (list, tuple)):
                choices = [choices]

            normalized_choices = [int(choice) for choice in choices]
            invalid_choices = sorted(
                choice for choice in set(normalized_choices) if choice not in {-1, 0, 1}
            )
            if invalid_choices:
                raise ValueError(
                    "randomization.module_latency only supports latency schemes -1, 0, "
                    f"and 1 per module, got {invalid_choices}"
                )

            sample_each_module = bool(raw_module_cfg.get("sample_each_module", False))
            if sample_each_module:
                sampled = self.np_random.choice(normalized_choices, size=len(joint_indices))
                for joint_index, scheme in zip(joint_indices, sampled.tolist()):
                    schemes[joint_index] = int(scheme)
            else:
                sampled_scheme = int(self.np_random.choice(normalized_choices))
                schemes[joint_indices] = sampled_scheme

        return schemes

    def _reset_asymmetric_randomization_state(self) -> None:
        """Reset all per-module randomization state for a new episode."""
        self.current_module_torque_disturbance.fill(0.0)
        self.current_module_action_scale.fill(1.0)
        self._module_torque_event_states = [
            self._create_module_event_state(spec)
            for spec in self._module_torque_event_specs
        ]
        self._module_action_scale_event_states = [
            self._create_module_event_state(spec)
            for spec in self._module_action_scale_event_specs
        ]

        if self._module_latency_enabled:
            self.current_latency_schemes = self._sample_module_latency_schemes()
            resample_interval = self._module_latency_cfg.get(
                "resample_interval_steps", None
            )
            if resample_interval is None:
                self._module_latency_resample_interval = None
                self._module_latency_steps_until_resample = None
            else:
                self._module_latency_resample_interval = self._normalize_int_range(
                    resample_interval,
                    field_name="module_latency.resample_interval_steps",
                    minimum=1,
                )
                self._module_latency_steps_until_resample = self._sample_int_from_range(
                    self._module_latency_resample_interval
                )
        else:
            self.current_latency_schemes.fill(int(self.sim_cfg.get("latency_scheme", -1)))
            self._module_latency_resample_interval = None
            self._module_latency_steps_until_resample = None

    def _update_asymmetric_randomization_state(self) -> None:
        """Advance per-module latency and event randomization for one control step."""
        if self._module_latency_enabled and self._module_latency_steps_until_resample is not None:
            if self._module_latency_steps_until_resample <= 0:
                self.current_latency_schemes = self._sample_module_latency_schemes()
                self._module_latency_steps_until_resample = self._sample_int_from_range(
                    self._module_latency_resample_interval
                )
            else:
                self._module_latency_steps_until_resample -= 1

        if self._module_torque_event_states:
            self.current_module_torque_disturbance = self._advance_module_event_states(
                self._module_torque_event_states,
                value_field="torque_range",
                neutral_value=0.0,
                combine_mode="sum",
            )
        else:
            self.current_module_torque_disturbance.fill(0.0)

        if self._module_action_scale_event_states:
            self.current_module_action_scale = self._advance_module_event_states(
                self._module_action_scale_event_states,
                value_field="scale_range",
                neutral_value=1.0,
                combine_mode="product",
            )
        else:
            self.current_module_action_scale.fill(1.0)

    def _setup_initialization_parameters(self) -> None:
        """Setup robot initialization parameters."""
        # Get initialization config
        self.init_cfg = getattr(self.cfg, "initialization", {})

        # Position and orientation
        self.init_pos = self.init_cfg.get("init_pos", [0, 0, 0.1])
        self.init_joint_pos = self.init_cfg.get("init_joint_pos", 0)
        if self.init_joint_pos is None:
            self.init_joint_pos = self.cfg.control.default_dof_pos  # type: ignore

        if isinstance(self.init_joint_pos, (list, np.ndarray)):
            self.init_joint_pos = np.array(self.init_joint_pos)
        else:
            self.init_joint_pos = np.full(self.num_joint, self.init_joint_pos)
        self.given_init_qpos = self.init_cfg.get("init_qpos")

        # Calculate initial quaternion
        self._calculate_initial_quaternion()

        # Forward vector
        forward_vec = self.cfg.observation.get("forward_vec")  # type: ignore
        self.forward_vec = np.array(forward_vec) if forward_vec else None

    def _calculate_initial_quaternion(self) -> None:
        """Calculate initial robot orientation quaternion."""
        init_quat_cfg = self.init_cfg.get("init_quat", "x")
        self.theta = 0.610865  # Default theta value
        lleg_vec = np.array([0, np.cos(self.theta), np.sin(self.theta)])

        if is_list_like(init_quat_cfg):
            if len(init_quat_cfg) == 4:
                self.init_quat = np.array(init_quat_cfg)
            elif len(init_quat_cfg) == 3:
                self.init_quat = quaternion_from_vectors(
                    lleg_vec, np.array(init_quat_cfg)
                )
            else:
                raise ValueError("init_quat list must have 3 or 4 elements")
        elif init_quat_cfg == "x":
            self.init_quat = quaternion_from_vectors(lleg_vec, np.array([1, 0, 0]))
        elif init_quat_cfg == "y":
            self.init_quat = quaternion_from_vectors(lleg_vec, np.array([0, 1, 0]))
        else:
            raise ValueError("init_quat must be list of 3/4 elements or 'x'/'y'")

    def _reset_external_forces(self) -> None:
        """Reset external force applications."""
        self.data.qfrc_applied = np.zeros(self.model.nv)
        if hasattr(self, "external_force_counter"):
            self.external_force_counter = dict.fromkeys(self.external_force_counter, 0)

    def _apply_force(
        self, force: float, body: str, position: list[float], direction: list[float]
    ) -> None:
        """Apply external force to specified body.

        Args:
            force: Force magnitude
            body: Body name
            position: Force application point in local coordinates
            direction: Force direction in global coordinates
        """
        body_id = self.model.body(body).id
        rotation_matrix = self.data.xmat[body_id].reshape(3, 3)
        body_pos = self.data.xpos[body_id]

        # Transform to global coordinates
        force_global = np.array(direction) * force
        point_global = body_pos + rotation_matrix @ np.array(position)

        # Apply force
        torque = np.zeros(3)
        qfrc_result = np.zeros(len(self.data.qvel))
        mujoco.mj_applyFT(
            self.model,
            self.data,
            force_global,
            torque,
            point_global,
            body_id,
            qfrc_result,
        )
        self.data.qfrc_applied = qfrc_result

    def _perform_action(self, action: np.ndarray) -> dict[str, Any]:
        """Execute action with comprehensive control pipeline.

        Args:
            action: Processed action from action processor

        Returns:
            Action execution information
        """
        self._update_asymmetric_randomization_state()

        # Apply external forces
        if self.external_forces_enabled:
            self._handle_external_forces()

        # Validate frame skip for latency
        latency_scheme: Union[int, np.ndarray]
        if self._module_latency_enabled:
            latency_scheme = self.current_latency_schemes.copy()
            latency_requires_even_skip = np.any(latency_scheme >= 0)
        else:
            latency_scheme = int(self.sim_cfg.get("latency_scheme", -1))
            latency_requires_even_skip = latency_scheme >= 0

        if latency_requires_even_skip and self.frame_skip % 2 != 0:
            raise ValueError("frame_skip must be even for latency simulation")

        # Prepare control targets (separate position/velocity, add noise if enabled)
        pos, vel = self._prepare_control_targets(action)

        # Record positions before action
        pos_before = self._record_positions()

        # Execute control with latency simulation
        self._execute_control(pos, vel, latency_scheme)

        # Capture frame for video recording if needed
        self.render()

        # Update latency tracking
        self._update_latency_tracking(pos, vel)

        # Record positions after action
        pos_after = self._record_positions()

        return self._create_action_info(pos_before, pos_after)

    def _handle_external_forces(self) -> None:
        """Handle external force application."""
        config = self.external_force_config

        if self.step_count % config["interval"] == 0:
            for _i, (force_range, body, position, direction) in enumerate(
                zip(
                    config["ranges"],
                    config["bodies"],
                    config["positions"],
                    config["directions"],
                )
            ):
                force = np.random.uniform(*force_range)
                self._apply_force(force, body, position, direction)

        # Update force duration counters
        for i, duration in enumerate(config["durations"]):
            self.external_force_counter[i] += 1
            if self.external_force_counter[i] >= duration:
                self._reset_external_forces()

    def _prepare_control_targets(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Prepare position and velocity control targets from actions.
        
        This method:
        1. Separates actions into position/velocity targets based on per-joint control modes
        2. Adds noise if enabled
        
        For position-controlled joints:
            - pos_target = action
            - vel_target = 0
            
        For velocity-controlled joints (wheels):
            - pos_target = 0 (will be multiplied by kp=0)
            - vel_target = action * wheel_action_scale + default_velocity

        Args:
            action: Raw action array from action processor
            
        Returns:
            (pos_target, vel_target): Tuple of position and velocity targets
        """
        pos = action.copy()
        vel = np.zeros_like(pos)

        # Handle per-joint control modes (position vs velocity)
        if self.joint_control_modes is not None:
            for i, mode in enumerate(self.joint_control_modes):
                if mode == 'velocity':
                    # For wheel joints: action is velocity target
                    vel[i] = (action[i] * self.wheel_action_scale 
                              + self.default_dof_vel[i])
                    pos[i] = 0  # Position target is 0 (will be multiplied by kp=0)

        if getattr(self, "current_module_action_scale", None) is not None:
            pos = pos * self.current_module_action_scale
            vel = vel * self.current_module_action_scale

        # Add noise if enabled
        if self.sim_cfg.get("noisy_actions", False):
            noise_std = self.sim_cfg.action_noise_std
            pos += self.np_random.normal(0, noise_std, size=pos.shape)
            vel += self.np_random.normal(0, noise_std, size=vel.shape)

        return pos, vel

    def _record_positions(self) -> dict[str, np.ndarray]:
        """Record current robot positions."""
        return {
            "base_pos": self.data.qpos.flat[:2].copy(),
            "general_coords": np.array(
                [self.data.qpos.flat[i : i + 2] for i in self.free_joint_addr]
            ).reshape(-1, 2),
        }

    def _execute_control(
        self, pos: np.ndarray, vel: np.ndarray, latency_scheme: Union[int, np.ndarray]
    ) -> None:
        """Execute control with latency simulation."""
        if isinstance(latency_scheme, np.ndarray):
            latency_scheme = np.asarray(latency_scheme, dtype=np.int32)
            if latency_scheme.shape != pos.shape:
                raise ValueError(
                    f"Per-module latency shape mismatch: expected {pos.shape}, got "
                    f"{latency_scheme.shape}"
                )
            if np.any(latency_scheme == -2):
                raise ValueError(
                    "Per-module latency does not support scheme -2. Use -1, 0, or 1."
                )
            if np.all(latency_scheme == -1):
                self._pd_control(pos, self.frame_skip, vel)
                return

            half_skip = self.frame_skip // 2
            first_pos = pos.copy()
            first_vel = vel.copy()
            second_pos = pos.copy()
            second_vel = vel.copy()

            one_step_mask = latency_scheme == 0
            two_step_mask = latency_scheme == 1

            first_pos[one_step_mask] = self.last_pos_sim[one_step_mask]
            first_vel[one_step_mask] = self.last_vel_sim[one_step_mask]
            first_pos[two_step_mask] = self.last_last_pos_sim[two_step_mask]
            first_vel[two_step_mask] = self.last_last_vel_sim[two_step_mask]
            second_pos[two_step_mask] = self.last_pos_sim[two_step_mask]
            second_vel[two_step_mask] = self.last_vel_sim[two_step_mask]

            self._pd_control(first_pos, half_skip, first_vel)
            self._pd_control(second_pos, half_skip, second_vel)
            return

        if latency_scheme == -1:
            # No latency
            self._pd_control(pos, self.frame_skip, vel)
        elif latency_scheme == 0:
            # One step latency
            half_skip = self.frame_skip // 2
            self._pd_control(self.last_pos_sim, half_skip, self.last_vel_sim)
            self._pd_control(pos, half_skip, vel)
        elif latency_scheme == 1:
            # Two step latency
            half_skip = self.frame_skip // 2
            self._pd_control(self.last_last_pos_sim, half_skip, self.last_last_vel_sim)
            self._pd_control(self.last_pos_sim, half_skip, self.last_vel_sim)
        elif latency_scheme == -2:
            # Fine control with latency
            self._pd_control_fine(
                self.last_pos_sim, self.frame_skip // 2, self.last_vel_sim
            )

    def _update_latency_tracking(self, pos: np.ndarray, vel: np.ndarray) -> None:
        """Update latency simulation tracking variables."""
        self.last_last_pos_sim = self.last_pos_sim.copy()
        self.last_pos_sim = pos.copy()
        self.last_last_vel_sim = self.last_vel_sim.copy()
        self.last_vel_sim = vel.copy()

    def _create_action_info(self, pos_before: dict, pos_after: dict) -> dict[str, Any]:
        """Create action execution info dictionary."""
        info = {
            "coordinates": pos_before["base_pos"],
            "next_coordinates": pos_after["base_pos"],
            "coordinates_general": pos_before["general_coords"],
            "next_coordinates_general": pos_after["general_coords"],
        }

        # Add render data if needed
        if getattr(self, "render_on_bg", False):
            rendered_frame = self.render()
            if rendered_frame is not None:
                info["render"] = rendered_frame.transpose(2, 0, 1)

        return info

    def _pd_control(
        self,
        pos_desired: np.ndarray,
        frame_skip: int,
        vel_desired: Optional[np.ndarray] = None,
    ) -> None:
        """Execute PD control for joint positions with mixed control mode support.

        For position-controlled joints:
            torque = kp * (pos_desired - pos_current) + kd * (vel_desired - vel_current)
        
        For velocity-controlled joints (wheels):
            torque = kd * (vel_target - vel_current)  (kp = 0)
            
        Note: The separation of actions into position/velocity targets for wheel
        joints is handled in _prepare_control_targets(), which sets:
        - pos_desired[i] = 0 for wheel joints (multiplied by kp=0)
        - vel_desired[i] = velocity target for wheel joints

        Args:
            pos_desired: Target joint positions (0 for velocity-controlled joints)
            frame_skip: Number of simulation steps
            vel_desired: Target joint velocities (velocity target for wheel joints)
        """
        if vel_desired is None:
            vel_desired = np.zeros_like(pos_desired)

        # Get current joint states
        dof_pos = self.data.qpos[self.model.jnt_qposadr[self.joint_idx]]
        dof_vel = self.data.qvel[self.model.jnt_dofadr[self.joint_idx]]

        # Calculate PD torques
        # For position joints: kp * (pos_error) + kd * (vel_error)
        # For velocity joints: kp=0, so just kd * (vel_target - vel_current)
        torques = self.kps * (pos_desired - dof_pos) + self.kds * (
            vel_desired - dof_vel
        )

        # Apply constraints
        torques = self._apply_torque_constraints(torques, dof_vel)
        # print(f"Applying torques: {torques}")

        # Execute simulation
        self.do_simulation(torques, int(frame_skip))

    def _apply_torque_constraints(
        self, torques: np.ndarray, dof_vel: np.ndarray
    ) -> np.ndarray:
        """Apply torque and velocity constraints."""
        # Torque-velocity constraints
        if self.sim_cfg.get("tn_constraint", True):
            torque_limits = self._calculate_torque_limits(dof_vel)
            torques = np.clip(torques, -torque_limits, torque_limits)

        # Handle disabled motors
        broken_motors = self.sim_cfg.get("broken_motors")
        if broken_motors is not None:
            torques[broken_motors] = 0

        if getattr(self, "current_module_torque_disturbance", None) is not None:
            torques = torques + self.current_module_torque_disturbance

        return torques

    def _calculate_torque_limits(self, dof_vel: np.ndarray) -> np.ndarray:
        """Calculate velocity-dependent torque limits."""
        abs_vel = np.abs(dof_vel)
        per_joint_cfg = getattr(self, "_actuation_cfg_per_joint", None)
        if per_joint_cfg:
            if len(per_joint_cfg) != len(abs_vel):
                raise ValueError(
                    f"Per-joint actuation config length {len(per_joint_cfg)} does not match "
                    f"number of actuated joints {len(abs_vel)}"
                )
            return np.asarray(
                [
                    self._calculate_torque_limits_for_cfg(float(v), cfg)
                    for v, cfg in zip(abs_vel, per_joint_cfg)
                ],
                dtype=np.float64,
            )

        cfg = getattr(self, "_actuation_cfg", {}) or {}
        return self._calculate_torque_limits_for_cfg(abs_vel, cfg)

    def _calculate_torque_limits_for_cfg(
        self,
        abs_vel: Union[float, np.ndarray],
        cfg: dict[str, Any],
    ) -> np.ndarray:
        """Calculate torque limits for one actuation config."""
        model = str(cfg.get("model", "legacy_piecewise_linear")).lower()

        if model in {"legacy", "legacy_piecewise_linear"}:
            return np.where(
                abs_vel < 11.5, 12.0, np.clip(-0.656 * abs_vel + 19.541, 0, None)
            )

        if model == "none":
            return np.full_like(abs_vel, np.inf, dtype=np.float64)

        if model == "constant":
            max_torque = float(cfg.get("max_torque", 12.0))
            return np.full_like(abs_vel, max_torque, dtype=np.float64)

        if model == "linear":
            max_torque = float(cfg.get("max_torque", 12.0))
            max_velocity = float(cfg.get("max_velocity", 19.541 / 0.656))
            if max_velocity <= 0:
                raise ValueError("simulation.actuation.max_velocity must be > 0 for linear model")
            return np.clip(max_torque * (1.0 - abs_vel / max_velocity), 0.0, None)

        if model == "piecewise_linear":
            max_torque = float(cfg.get("max_torque", 12.0))
            knee_velocity = float(cfg.get("knee_velocity", 11.5))
            max_velocity = float(cfg.get("max_velocity", 19.541 / 0.656))
            knee_torque = float(cfg.get("knee_torque", max_torque))
            min_torque = float(cfg.get("min_torque", 0.0))

            if knee_velocity < 0:
                raise ValueError("simulation.actuation.knee_velocity must be >= 0")
            if max_velocity < knee_velocity:
                raise ValueError(
                    "simulation.actuation.max_velocity must be >= knee_velocity"
                )

            torque_limits = np.full_like(abs_vel, max_torque, dtype=np.float64)
            if max_velocity == knee_velocity:
                torque_limits[abs_vel >= knee_velocity] = min_torque
                return np.clip(torque_limits, 0.0, None)

            after_knee = abs_vel > knee_velocity
            torque_limits[after_knee] = np.interp(
                abs_vel[after_knee],
                [knee_velocity, max_velocity],
                [knee_torque, min_torque],
                left=knee_torque,
                right=min_torque,
            )
            return np.clip(torque_limits, 0.0, None)

        if model == "lookup_table":
            velocity_breakpoints = np.asarray(
                cfg.get("velocity_breakpoints", []), dtype=np.float64
            )
            torque_limits = np.asarray(cfg.get("torque_limits", []), dtype=np.float64)
            if velocity_breakpoints.ndim != 1 or torque_limits.ndim != 1:
                raise ValueError("simulation.actuation lookup_table arrays must be 1D")
            if len(velocity_breakpoints) < 2 or len(velocity_breakpoints) != len(torque_limits):
                raise ValueError(
                    "simulation.actuation lookup_table requires equal-length arrays with at least 2 entries"
                )
            if np.any(np.diff(velocity_breakpoints) < 0):
                raise ValueError(
                    "simulation.actuation.velocity_breakpoints must be nondecreasing"
                )
            return np.clip(
                np.interp(
                    abs_vel,
                    velocity_breakpoints,
                    torque_limits,
                    left=torque_limits[0],
                    right=torque_limits[-1],
                ),
                0.0,
                None,
            )

        raise ValueError(f"Unknown simulation.actuation.model '{model}'")

    def _pd_control_fine(
        self,
        pos_desired: np.ndarray,
        frame_skip: int,
        vel_desired: Optional[np.ndarray] = None,
    ) -> None:
        """Execute fine-grained PD control with per-step updates.
        
        Supports mixed control modes (position/velocity) per joint.
        The separation of actions into position/velocity targets is handled
        in _prepare_control_targets().
        """
        if vel_desired is None:
            vel_desired = np.zeros_like(pos_desired)

        for _ in range(int(frame_skip)):
            dof_pos = self.data.qpos[self.model.jnt_qposadr[self.joint_idx]]
            dof_vel = self.data.qvel[self.model.jnt_dofadr[self.joint_idx]]

            torques = self.kps * (pos_desired - dof_pos) + self.kds * (
                vel_desired - dof_vel
            )
            torques = self._apply_torque_constraints(torques, dof_vel)

            self.do_simulation(torques, 1)

    def _get_observable_data(self) -> dict[str, Any]:
        """Get comprehensive observable state data."""
        # Extract basic state
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()

        # Core state components
        state_data = self._extract_core_state(qpos, qvel)

        # Sensor data
        self._update_sensor_readings()
        sensor_data = self._extract_sensor_data()
        sensor_module_count = len(self.joint_idx)
        if sensor_module_count <= 0:
            sensor_module_count = self.num_act
        if sensor_module_count <= 0:
            sensor_module_count = 1

        if "quats" not in sensor_data or np.asarray(sensor_data["quats"]).size == 0:
            # Generic XMLs may not define per-module IMU sensors.
            sensor_data["quats"] = np.repeat(
                state_data["quat"][None, :], sensor_module_count, axis=0
            )

        sensor_data["gyros"] = self._coerce_sensor_matrix(
            sensor_data.get("gyros"),
            rows=sensor_module_count,
            cols=3,
            fill_value=0.0,
        )
        sensor_data["accs"] = self._coerce_sensor_matrix(
            sensor_data.get("accs"),
            rows=sensor_module_count,
            cols=3,
            fill_value=0.0,
        )

        # Extended data with noise handling
        extended_data = self._create_extended_state_data({**state_data, **sensor_data})

        # Simulation-specific data
        sim_data = self._extract_simulation_data()
        goal_data = self._get_goal_observable_data(state_data["pos_world"])

        return {**extended_data, **sim_data, **goal_data}

    def _extract_core_state(self, qpos: np.ndarray, qvel: np.ndarray) -> dict[str, Any]:
        """Extract core robot state information."""
        # Position and orientation
        torso_body_id = self.torso_body_id

        self.pos_world = self.data.xpos[torso_body_id]
        quat = self.data.xquat[torso_body_id]
        quat = wxyz_to_xyzw(quat)
        pos_worlds = [self.data.xpos[i] for i in self.joint_body_idx]

        # Joint states
        dof_pos = qpos[self.model.jnt_qposadr[self.joint_idx]]
        dof_vel = qvel[self.model.jnt_dofadr[self.joint_idx]]

        # Velocities
        cvel = self.data.cvel[torso_body_id]
        vel_world = cvel[3:]
        vel_body = quat_rotate_inverse(quat, vel_world)
        ang_vel_world = cvel[:3]
        ang_vel_body = quat_rotate_inverse(quat, ang_vel_world)
        
        return {
            "pos_world": self.pos_world,
            "pos_worlds": pos_worlds,
            "quat": quat,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "vel_world": vel_world,
            "vel_body": vel_body,
            "ang_vel_body": ang_vel_body,
            "ang_vel_world": ang_vel_world,
            "qpos": qpos,
            "qvel": qvel,
        }

    def _update_sensor_readings(self) -> None:
        """Update all sensor readings from simulation."""
        # Rebuild every step to avoid stale keys when sensors are absent.
        self.sensors.clear()

        sensor_specs = [
            ("quat", 4),
            ("gyro", 3),
            ("vel", 3),
            ("globvel", 3),
            ("back_quat", 4),
            ("back_gyro", 3),
            ("back_vel", 3),
            ("acc", 3),
        ]

        for sensor_type, size in sensor_specs:
            try:
                prefix = "back_imu_" if sensor_type.startswith("back_") else "imu_"
                clean_type = sensor_type.replace("back_", "")

                sensor_data = []
                for module_id in self.jointed_module_ids:
                    sensor_name = f"{prefix}{clean_type}{module_id}"
                    start_addr = self.model.sensor(sensor_name).adr[0]
                    sensor_data.append(
                        self.data.sensordata[start_addr : start_addr + size]
                    )

                # Only keep non-empty sensor arrays. Generic XML assets may have no IMU sensors.
                if sensor_data:
                    self.sensors[sensor_type] = np.array(sensor_data)
            except (KeyError, IndexError):
                continue

    def _extract_sensor_data(self) -> dict[str, Any]:
        """Extract sensor data for observations."""
        sensor_data = {}

        if "quat" in self.sensors:
            sensor_data["quats"] = np.array(
                [wxyz_to_xyzw(q) for q in self.sensors["quat"]]
            )
        if "gyro" in self.sensors:
            sensor_data["gyros"] = self.sensors["gyro"]
        if "acc" in self.sensors:
            sensor_data["accs"] = self.sensors["acc"]

        return sensor_data

    def _coerce_sensor_matrix(
        self,
        data: Optional[np.ndarray],
        rows: int,
        cols: int,
        fill_value: float = 0.0,
    ) -> np.ndarray:
        """Convert sensor data to fixed (rows, cols) shape via crop/pad/fill."""
        out = np.full((rows, cols), fill_value, dtype=np.float32)
        if data is None:
            return out

        arr = np.asarray(data, dtype=np.float32)
        if arr.size == 0:
            return out

        if arr.ndim == 1:
            if arr.shape[0] == cols:
                arr = arr.reshape(1, cols)
            else:
                return out

        if arr.ndim != 2 or arr.shape[1] != cols:
            return out

        n = min(rows, arr.shape[0])
        out[:n] = arr[:n]
        return out

    def _create_extended_state_data(self, base_data: dict[str, Any]) -> dict[str, Any]:
        """Create extended data with accurate versions and noise."""
        extended = base_data.copy()

        # Add accurate (noise-free) versions
        for key, value in base_data.items():
            extended[f"accurate_{key}"] = copy.deepcopy(value)

            # Add observation noise if enabled
            if self.sim_cfg.get("noisy_observations", False) and isinstance(
                value, np.ndarray
            ):
                noise_std = self.sim_cfg.obs_noise_std
                extended[key] = value + self.np_random.normal(
                    0, noise_std, size=value.shape
                )

        return extended

    def _extract_simulation_data(self) -> dict[str, Any]:
        """Extract simulation-specific data."""
        sim_data = {
            "mj_data": self.data,
            "mj_model": self.model,
            "adjusted_forward_vec": getattr(
                self, "adjusted_forward_vec", self.forward_vec
            ),
        }

        # Contact information
        sim_data.update(self._extract_contact_data())

        # Additional sensor data
        if "vel" in self.sensors:
            sim_data["vels"] = self.sensors["vel"]
        if "back_vel" in self.sensors:
            sim_data["back_vels"] = self.sensors["back_vel"]
        if "back_quat" in self.sensors:
            sim_data["back_quats"] = np.array(
                [wxyz_to_xyzw(q) for q in self.sensors["back_quat"]]
            )
        if "back_gyro" in self.sensors:
            sim_data["back_gyros"] = self.sensors["back_gyro"]

        # Center of mass data
        sim_data.update(self._calculate_com_data())

        return sim_data

    def _extract_contact_data(self) -> dict[str, Any]:
        """Extract contact information."""
        floor_id_set = set(getattr(self, "floor_geom_ids", [0]))
        contacts = [tuple(int(g) for g in c.geom) for c in self.data.contact]
        floor_contacts = [
            pair
            for pair in contacts
            if (pair[0] in floor_id_set) or (pair[1] in floor_id_set)
        ]
        non_floor_contact_geoms = {
            geom for pair in floor_contacts for geom in pair if geom not in floor_id_set
        }

        # Count joint-floor contacts
        joint_floor_count = sum(
            (contact[0] in self.joint_geom_idx) or (contact[1] in self.joint_geom_idx)
            for contact in floor_contacts
        )

        return {
            "contact_geoms": contacts,
            "num_jointfloor_contact": joint_floor_count,
            "contact_floor_geoms": list(non_floor_contact_geoms),
            "contact_floor_socks": list(
                {
                    geom
                    for geom in non_floor_contact_geoms
                    if self.model.geom(geom).name.startswith("sock")
                }
            ),
            "contact_floor_balls": list(
                {
                    geom
                    for geom in non_floor_contact_geoms
                    if (
                        self.model.geom(geom).name.startswith("left")
                        or self.model.geom(geom).name.startswith("right")
                    )
                }
            ),
        }

    def _calculate_com_data(self) -> dict[str, Any]:
        """Calculate center of mass data."""
        com_pos = np.mean(self.data.xpos[self.joint_body_idx], axis=0)
        com_vel_world = (com_pos - self.last_com_pos) / self.dt
        self.last_com_pos = com_pos.copy()

        return {"com_vel_world": com_vel_world}

    def _reset_robot(self) -> None:
        """Reset robot with domain randomization."""
        # Reset MuJoCo model
        self.reset_model()

        # Reset control tracking
        self.last_pos_sim = self.default_dof_pos.copy()
        self.last_last_pos_sim = self.default_dof_pos.copy()
        self.last_vel_sim = np.zeros(self.num_joint)
        self.last_last_vel_sim = np.zeros(self.num_joint)
        self.last_com_pos = np.zeros(3)

        # Initialize rendering filter
        self.render_lookat_filter = AverageFilter(10)

    def reset_model(self) -> np.ndarray:
        """Reset MuJoCo model with comprehensive domain randomization."""

        # Check if this episode should be recorded
        if self._should_record_episode():
            self.start_video_recording()

        # Pre-reset operations
        self._pre_reset()

        # Handle model reloading if needed
        if self._need_model_reload():
            self._reload_model_with_randomization()

        # Reset external forces
        self._reset_external_forces()

        # Apply domain randomization
        self._apply_domain_randomization()

        # Set initial state
        self._set_initial_state()

        # Post-reset operations
        self._post_reset()

        # Return empty observation as placeholder for parent class compatibility
        return np.array([])

    def _pre_reset(self) -> None:
        """Pre-reset operations."""
        pass

    def _post_reset(self) -> None:
        """Post-reset operations."""
        self._reset_goal_task()

    def _post_done(self) -> None:
        """Post-done operations after an episode ends."""
        # Stop video recording if active
        if self.render_mode == "mp4" and self.recording_active:
            success = self.stop_video_recording()
            if success:
                print(f"Episode {self.episode_counter} video recording completed")
            else:
                print(f"Episode {self.episode_counter} video recording failed")
        self.episode_counter += 1

    def _need_model_reload(self) -> bool:
        """Check if model needs to be reloaded for randomization."""
        self.randomize_asset = is_list_like(self.cfg.morphology.asset_file)  # type: ignore
        morphology_enabled = self._has_morphology_randomization()
        
        # Check randomization config (new style: cfg.randomization.*)
        randomization_cfg = getattr(self.cfg, "randomization", {})
        mass_enabled = randomization_cfg.get("mass", {}).get("enabled", False)
        damping_enabled = randomization_cfg.get("damping", {}).get("enabled", False)
        
        # Fallback to old style for backward compatibility
        if not mass_enabled:
            mass_enabled = self.sim_cfg.get("randomize_mass", False)
        if not damping_enabled:
            damping_enabled = self.sim_cfg.get("randomize_damping", False)
        
        return any(
            [
                mass_enabled,
                damping_enabled,
                self.sim_cfg.get("add_scaffold_walls", False),
                self.randomize_asset,
                morphology_enabled,
            ]
        )

    def _reload_model_with_randomization(self) -> None:
        """Reload model with XML-level randomization applied.

        For asset-based robots with multiple XML files, this may select a new
        source asset. For morphology-built robots, this either rebuilds the
        XML from a randomized copy of the base morphology graph or reuses the
        existing XMLCompiler and mutates it in place before reloading the
        MuJoCo model.
        """
        # Asset randomization (for asset-based robots with multiple asset files)
        if self.randomize_asset:
            self._load_robot_asset()
        elif self._has_morphology_randomization():
            randomized_morphology = self._get_randomized_morphology_configuration()
            self._load_robot_asset_from_morphology(
                self.cfg.morphology.robot_type, randomized_morphology
            )
        # Morphology-built robots normally already have an XMLCompiler from
        # _load_robot_asset_from_morphology(). If not, there is no XML to
        # mutate for mass/damping randomization, so skip this reload path.
        elif not hasattr(self, 'xml_compiler') or self.xml_compiler is None:
            return

        # Get randomization config (new style)
        randomization_cfg = getattr(self.cfg, "randomization", {})
        
        # Mass randomization
        mass_cfg = randomization_cfg.get("mass", {})
        mass_enabled = mass_cfg.get("enabled", False)
        
        # Fallback to old style for backward compatibility
        if not mass_enabled:
            mass_enabled = self.sim_cfg.get("randomize_mass", False)
        
        if mass_enabled and hasattr(self, 'mass_range') and hasattr(self, 'xml_compiler'):
            # Get mass offset from new or old config
            mass_offset = mass_cfg.get("offset", None)
            if mass_offset is None:
                mass_offset = self.sim_cfg.get("mass_offset", 0)
            
            mass_dict = {
                key: np.random.uniform(*value) + mass_offset
                for key, value in self.mass_range.items()
            }
            self.xml_compiler.update_mass(mass_dict)

        # Damping randomization
        damping_cfg = randomization_cfg.get("damping", {})
        damping_enabled = damping_cfg.get("enabled", False)
        
        # Fallback to old style for backward compatibility
        if not damping_enabled:
            damping_enabled = self.sim_cfg.get("randomize_damping", False)
        
        if damping_enabled and hasattr(self, 'xml_compiler'):
            # Get ranges from new or old config
            damping_range = damping_cfg.get("range", None)
            armature_range = damping_cfg.get("armature_range", None)
            
            if damping_range is None:
                damping_range = self.sim_cfg.get("random_damping_range", [0.02, 0.2])
            if armature_range is None:
                armature_range = self.sim_cfg.get("random_armature_range", [0.01, 0.05])
            
            armature = np.random.uniform(*armature_range)
            damping = np.random.uniform(*damping_range)
            self.xml_compiler.update_damping(armature=armature, damping=damping)

        # Scaffold walls
        if self.sim_cfg.get("add_scaffold_walls", False):
            self.xml_compiler.remove_walls()
            angle = quaternion_to_euler(self.init_quat)[0] * 180 / np.pi - 90
            self.xml_compiler.add_walls(transparent=False, angle=angle)

        # Reload MuJoCo environment
        self.reload_model(self.xml_compiler.get_string())

        # SAve reloaded model to log directory
        if self._log_dir is not None:
            self.xml_compiler.save(os.path.join(self._log_dir, "reloaded_robot_debug.xml"))

    def _apply_domain_randomization(self) -> None:
        """Apply domain randomization parameters."""
        # PD controller randomization
        randomization_cfg = getattr(self.cfg, "randomization", {})
        pd_cfg = randomization_cfg.get("pd_controller", {})
        if pd_cfg.get("enabled", False):
            self.kp = np.random.uniform(*pd_cfg.get("kp_range", [8.0, 8.0]))
            self.kd = np.random.uniform(*pd_cfg.get("kd_range", [0.2, 0.2]))
            
            # Preserve per-joint control settings when randomizing
            if self.joint_control_modes is not None:
                # Only randomize position-controlled joints
                for i, mode in enumerate(self.joint_control_modes):
                    if mode == 'position':
                        self.kps[i] = self.kp
                        self.kds[i] = self.kd
                    # Velocity-controlled joints keep their original kd
            else:
                # Original behavior: uniform gains
                self.kps = np.full(self.num_act * self.num_envs, self.kp, dtype=np.float32)
                self.kds = np.full(self.num_act * self.num_envs, self.kd, dtype=np.float32)

        # Latency randomization
        if self.sim_cfg.get("random_latency_scheme", False):
            self.sim_cfg.latency_scheme = np.random.randint(0, 2)

        # Friction randomization
        self._randomize_friction()

        # Per-module latency and event randomization
        self._reset_asymmetric_randomization_state()

    def _randomize_friction(self) -> None:
        """Apply friction randomization."""
        randomization_cfg = getattr(self.cfg, "randomization", {})
        friction_cfg = randomization_cfg.get("friction", {})
        if not friction_cfg.get("enabled", False):
            return

        friction_range = friction_cfg.get("range", [0.8, 1.2])
        rolling_friction_range = friction_cfg.get("rolling_range", [0.01, 0.1])
        detailed_range = friction_cfg.get("detailed_range", None)

        if is_number(friction_range[0]):
            # Single friction value
            friction = np.random.uniform(*friction_range)
            self.model.geom("floor").friction[0] = friction
            self.model.geom("floor").priority[0] = 10
            roll_friction = np.random.uniform(*rolling_friction_range)
            self.model.geom("floor").friction[2] = 0.01 # roll_friction
            self.model.geom("floor").friction[1] = roll_friction
            # print(f"Applied floor rolling friction: {roll_friction:.4f}")
        else:
            # Separate friction for different components
            # TODO: This is buggy
            stick_friction = np.random.uniform(*detailed_range[0])
            ball_friction = np.random.uniform(*detailed_range[1])
            roll_friction = np.random.uniform(*rolling_friction_range)

            passive_component_geoms = [self.model.geom(i).name for i in range(self.model.ngeom)]
            passive_component_geoms.remove("floor")

            for module_id in self.jointed_module_ids:
                self.model.geom(f"left{module_id}").friction[0] = 1
                self.model.geom(f"right{module_id}").friction[0] = 1
                # self.model.geom(f"stick{module_id}").friction[0] = stick_friction
                
                self.model.geom(f"left{module_id}").friction[1] = 0.50
                self.model.geom(f"right{module_id}").friction[1] = 0.50

                self.model.geom(f"left{module_id}").friction[2] = roll_friction
                self.model.geom(f"right{module_id}").friction[2] = roll_friction
                # self.model.geom(f"stick{module_id}").friction[1:3] = roll_friction

                for geom_name in [
                    f"left{module_id}",
                    f"right{module_id}",
                    # f"stick{module_id}",
                ]:
                    self.model.geom(geom_name).priority[0] = 2

                passive_component_geoms.remove(f"left{module_id}")
                passive_component_geoms.remove(f"right{module_id}")

            # all other geoms except floor
            for geom_name in passive_component_geoms:
                self.model.geom(geom_name).friction[0] = 1
                self.model.geom(geom_name).friction[1:3] = roll_friction
                self.model.geom(geom_name).priority[0] = 2

            
            

        # Rolling friction
        # rolling_cfg = friction_cfg.get("rolling", {})
        # if rolling_cfg.get("enabled", False):
        #     rolling_range = rolling_cfg.get("range", [0.0001, 0.0005])
        #     roll_friction = np.random.uniform(
        #         rolling_range[0], rolling_range[1], size=2
        #     )
        #     self.model.geom("floor").friction[1:3] = roll_friction

    def _set_initial_state(self) -> None:
        """Set initial robot state with randomization."""
        # Handle orientation randomization
        final_quat = self._get_randomized_orientation()

        # Setup initial position
        self._setup_initial_positions(final_quat)

        # Apply initial state with noise
        qpos = self._apply_initial_noise()
        qvel = self._get_initial_velocities()

        # Set MuJoCo state
        self.set_state(qpos, qvel)

        # Record reset position
        self.reset_pos = self.data.qpos[:2].copy()

    def _get_randomized_orientation(self) -> np.ndarray:
        """Get randomized initial orientation."""
        if self.init_cfg.get("fully_randomize_orientation", False):
            # Fully random orientation
            rand_quat = self.np_random.normal(0, 1, 4)
            return rand_quat / np.linalg.norm(rand_quat)
        elif self.init_cfg.get("randomize_orientation", False):
            # Random rotation around Z-axis
            rotate_angle = self.np_random.uniform(0, 2 * np.pi)
            rand_rotation = construct_quaternion([0, 0, 1], rotate_angle)
            final_quat = quaternion_multiply_alt(self.init_quat, rand_rotation)

            # Update forward vector
            if self.forward_vec is not None:
                self.adjusted_forward_vec = rotate_vector2D(
                    self.forward_vec[:2], -rotate_angle
                )
            return final_quat
        else:
            # No orientation randomization
            if self.forward_vec is not None:
                self.adjusted_forward_vec = self.forward_vec
            return self.init_quat

    def _setup_initial_positions(self, quat: np.ndarray) -> None:
        """Setup initial positions and joint states."""
        # Handle multiple initial positions
        if is_list_like(self.init_pos[0]):
            init_pos_list = copy.deepcopy(self.init_pos)
            quat_list = [quat] + [[1, 0, 0, 0]] * 10  # Default quaternion list

            for i in range(self.model.njnt):
                if self.model.jnt_type[i] == 0:  # mjJNT_FREE
                    qpos_adr = self.model.jnt_qposadr[i]
                    self.init_qpos[qpos_adr : qpos_adr + 3] = init_pos_list.pop(0)
                    self.init_qpos[qpos_adr + 3 : qpos_adr + 7] = quat_list.pop(0)
        else:
            self.init_qpos[:3] = self.init_pos
            self.init_qpos[3:7] = quat

        # Setup free joint addresses
        self.free_joint_addr = [
            self.model.jnt_qposadr[i]
            for i in range(self.model.njnt)
            if self.model.jnt_type[i] == 0
        ]

        # Joint position randomization
        randomization_cfg = getattr(self.cfg, "randomization", {})
        dof_cfg = randomization_cfg.get("init_joint_pos", {})
        if dof_cfg.get("enabled", False):
            clip_actions = self.cfg.control.symmetric_limit  # type: ignore
            joint_noise = self.np_random.uniform(
                -clip_actions, clip_actions, self.num_joint
            )
            self.init_qpos[self.model.jnt_qposadr[self.joint_idx]] = (
                self.default_dof_pos + joint_noise
            )
            # TODO: consider frozen joints
        else:
            self.init_qpos[self.model.jnt_qposadr[self.joint_idx]] = self.init_joint_pos

    def _apply_initial_noise(self) -> np.ndarray:
        """Apply initial position noise."""
        if self.given_init_qpos is not None:
            return np.array(self.given_init_qpos)

        if self.init_cfg.get("noisy_init", True):
            return np.asarray(
                self.init_qpos + self.np_random.uniform(-0.1, 0.1, size=self.model.nq)
            )
        else:
            return np.asarray(self.init_qpos.copy())

    def _get_initial_velocities(self) -> np.ndarray:
        """Get initial velocities with randomization."""
        init_qvel = getattr(self, "init_qvel", np.zeros(self.model.nv))

        if self.init_cfg.get("randomize_ini_vel", True):
            randomize_vel = self.init_cfg.get("randomize_ini_vel", True)
            if is_list_like(randomize_vel):
                for i, vel_range in enumerate(randomize_vel):
                    init_qvel[i] = self.np_random.uniform(-vel_range, vel_range)
            else:
                init_qvel[:6] = self.np_random.uniform(-1, 1, 6)

        return np.asarray(init_qvel + self.np_random.normal(0, 0.1, self.model.nv))

    def reload_model(self, xml_string: str) -> None:
        """Reload MuJoCo model with new XML."""
        # Preserve render settings
        saved_render_mode = getattr(self, "render_mode", "none")
        saved_video_path = getattr(self, "video_path", "robot_video.mp4")
        saved_video_fps = getattr(self, "video_fps", 20)
        saved_recording_active = getattr(self, "recording_active", False)
        saved_video_frames = getattr(self, "video_frames", [])

        # Clean up renderers before reloading
        self._cleanup_egl_renderer()
        if hasattr(self, "mujoco_renderer") and self.mujoco_renderer is not None:
            try:
                self.mujoco_renderer.close()
            except Exception:
                pass
            self.mujoco_renderer = None

        MujocoEnv.__init__(
            self,
            xml_string,
            self.frame_skip,
            observation_space=None,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode="human" if self.render_mode == "viewer" else "rgb_array",
            width=self.render_size[0],
            height=self.render_size[1],
        )

        # Restore render settings
        self.render_mode = saved_render_mode
        self.video_path = saved_video_path
        self.video_fps = saved_video_fps
        self.recording_active = saved_recording_active
        self.video_frames = saved_video_frames

        # Recreate EGL renderer if needed
        if self.render_mode == "mp4":
            self.egl_renderer = None
            self.preferred_camera_id = None

    def render(self) -> Optional[Union[np.ndarray, Any]]:  # type: ignore
        """Render environment with different modes: 'none', 'viewer', 'mp4'."""
        if self.render_mode == "none":
            return None

        elif self.render_mode == "viewer":
            if self._passive_viewer is None:
                self._viewer_context_manager = mujoco.viewer.launch_passive(
                    self.model, self.data
                )
                assert self._viewer_context_manager is not None
                self._passive_viewer = self._viewer_context_manager.__enter__()
                print("Passive viewer initialized successfully")
            return None

        elif self.render_mode == "mp4":
            return self._capture_frame_egl()

        else:
            return None

    def close(self) -> None:
        """Close environment and cleanup resources."""
        # Stop any active video recording
        if (
            hasattr(self, "render_mode")
            and self.render_mode == "mp4"
            and getattr(self, "recording_active", False)
        ):
            self.stop_video_recording()

        # Cleanup passive viewer
        if (
            hasattr(self, "_viewer_context_manager")
            and self._viewer_context_manager is not None
        ):
            try:
                self._viewer_context_manager.__exit__(None, None, None)
            except Exception:
                pass
            finally:
                self._passive_viewer = None
                self._viewer_context_manager = None

        # Cleanup EGL renderer
        self._cleanup_egl_renderer()

        # Call parent close
        super().close()

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup

    def _get_available_cameras(self) -> list:
        """Get list of available cameras from the MuJoCo model."""
        if not hasattr(self, "model") or self.model is None:
            return []

        cameras = []
        for i in range(self.model.ncam):
            camera_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if camera_name:
                cameras.append({"id": i, "name": camera_name})
        return cameras

    def _get_preferred_camera(self) -> Optional[Union[str, int]]:
        """Get the preferred camera for rendering, preferring XML-defined cameras."""
        cameras = self._get_available_cameras()

        if not cameras:
            # No cameras defined in XML, use default free camera
            return -1

        # Check if a specific camera is configured
        configured_camera = self.sim_cfg.get("render_camera", None)
        if configured_camera is not None:
            # Try to find camera by name or ID
            if isinstance(configured_camera, str):
                for camera in cameras:
                    if camera["name"] == configured_camera:
                        print(
                            f"Using configured XML camera: {camera['name']} (ID: {camera['id']})"
                        )
                        return int(camera["id"])
                print(f"Warning: Configured camera '{configured_camera}' not found")
            elif isinstance(configured_camera, int):
                if 0 <= configured_camera < len(cameras):
                    camera = cameras[configured_camera]
                    print(
                        f"Using configured XML camera by ID: {camera['name']} (ID: {camera['id']})"
                    )
                    return int(configured_camera)
                print(f"Warning: Configured camera ID {configured_camera} out of range")

        # Prefer cameras with specific names in order of preference
        preferred_names = [
            "follow_camera",
            "main_camera",
            "robot_camera",
            "tracking_camera",
        ]

        for preferred_name in preferred_names:
            for camera in cameras:
                if preferred_name in camera["name"].lower():
                    print(f"Using XML camera: {camera['name']} (ID: {camera['id']})")
                    return int(camera["id"])

        # If no preferred camera found, use the first available camera
        camera = cameras[0]
        print(
            f"Using first available XML camera: {camera['name']} (ID: {camera['id']})"
        )
        return int(camera["id"])

    def _add_metrics_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add text overlays with action, reward, and custom metrics."""
        if not getattr(self, "video_show_metrics", True):
            return frame

        # Get current metrics
        # metrics = self._get_current_metrics()
        # np.array2string(self.state.action_history.last_action, precision=2, separator=',', suppress_small=True)
        metrics = {
            "Action": np.array2string(
                self.state.action_history.last_action,
                precision=2,
                separator=",",
                suppress_small=True,
            ),
            "Reward": f"{self.state.reward_history.last_reward:.2f}",
            "Speed": f"{self.state.speed[0]:.2f}",
            "Step": self.step_count,
            "Episode": self.episode_counter + 1,
        }

        if self.goal_task_enabled and self.goal_position_world is not None:
            metrics["GoalDist"] = f"{self.goal_distance:.3f}"
            metrics["GoalDelta"] = f"{self.goal_distance_delta:+.4f}"
            metrics["GoalXY"] = np.array2string(
                np.asarray(self.goal_position_world[:2], dtype=np.float32),
                precision=2,
                separator=",",
                suppress_small=True,
            )

        cmd_dict = self.state.command_manager.get_commands_dict()
        if cmd_dict:
            for key, value in cmd_dict.items():
                metrics["Cmd:" + key] = (
                    f"{value:.2f}" if isinstance(value, float) else str(value)
                )

        if not metrics:
            return frame

        # Configure text appearance
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)  # White text
        thickness = 1
        line_height = 20
        start_y = 20

        # Draw semi-transparent background for better readability
        overlay = frame.copy()
        text_lines = [f"{label}: {value}" for label, value in metrics.items()]
        max_width = max(
            cv2.getTextSize(text, font, font_scale, thickness)[0][0]
            for text in text_lines
        )
        bg_height = len(metrics) * line_height + 10
        bg_width = max(350, max_width + 20)
        cv2.rectangle(overlay, (5, 5), (bg_width, bg_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        for i, text in enumerate(text_lines):
            y_pos = start_y + (i * line_height)
            cv2.putText(frame, text, (10, y_pos), font, font_scale, color, thickness)

        return frame

    def do_simulation(self, ctrl: np.ndarray, n_frames: int) -> None:
        """
        Step the simulation n number of frames and applying a control action.
        Integrates with passive viewer when in 'viewer' mode.
        """
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(
                f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}"
            )

        if self.render_mode == "viewer" and self._passive_viewer is not None:
            self.data.ctrl[:] = ctrl
            for _i in range(n_frames):
                mujoco.mj_step(self.model, self.data)
                self._passive_viewer.sync()
        else:
            self._step_mujoco_simulation(ctrl, n_frames)

    def _step_mujoco_simulation(self, ctrl: np.ndarray, n_frames: int) -> None:
        """
        Step over the MuJoCo simulation (standard implementation).
        """
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, nstep=n_frames)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        mujoco.mj_rnePostConstraint(self.model, self.data)
