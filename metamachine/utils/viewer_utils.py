"""
Viewer Utilities for Real Robot Visualization

This module provides utilities for visualizing real robot orientation
and joint positions in MuJoCo viewer in real-time.

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>
Licensed under the Apache License, Version 2.0
"""

import time
import xml.etree.ElementTree as ET
import numpy as np
from typing import Callable, Optional, Tuple
import mujoco
import mujoco.viewer


# =============================================================================
# Quaternion Utilities
# =============================================================================

def canonicalize_quat(q: np.ndarray) -> np.ndarray:
    """Ensure quaternion has positive w component.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        Canonicalized quaternion
    """
    return q if q[0] >= 0 else -q


def normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion to unit length.
    
    Args:
        q: Quaternion (any format)
        
    Returns:
        Normalized quaternion
    """
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    return q / norm


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert quaternion from [x, y, z, w] to [w, x, y, z] format.
    
    Args:
        q: Quaternion in [x, y, z, w] format (common in IMU sensors)
        
    Returns:
        Quaternion in [w, x, y, z] format (MuJoCo convention)
    """
    return np.array([q[3], q[0], q[1], q[2]])


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Convert quaternion from [w, x, y, z] to [x, y, z, w] format.
    
    Args:
        q: Quaternion in [w, x, y, z] format (MuJoCo convention)
        
    Returns:
        Quaternion in [x, y, z, w] format (common in IMU sensors)
    """
    return np.array([q[1], q[2], q[3], q[0]])


def _format_floats(values: np.ndarray | list[float]) -> str:
    """Format float values compactly for MuJoCo XML attributes."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return " ".join(f"{value:.12g}" for value in arr)


def _mat9_to_quat_wxyz(mat9: np.ndarray) -> np.ndarray:
    """Convert a flattened 3x3 MuJoCo rotation matrix to [w, x, y, z]."""
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(mat9, dtype=np.float64))
    return quat


def _inject_welded_tracker_body(
    xml_string: str,
    reference_model: mujoco.MjModel,
    *,
    tracked_site: str = "imu_site0",
) -> tuple[str, str | None]:
    """Add a viewer-only freejoint tracker body welded to a target site."""
    site_id = mujoco.mj_name2id(reference_model, mujoco.mjtObj.mjOBJ_SITE, tracked_site)
    if site_id < 0:
        return xml_string, None

    reference_data = mujoco.MjData(reference_model)
    mujoco.mj_forward(reference_model, reference_data)

    site_pos = np.asarray(reference_data.site_xpos[site_id], dtype=np.float64)
    site_quat = _mat9_to_quat_wxyz(reference_data.site_xmat[site_id])

    root = ET.fromstring(xml_string)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Viewer model XML is missing <worldbody>")

    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    tracker_body_name = f"viewer_tracker__{tracked_site}"
    tracker_joint_name = f"viewer_tracker_freejoint__{tracked_site}"
    tracker_site_name = f"viewer_tracker_site__{tracked_site}"

    tracker_body = ET.SubElement(
        worldbody,
        "body",
        name=tracker_body_name,
        pos=_format_floats(site_pos),
        quat=_format_floats(site_quat),
    )
    ET.SubElement(tracker_body, "freejoint", name=tracker_joint_name)
    ET.SubElement(
        tracker_body,
        "inertial",
        pos="0 0 0",
        mass="1000",
        diaginertia="1000 1000 1000",
    )
    ET.SubElement(
        tracker_body,
        "site",
        name=tracker_site_name,
        pos="0 0 0",
        size="0.012",
        rgba="1 0 0 0.12",
    )
    ET.SubElement(
        equality,
        "weld",
        site1=tracked_site,
        site2=tracker_site_name,
        solref="0.0001 1",
        solimp="1 1 0.01",
    )

    return ET.tostring(root, encoding="unicode"), tracker_joint_name


# =============================================================================
# Model Creation
# =============================================================================

def create_viewer_model(cfg, viewer_config=None) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """Create a MuJoCo model for visualization.
    
    Args:
        cfg: Real robot config (used if viewer_config is None)
        viewer_config: Optional separate config for the viewer
        
    Returns:
        (model, data): MuJoCo model and data objects
    """
    import os
    from pathlib import Path
    
    # Determine which config to use for the viewer
    if viewer_config is not None:
        from metamachine.environments.configs.config_registry import ConfigRegistry
        if os.path.exists(viewer_config):
            vis_cfg = ConfigRegistry.create_from_file(viewer_config)
        else:
            vis_cfg = ConfigRegistry.create_from_name(viewer_config)
    else:
        vis_cfg = cfg
    
    # Create a temporary sim environment to generate the XML
    from metamachine.environments.env_sim import MetaMachine
    temp_cfg = vis_cfg.copy()
    temp_cfg.environment.mode = "sim"
    temp_env = MetaMachine(temp_cfg)
    
    xml_string = getattr(temp_env, "xml_string", None)
    if not xml_string:
        raise ValueError("Temporary viewer environment did not expose xml_string")

    viewer_xml_string, tracker_joint_name = _inject_welded_tracker_body(
        xml_string,
        temp_env.model,
        tracked_site="imu_site0",
    )
    model = mujoco.MjModel.from_xml_string(viewer_xml_string)
    model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)
    if tracker_joint_name is not None:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, tracker_joint_name)
        if joint_id >= 0:
            qpos_adr = model.jnt_qposadr[joint_id]
            tracked_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu_site0")
            mujoco.mj_forward(model, data)
            if tracked_site_id >= 0:
                data.qpos[qpos_adr:qpos_adr + 3] = data.site_xpos[tracked_site_id]
                data.qpos[qpos_adr + 3:qpos_adr + 7] = _mat9_to_quat_wxyz(
                    data.site_xmat[tracked_site_id]
                )
                mujoco.mj_forward(model, data)
    
    # Close temp environment (we just needed the model)
    temp_env.close()
    
    return model, data


def print_model_info(model: mujoco.MjModel, verbose: bool = True) -> dict:
    """Print MuJoCo model information and return structure details.
    
    Args:
        model: MuJoCo model
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary with model structure information
    """
    has_floating_base = (model.nq >= 7)
    has_mocap = (model.nmocap > 0)
    joint_qpos_start = 7 if has_floating_base else 0
    
    info = {
        'nq': model.nq,
        'nv': model.nv,
        'nu': model.nu,
        'nmocap': model.nmocap,
        'nbody': model.nbody,
        'has_floating_base': has_floating_base,
        'has_mocap': has_mocap,
        'joint_qpos_start': joint_qpos_start,
    }
    
    if verbose:
        print(f"\n  Model info:")
        print(f"    nq (position DoF): {model.nq}")
        print(f"    nv (velocity DoF): {model.nv}")
        print(f"    nu (actuators): {model.nu}")
        print(f"    nmocap (mocap bodies): {model.nmocap}")
        print(f"    nbody (bodies): {model.nbody}")
        
        if has_floating_base:
            print(f"\n  ✓ Found floating base. Will update base orientation via qpos.")
            print(f"    Base qpos: [x, y, z] = qpos[0:3]")
            print(f"    Base quat: [w, x, y, z] = qpos[3:7]")
        elif has_mocap:
            print(f"\n  ✓ Found {model.nmocap} mocap bodies. Will update orientation via mocap.")
        else:
            print("\n  ⚠ No floating base or mocap found. Will only mirror joint positions.")
        
        print(f"    Joint qpos start index: {joint_qpos_start}")
    
    return info


# =============================================================================
# Viewer State Updater
# =============================================================================

class ViewerStateUpdater:
    """Helper class to update MuJoCo viewer state from real robot data."""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, verbose: bool = False):
        """Initialize the updater.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            verbose: Print debug information
        """
        self.model = model
        self.data = data
        self.verbose = verbose
        
        # Get model structure info
        self.info = print_model_info(model, verbose=verbose)
        self.has_floating_base = self.info['has_floating_base']
        self.has_mocap = self.info['has_mocap']
        self.joint_qpos_start = self.info['joint_qpos_start']
        self.mocap_idx = 0
        self.tracked_site_name = "imu_site0"
        self.tracked_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, self.tracked_site_name
        )
        self.tracker_joint_name = f"viewer_tracker_freejoint__{self.tracked_site_name}"
        self.tracker_joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, self.tracker_joint_name
        )
        self.tracker_qpos_adr = (
            int(model.jnt_qposadr[self.tracker_joint_id]) if self.tracker_joint_id >= 0 else None
        )
        self.tracker_settle_steps = 15
        
        self.step_count = 0

    def _get_tracked_site_quat_wxyz(self, observable_data: dict) -> np.ndarray | None:
        """Get the desired world quaternion for the tracked site."""
        if self.tracked_site_name.startswith("imu_site") and "quats" in observable_data:
            suffix = self.tracked_site_name.removeprefix("imu_site")
            if suffix.isdigit():
                module_idx = int(suffix)
                quats = observable_data.get("quats")
                if quats is not None and module_idx < len(quats):
                    quat_wxyz = xyzw_to_wxyz(quats[module_idx])
                    quat_wxyz = normalize_quat(quat_wxyz)
                    return canonicalize_quat(quat_wxyz)

        if "quat" in observable_data:
            quat_wxyz = xyzw_to_wxyz(observable_data["quat"])
            quat_wxyz = normalize_quat(quat_wxyz)
            return canonicalize_quat(quat_wxyz)

        return None
    
    def update(self, observable_data: dict) -> None:
        """Update viewer state from real robot observable data.
        
        Args:
            observable_data: Dictionary containing robot sensor data
                Expected keys: 'quat' [x,y,z,w], 'dof_pos'
        """
        # Tracked-site weld mode only: start each frame from a neutral base pose.
        if self.has_floating_base:
            self.data.qpos[0:3] = [0.0, 0.0, 0.5]
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        elif self.has_mocap:
            self.data.mocap_pos[self.mocap_idx] = np.array([0.0, 0.0, 0.5])
            self.data.mocap_quat[self.mocap_idx] = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Update joint positions
        if 'dof_pos' in observable_data:
            dof_pos = observable_data['dof_pos']
            num_joints = min(len(dof_pos), self.model.nu)
            
            # Set joint positions in qpos (after base position/orientation)
            if self.has_floating_base and (self.joint_qpos_start + num_joints) <= self.model.nq:
                self.data.qpos[self.joint_qpos_start:self.joint_qpos_start + num_joints] = dof_pos[:num_joints]
            
            # Also set control inputs
            self.data.ctrl[:num_joints] = dof_pos[:num_joints]

        # Compute current robot site world position before driving the tracker body.
        mujoco.mj_forward(self.model, self.data)

        tracked_site_quat = self._get_tracked_site_quat_wxyz(observable_data)
        if (
            self.tracker_qpos_adr is not None
            and self.tracked_site_id >= 0
            and tracked_site_quat is not None
        ):
            tracked_site_pos = np.asarray(
                self.data.site_xpos[self.tracked_site_id], dtype=np.float64
            )
            self.data.qpos[self.tracker_qpos_adr:self.tracker_qpos_adr + 3] = tracked_site_pos
            self.data.qpos[self.tracker_qpos_adr + 3:self.tracker_qpos_adr + 7] = tracked_site_quat
            self.data.qvel[:] = 0.0
            for _ in range(self.tracker_settle_steps):
                mujoco.mj_step(self.model, self.data)

        if self.verbose and tracked_site_quat is not None and self.step_count % 50 == 0:
            print(
                f"\n[Debug] Tracked site quat (wxyz): "
                f"[{tracked_site_quat[0]:+.3f}, {tracked_site_quat[1]:+.3f}, "
                f"{tracked_site_quat[2]:+.3f}, {tracked_site_quat[3]:+.3f}]",
                flush=True,
            )
        
        self.step_count += 1


# =============================================================================
# High-Level Runner
# =============================================================================

def run_with_viewer(
    env,
    action_fn: Callable[[np.ndarray, int], np.ndarray],
    duration: Optional[float] = None,
    viewer_config: Optional[str] = None,
    verbose: bool = False
) -> None:
    """Run real robot with MuJoCo viewer showing robot orientation.
    
    Args:
        env: RealMetaMachine environment
        action_fn: Function that takes (obs, step_count) and returns action
        duration: Duration in seconds (None = run until interrupt)
        viewer_config: Optional config for viewer model
        verbose: Print debug information
    """
    print("\n" + "=" * 60)
    print("Running with MuJoCo Viewer")
    print("=" * 60)
    print("  Viewer will show robot orientation from real IMU")
    print("  Motor positions will mirror real robot")
    print("=" * 60)
    
    # Create viewer model
    print("\nCreating MuJoCo viewer model...")
    m, d = create_viewer_model(env.cfg, viewer_config)
    
    # Create state updater
    updater = ViewerStateUpdater(m, d, verbose=verbose)
    
    # Reset environment
    obs, info = env.reset()
    
    start_time = time.time()
    step_count = 0
    episode_reward = 0
    
    # Launch viewer in passive mode
    with mujoco.viewer.launch_passive(m, d) as viewer:
        print("\nViewer launched. Press Ctrl+C to stop.")
        
        try:
            while viewer.is_running():
                loop_start = time.time()
                elapsed = time.time() - start_time
                
                # Check duration
                if duration is not None and elapsed >= duration:
                    print(f"\n[Done] Reached duration limit ({duration}s)")
                    break
                
                # Get action from action function
                action = action_fn(obs, step_count)
                
                # Execute step in real environment
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                
                # Update viewer with real robot state
                if hasattr(env, 'observable_data'):
                    updater.update(env.observable_data)
                
                # Sync viewer
                with viewer.lock():
                    viewer.sync()
                
                # Print status periodically
                if step_count % 100 == 0:
                    print(f"\r[Step {step_count}] Time: {elapsed:.1f}s, "
                          f"Reward: {episode_reward:.2f}", end="", flush=True)
                
                # Check for episode end
                if done or truncated:
                    print(f"\n[Episode ended] Reward: {episode_reward:.2f}")
                    obs, info = env.reset()
                    episode_reward = 0
                
                # Maintain loop timing
                loop_time = time.time() - loop_start
                if hasattr(env.cfg, 'control') and hasattr(env.cfg.control, 'dt'):
                    dt = env.cfg.control.dt
                    time.sleep(max(0, dt - loop_time))
                
                step_count += 1
        
        except KeyboardInterrupt:
            print("\n[Interrupted]")
        
        finally:
            elapsed = time.time() - start_time
            print(f"\n\nViewer run completed: {step_count} steps in {elapsed:.1f}s")
