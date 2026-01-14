"""Rendering utilities for MuJoCo-based robot visualization and simulation.

This module provides functions for:
- Rendering 3D lines in MuJoCo viewer
- Interactive robot visualization with physics simulation
- Trajectory plotting and analysis
- Robot state monitoring and debugging

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

import os
import time
from typing import Any, Optional, Tuple, Union

import mujoco
import numpy as np
from numpy import cos, sin

# Local imports
from .math_utils import quat_rotate_inverse, quaternion_to_euler2, wxyz_to_xyzw
from .visual_utils import compile_xml, get_joint_pos_addr


# =============================================================================
# Scene Marker Utilities
# =============================================================================

def add_marker_to_scene(
    scene: Any,
    pos: np.ndarray,
    geom_type: int = mujoco.mjtGeom.mjGEOM_SPHERE,
    size: np.ndarray = None,
    rgba: np.ndarray = None,
    mat: np.ndarray = None,
) -> int:
    """Add a visual marker to a MuJoCo scene.
    
    This works with both passive viewer (user_scn) and EGL renderer (scene).
    
    Args:
        scene: MuJoCo scene object (mjvScene). For passive viewer use viewer.user_scn,
               for EGL renderer use renderer.scene.
        pos: Position [x, y, z] of the marker.
        geom_type: MuJoCo geometry type (default: SPHERE).
                   Common types: mjGEOM_SPHERE, mjGEOM_CYLINDER, mjGEOM_BOX
        size: Size parameters (depends on geom type).
              - SPHERE: [radius, 0, 0]
              - CYLINDER: [radius, half_height, 0]
              - BOX: [half_x, half_y, half_z]
        rgba: Color [r, g, b, a] in range [0, 1].
        mat: 3x3 rotation matrix (flattened to 9 elements).
    
    Returns:
        Index of the added geom.
        
    Example:
        # Add a red sphere marker
        add_marker_to_scene(
            renderer.scene, 
            pos=np.array([1.0, 0.5, 0.0]),
            rgba=np.array([1.0, 0.0, 0.0, 0.8])
        )
        
        # Add a green disc on the ground
        add_ground_disc_marker(renderer.scene, pos_xy=[1.0, 0.5])
    """
    if size is None:
        size = np.array([0.05, 0, 0])
    if rgba is None:
        rgba = np.array([1.0, 0.0, 0.0, 0.8], dtype=np.float32)
    if mat is None:
        mat = np.eye(3).flatten()
    
    geom_index = scene.ngeom
    
    if geom_index >= scene.maxgeom:
        print(f"Warning: Scene geom limit reached ({scene.maxgeom})")
        return -1
    
    mujoco.mjv_initGeom(
        scene.geoms[geom_index],
        type=geom_type,
        size=np.asarray(size, dtype=np.float64),
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.asarray(mat, dtype=np.float64).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    
    scene.ngeom += 1
    return geom_index


def add_ground_disc_marker(
    scene: Any,
    pos_xy: Union[np.ndarray, Tuple[float, float]],
    radius: float = 0.1,
    height: float = 0.005,
    color: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.8),
    z_offset: float = 0.001,
) -> int:
    """Add a disc marker on the ground plane.
    
    Args:
        scene: MuJoCo scene object.
        pos_xy: 2D position [x, y] on the ground.
        radius: Radius of the disc.
        height: Height (thickness) of the disc.
        color: RGBA color tuple.
        z_offset: Small offset above ground to prevent z-fighting.
        
    Returns:
        Index of the added geom.
    """
    pos_xy = np.asarray(pos_xy)
    pos = np.array([pos_xy[0], pos_xy[1], z_offset])
    size = np.array([radius, height / 2, 0])
    
    return add_marker_to_scene(
        scene,
        pos=pos,
        geom_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=size,
        rgba=np.array(color, dtype=np.float32),
        mat=np.eye(3).flatten(),
    )


def add_sphere_marker(
    scene: Any,
    pos: np.ndarray,
    radius: float = 0.05,
    color: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.8),
) -> int:
    """Add a sphere marker at a given position.
    
    Args:
        scene: MuJoCo scene object.
        pos: 3D position [x, y, z].
        radius: Radius of the sphere.
        color: RGBA color tuple.
        
    Returns:
        Index of the added geom.
    """
    return add_marker_to_scene(
        scene,
        pos=np.asarray(pos),
        geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0, 0]),
        rgba=np.array(color, dtype=np.float32),
    )


def add_line_marker(
    scene: Any,
    start: np.ndarray,
    end: np.ndarray,
    radius: float = 0.01,
    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.8),
) -> int:
    """Add a line (cylinder) marker between two points.
    
    Args:
        scene: MuJoCo scene object.
        start: Starting position [x, y, z].
        end: Ending position [x, y, z].
        radius: Radius of the line (cylinder).
        color: RGBA color tuple.
        
    Returns:
        Index of the added geom.
    """
    start = np.asarray(start)
    end = np.asarray(end)
    
    # Calculate length and direction
    diff = end - start
    length = np.linalg.norm(diff)
    
    if length < 1e-6:
        return -1
    
    direction = diff / length
    midpoint = (start + end) / 2
    
    # Calculate rotation matrix to align Z-axis with direction
    z_axis = np.array([0, 0, 1])
    
    if np.allclose(np.abs(np.dot(direction, z_axis)), 1.0):
        # Direction is parallel to Z-axis
        rotation_matrix = np.eye(3)
        if np.dot(direction, z_axis) < 0:
            rotation_matrix[0, 0] = -1
            rotation_matrix[2, 2] = -1
    else:
        v = np.cross(z_axis, direction)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, direction)
        
        # Skew-symmetric matrix
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        
        # Rodrigues' formula
        rotation_matrix = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s**2 + 1e-8))
    
    return add_marker_to_scene(
        scene,
        pos=midpoint,
        geom_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=np.array([radius, length / 2, 0]),
        rgba=np.array(color, dtype=np.float32),
        mat=rotation_matrix.flatten(),
    )


def add_ground_line_marker(
    scene: Any,
    start_xy: Union[np.ndarray, Tuple[float, float]],
    end_xy: Union[np.ndarray, Tuple[float, float]],
    radius: float = 0.01,
    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.8),
    z_offset: float = 0.02,
) -> int:
    """Add a line marker on the ground plane (projected from 2D coordinates).
    
    Args:
        scene: MuJoCo scene object.
        start_xy: Starting 2D position [x, y].
        end_xy: Ending 2D position [x, y].
        radius: Radius of the line.
        color: RGBA color tuple.
        z_offset: Height above ground plane.
        
    Returns:
        Index of the added geom.
    """
    start = np.array([start_xy[0], start_xy[1], z_offset])
    end = np.array([end_xy[0], end_xy[1], z_offset])
    return add_line_marker(scene, start, end, radius, color)


def add_ground_arrow_marker(
    scene: Any,
    start_xy: Union[np.ndarray, Tuple[float, float]],
    direction_xy: Union[np.ndarray, Tuple[float, float]],
    length: float = 0.5,
    shaft_radius: float = 0.015,
    head_radius: float = 0.04,
    head_length: float = 0.08,
    color: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.8),
    z_offset: float = 0.02,
) -> Tuple[int, int]:
    """Add an arrow marker on the ground plane pointing in a 2D direction.
    
    The arrow consists of a shaft (cylinder) and a head (cone).
    
    Args:
        scene: MuJoCo scene object.
        start_xy: Starting 2D position [x, y].
        direction_xy: 2D direction vector [dx, dy] (will be normalized).
        length: Total length of the arrow (shaft + head).
        shaft_radius: Radius of the shaft.
        head_radius: Radius of the arrow head base.
        head_length: Length of the arrow head.
        color: RGBA color tuple.
        z_offset: Height above ground plane.
        
    Returns:
        Tuple of (shaft_geom_index, head_geom_index).
    """
    direction_xy = np.asarray(direction_xy, dtype=np.float64)
    dir_norm = np.linalg.norm(direction_xy)
    
    if dir_norm < 1e-6:
        return (-1, -1)
    
    direction_xy = direction_xy / dir_norm
    
    start = np.array([start_xy[0], start_xy[1], z_offset])
    direction_3d = np.array([direction_xy[0], direction_xy[1], 0])
    
    shaft_length = length - head_length
    shaft_end = start + direction_3d * shaft_length
    
    # Draw shaft
    shaft_idx = add_line_marker(
        scene, start, shaft_end, shaft_radius, color
    )
    
    # Draw head (cone) - use a cylinder as approximation
    head_start = shaft_end
    head_end = head_start + direction_3d * head_length
    head_mid = (head_start + head_end) / 2
    
    # Calculate rotation for the cone
    z_axis = np.array([0, 0, 1])
    if np.allclose(np.abs(np.dot(direction_3d, z_axis)), 1.0):
        rotation_matrix = np.eye(3)
    else:
        v = np.cross(z_axis, direction_3d)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, direction_3d)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s**2 + 1e-8))
    
    # Use a cone-like shape (tapered cylinder approximation with a sphere at tip)
    head_idx = add_marker_to_scene(
        scene,
        pos=head_mid,
        geom_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=np.array([head_radius, head_length / 2, 0]),
        rgba=np.array(color, dtype=np.float32),
        mat=rotation_matrix.flatten(),
    )
    
    return (shaft_idx, head_idx)


def add_arrow_marker(
    scene: Any,
    start: np.ndarray,
    direction: np.ndarray,
    length: float = 0.3,
    radius: float = 0.01,
    color: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.8),
) -> int:
    """Add an arrow marker showing direction.
    
    Args:
        scene: MuJoCo scene object.
        start: Starting position [x, y, z].
        direction: Unit direction vector [dx, dy, dz].
        length: Length of the arrow.
        radius: Radius of the arrow shaft.
        color: RGBA color tuple.
        
    Returns:
        Index of the added geom (shaft).
    """
    start = np.asarray(start)
    direction = np.asarray(direction)
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    
    # Calculate midpoint and rotation
    midpoint = start + direction * (length / 2)
    
    # Calculate rotation matrix to align Z-axis with direction
    z_axis = np.array([0, 0, 1])
    
    if np.allclose(direction, z_axis) or np.allclose(direction, -z_axis):
        rotation_matrix = np.eye(3)
        if np.allclose(direction, -z_axis):
            rotation_matrix[2, 2] = -1
    else:
        v = np.cross(z_axis, direction)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, direction)
        
        # Skew-symmetric matrix
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        
        # Rodrigues' formula
        rotation_matrix = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s**2 + 1e-8))
    
    return add_marker_to_scene(
        scene,
        pos=midpoint,
        geom_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=np.array([radius, length / 2, 0]),
        rgba=np.array(color, dtype=np.float32),
        mat=rotation_matrix.flatten(),
    )

DEFAULT_ROBOT_CONFIG = {
    "theta": 0.4625123,
    "R": 0.07,
    "r": 0.03,
    "l_": 0.236,
    "delta_l": -0.001,
    "stick_ball_l": -0.001,
    "a": 0.236
    / 4,  # 0.0380409255338946, # l/6 stick center to the dock center on the side
    "stick_mass": 0.26,
    "top_hemi_mass": 0.74,
    "bottom_hemi_mass": 0.534,
}


def render_line(
    viewer,
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    color: tuple[float, float, float, float] = (1, 1, 1, 0.5),
) -> None:
    """Render a 3D line in the MuJoCo viewer.

    Args:
        viewer: MuJoCo viewer instance
        p1: Start point (x, y, z)
        p2: End point (x, y, z)
        color: RGBA color tuple (default: semi-transparent white)
    """
    p1, p2 = np.array(p1), np.array(p2)

    # Calculate direction vector and line properties
    direction = p2 - p1
    length = np.linalg.norm(direction)

    if length < 1e-8:  # Avoid division by zero for coincident points
        return

    direction_normalized = direction / length
    midpoint = (p1 + p2) / 2

    # Calculate rotation matrix to align cylinder with direction vector
    # Use Rodrigues' rotation formula to rotate z-axis to direction
    z_axis = np.array([0, 0, 1])

    if np.allclose(direction_normalized, z_axis) or np.allclose(
        direction_normalized, -z_axis
    ):
        rotation_matrix = np.eye(3)
        if np.allclose(direction_normalized, -z_axis):
            rotation_matrix[2, 2] = -1  # Flip z-axis
    else:
        v = np.cross(z_axis, direction_normalized)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, direction_normalized)

        # Skew-symmetric matrix
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

        # Rodrigues' formula
        rotation_matrix = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s**2))

    # Create cylinder geometry to represent the line
    geom_index = viewer.user_scn.ngeom

    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[geom_index],
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=np.array([0.01, length / 2, 1]),  # radius, half-length, unused
        pos=midpoint,
        mat=rotation_matrix.flatten(),
        rgba=np.array(color, dtype=np.float32),
    )

    viewer.user_scn.ngeom += 1


def view(
    file,
    fixed=False,
    pos=None,
    quat=None,
    vis_contact=False,
    joint_pos=None,
    callback=None,
):

    import mujoco.viewer

    if file.endswith(".xml"):
        xml_string = compile_xml(file)
    else:
        xml_string = file

    # m = mujoco.MjModel.from_xml_path(file)
    m = mujoco.MjModel.from_xml_string(xml_string)
    d = mujoco.MjData(m)
    d.qpos[:] = 0
    pos = d.qpos[0:3] if pos is None else pos
    quat = d.qpos[3:7] if quat is None else quat

    init_pos = np.copy(pos)
    init_quat = np.copy(quat)

    d.qpos[0:3] = pos
    d.qpos[3:7] = quat

    if joint_pos is not None:
        d.qpos[get_joint_pos_addr(m)] = joint_pos

    ini_pos2d = np.copy(d.qpos)[:2]
    acc_vel_body = np.array([0.0, 0, 0])
    last_com_pos = np.zeros(3)

    # Extract joint module information
    try:
        jointed_module_ids = sorted(
            [
                int(
                    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).replace(
                        "joint", ""
                    )
                )
                for j in m.actuator_trnid[:, 0]
            ]
        )
        joint_geom_idx = [m.geom(f"left{i}").id for i in jointed_module_ids]
        joint_body_idx = [m.geom(i).bodyid.item() for i in joint_geom_idx]
    except (ValueError, AttributeError) as e:
        print(f"Warning: Could not extract joint information: {e}")
        jointed_module_ids = []
        joint_body_idx = []

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
        n_ctr = 0
        while viewer.is_running():
            step_start = time.time()

            if fixed:
                d.qpos[0:3] = init_pos
                d.qpos[3:7] = init_quat
                d.qvel[:] = 0
                if joint_pos is not None:
                    d.qpos[get_joint_pos_addr(m)] = joint_pos
            if joint_pos is not None:
                d.ctrl[:] = joint_pos

            # Optional physics analysis and debugging
            if jointed_module_ids:  # Only if joint information is available
                try:
                    quat = d.qpos[3:7]
                    ang_vel_body = d.qvel[3:6]
                    gravity_vec = np.array([0, 0, -1])

                    # Get sensor data if available
                    np.array(
                        [
                            d.sensordata[
                                m.sensor(f"back_imu_vel{i}")
                                .adr[0] : m.sensor(f"back_imu_vel{i}")
                                .adr[0]
                                + 3
                            ]
                            for i in jointed_module_ids
                        ]
                    )
                    back_quat = np.array(
                        [
                            d.sensordata[
                                m.sensor(f"back_imu_quat{i}")
                                .adr[0] : m.sensor(f"back_imu_quat{i}")
                                .adr[0]
                                + 4
                            ]
                            for i in jointed_module_ids
                        ]
                    )
                    np.array([wxyz_to_xyzw(q) for q in back_quat])

                    # Calculate projected gravity and other physics quantities
                    accurate_projected_gravity = quat_rotate_inverse(quat, gravity_vec)
                    np.dot(-accurate_projected_gravity, ang_vel_body)

                    # Analyze floor contacts
                    floor_contacts = [c.geom for c in d.contact if 0 in c.geom]
                    list(
                        {geom for pair in floor_contacts for geom in pair if geom != 0}
                    )

                except (KeyError, IndexError):
                    # Skip sensor analysis if sensors are not available
                    pass

            # Store root quaternion information
            d.qpos[3:7]
            d.xquat[1]

            mujoco.mj_step(m, d)

            if callback is not None:
                callback(m, d)

            # Toggle contact point visualization if requested
            if vis_contact:
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
                        d.time % 2
                    )
            # Count self-collisions (excluding valid left-right module contacts)
            n_self_collision = 0
            for contact in d.contact:
                if contact.geom1 != 0 and contact.geom2 != 0:
                    try:
                        b1 = m.body(m.geom(contact.geom1).bodyid).name
                        b2 = m.body(m.geom(contact.geom2).bodyid).name
                        # Allow contacts between left and right parts of the same module
                        valid_contact = (
                            (b1[0] == "l" and b2[0] == "r")
                            or (b1[0] == "r" and b2[0] == "l")
                        ) and (b1[1] == b2[1])
                        if not valid_contact:
                            n_self_collision += 1
                    except (IndexError, AttributeError):
                        # Skip malformed body names
                        continue

            # Periodic state logging (every 100 steps)
            if n_ctr % 100 == 0:
                print(
                    f"Step {n_ctr}: Quat: {d.qpos[3:7]}, Pos: {d.qpos[0:3]}, Height: {d.qpos[2]:.3f}"
                )
                if len(d.qpos) > 7:
                    print(f"Joint DOF: {d.qpos[7:]}")

            # Calculate body-frame velocities and orientations
            vel_body = quat_rotate_inverse(d.qpos[3:7], d.qvel[:3])
            quat_rotate_inverse(d.qpos[3:7], np.array([1.0, 0, 0]))
            forward_euler = quaternion_to_euler2(d.qpos[3:7])
            roll = forward_euler[0]

            local_upward = np.array([0, sin(roll), cos(roll)])
            local_forward = np.array([0.0, cos(roll), -sin(roll)])

            # Process IMU sensor data if available
            if jointed_module_ids:
                try:
                    quats = np.array(
                        [
                            d.sensordata[
                                m.sensor(f"imu_quat{i}")
                                .adr[0] : m.sensor(f"imu_quat{i}")
                                .adr[0]
                                + 4
                            ]
                            for i in jointed_module_ids
                        ]
                    )
                    np.array(
                        [
                            quat_rotate_inverse(wxyz_to_xyzw(q), np.array([0, 0, 1]))
                            for q in quats
                        ]
                    )
                    projected_forwards = np.array(
                        [
                            quat_rotate_inverse(wxyz_to_xyzw(q), np.array([0, 1, 0]))
                            for q in quats
                        ]
                    )

                    np.array(
                        [
                            d.sensordata[
                                m.sensor(f"imu_gyro{i}")
                                .adr[0] : m.sensor(f"imu_gyro{i}")
                                .adr[0]
                                + 3
                            ]
                            for i in jointed_module_ids
                        ]
                    )
                    np.array(
                        [
                            d.sensordata[
                                m.sensor(f"imu_vel{i}")
                                .adr[0] : m.sensor(f"imu_vel{i}")
                                .adr[0]
                                + 3
                            ]
                            for i in jointed_module_ids
                        ]
                    )

                    # Periodic sensor data logging
                    if n_ctr % 100 == 0 and len(projected_forwards) > 0:
                        print(
                            f"Local forward: {local_forward}, IMU forward[0]: {projected_forwards[0]}"
                        )

                except (KeyError, IndexError):
                    # Skip if sensors are not available
                    pass

            # Calculate robot configuration geometry
            theta = DEFAULT_ROBOT_CONFIG["theta"]
            R = DEFAULT_ROBOT_CONFIG["R"]
            length = DEFAULT_ROBOT_CONFIG["l_"]

            if len(d.qpos) > 7:  # If there are joint positions
                alpha = d.qpos[-1]  # Last joint angle
                # Calculate end-effector position based on robot geometry
                (R + length) * np.cos(theta) * np.sin(alpha)
                -(R + length) * np.cos(theta) * np.cos(alpha)
                -(R + length) * np.sin(theta)

            # Calculate center of mass velocity if joint bodies are available
            if joint_body_idx:
                com_pos = np.mean(d.xpos[joint_body_idx], axis=0)
                com_vel = (com_pos - last_com_pos) / m.opt.timestep
                last_com_pos = com_pos
                acc_vel_body += com_vel

                # Periodic orientation logging
                if n_ctr % 50 == 0:
                    print(
                        f"Local forward: {local_forward}, Local upward: {local_upward}"
                    )
                    print(
                        f"Body velocity: {vel_body[0]:.3f}, COM velocity: {np.linalg.norm(com_vel):.3f}"
                    )

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            n_ctr += 1

            # Analyze trajectory after sufficient steps
            if n_ctr > 250 and joint_body_idx:
                displacement = d.qpos[:2] - ini_pos2d
                if np.linalg.norm(displacement) > 1e-6:
                    forward_vec = displacement / np.linalg.norm(displacement)
                    avg_vel = acc_vel_body / n_ctr
                    if np.linalg.norm(avg_vel[:2]) > 1e-6:
                        local_forward_vec = avg_vel[:2] / np.linalg.norm(avg_vel[:2])

                        if n_ctr % 100 == 0:
                            print(
                                f"Trajectory analysis - Forward: {forward_vec}, Avg velocity: {local_forward_vec}"
                            )


# Note: Photo and video capture functions are disabled due to external dependencies
# These functions require additional packages not included in the core metamachine library:
# - PIL/Pillow for image processing
# - Additional configuration and model loading utilities
#
# To implement these features, users should:
# 1. Install required dependencies: pip install pillow
# 2. Implement custom photo/video capture using MuJoCo's rendering capabilities
# 3. Use mujoco.MjrContext and mujoco.mj_render for off-screen rendering


def info_to_traj(info: list, module_idx: int = 0) -> np.ndarray:
    """Convert episode info to trajectory positions.

    Args:
        info: List of episode information dictionaries
        module_idx: Module index to extract trajectory for (0 for main body)

    Returns:
        Array of shape (n_episodes, n_steps, 2) containing 2D trajectories
    """
    n_episodes = len(info)
    n_steps = len(info[0])
    pos = np.zeros((n_episodes, n_steps, 2))

    if module_idx == 0:
        for ep_idx in range(n_episodes):
            for step_idx in range(n_steps):
                pos[ep_idx][step_idx] = info[ep_idx][step_idx][0]["next_coordinates"]
    else:
        for ep_idx in range(n_episodes):
            for step_idx in range(n_steps):
                pos[ep_idx][step_idx] = info[ep_idx][step_idx][0][
                    "next_coordinates_general"
                ][module_idx]

    # Normalize trajectories to start at origin
    pos = pos - pos[:, 0:1, :]

    return pos


def draw_2d_traj(
    loco_file: Optional[str] = None,
    trajectory: Optional[np.ndarray] = None,
    saved_figure: Optional[str] = None,
    hide_title: bool = False,
    lim: float = 10,
    cmap: str = "cool_r",
    linewidth: float = 2,
) -> None:
    """Draw 2D trajectory plot with time-based coloring.

    Args:
        loco_file: Path to .npz file containing trajectory data
        trajectory: Direct trajectory array of shape (n_episodes, n_steps, 2) or (n_steps, 2)
        saved_figure: Output figure path
        hide_title: Whether to hide plot title and axes
        lim: Plot axis limits (-lim to +lim)
        cmap: Matplotlib colormap name
        linewidth: Line width for trajectory
    """
    import matplotlib.collections as mc
    import matplotlib.pyplot as plt

    # Load trajectory data
    if loco_file is not None:
        loco_dict = np.load(loco_file)
        pos = loco_dict["positions"]
        trajectory = pos[0, :300, :2]  # Take first episode, first 300 steps
        robot_name = os.path.basename(loco_file)
    elif trajectory is not None:
        robot_name = os.path.basename(saved_figure) if saved_figure else "trajectory"
    else:
        raise ValueError("Either loco_file or trajectory must be provided")

    # Ensure trajectory has correct shape
    if trajectory.ndim == 2:
        trajectory = trajectory[
            np.newaxis, ...
        ]  # Convert (n_steps, 2) -> (1, n_steps, 2)

    if trajectory.shape[2] != 2:
        raise ValueError("Trajectory must have shape (..., n_steps, 2)")

    # Set up colormap
    try:
        cmap_obj = plt.get_cmap(cmap)
    except ValueError:
        try:
            import seaborn as sns

            cmap_obj = sns.color_palette(cmap, as_cmap=True)
        except ImportError:
            cmap_obj = plt.get_cmap("viridis")  # Fallback

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    for _i, traj in enumerate(trajectory):
        if len(traj) < 2:
            continue  # Skip trajectories with insufficient points

        # Create line segments
        points = traj.reshape(-1, 1, 2)
        segments = np.hstack([points[:-1], points[1:]])

        # Time-based coloring
        t = np.linspace(0, 1, len(traj))
        norm = plt.Normalize(t.min(), t.max())

        # Create LineCollection
        lc = mc.LineCollection(
            segments, cmap=cmap_obj, norm=norm, linewidth=linewidth, alpha=1
        )
        lc.set_capstyle("round")
        lc.set_array(t[:-1])

        ax.add_collection(lc)

    # Configure plot appearance
    ax.autoscale()
    if not hide_title:
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title(f"2D Trajectory: {robot_name}")
        if "lc" in locals():
            plt.colorbar(lc, label="Time Progress")
    else:
        ax.axis("off")

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")

    # Save figure
    if saved_figure is None and loco_file is not None:
        saved_figure = os.path.join(os.path.dirname(loco_file), f"{robot_name}_2d.pdf")

    if saved_figure is None:
        raise ValueError("saved_figure path must be provided")

    if saved_figure.endswith(".pdf"):
        plt.savefig(saved_figure, bbox_inches="tight")
    else:
        plt.savefig(saved_figure, dpi=400, bbox_inches="tight")

    plt.close()  # Free memory


if __name__ == "__main__":
    # Example usage - replace with actual XML file path
    xml_file = "path/to/your/robot.xml"
    print("Starting MuJoCo viewer...")
    print("Use this as a template for visualizing your robot models.")
    # view(xml_file, fixed=False, vis_contact=True)
