"""
Chase Visualization Utilities

This module provides reusable components for visualizing chase/tracking scenarios
with bearing information in both MuJoCo and matplotlib videos.

Features:
- MuJoCo scene markers (target, robot forward direction, bearing line)
- Video overlay with bearing/distance metrics
- Matplotlib chase video generation with bearing annotations

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


__all__ = [
    "add_chase_scene_markers",
    "add_multi_ranging_markers",
    "add_bearing_metrics_overlay",
    "compute_bearing_from_pose",
    "compute_forward_direction",
]


# =============================================================================
# Bearing and Direction Computation
# =============================================================================

def compute_forward_direction(heading: float) -> np.ndarray:
    """Compute robot's forward direction vector from heading.
    
    The heading is the angle of the robot's local +Y axis (forward) from the 
    world +X axis, measured counterclockwise.
    
    Args:
        heading: Robot's heading angle in radians.
        
    Returns:
        2D unit vector [dx, dy] pointing in the forward direction.
        
    Note:
        Forward in world frame = [cos(heading), sin(heading)]
        This is because heading measures the angle of local +Y from world +X.
    """
    return np.array([np.cos(heading), np.sin(heading)])


def compute_bearing_from_pose(
    robot_pos: np.ndarray,
    target_pos: np.ndarray,
    heading: float,
) -> float:
    """Compute bearing to target from robot's current pose.
    
    The bearing is the angle to the target in the robot's local frame:
    - 0 = target directly in front (local +Y direction)
    - >0 = target to the LEFT
    - <0 = target to the RIGHT
    
    Args:
        robot_pos: Robot's world position [x, y] or [x, y, z].
        target_pos: Target's world position [x, y].
        heading: Robot's heading angle in radians.
        
    Returns:
        Bearing in radians, range [-pi, pi].
        
    Coordinate frame explanation:
        - Robot's forward direction is LOCAL +Y axis
        - heading = arctan2(forward_world_y, forward_world_x) 
          (angle of local +Y from world +X, measured counterclockwise)
        - Local +Y in world: [cos(heading), sin(heading)] (FORWARD)
        - Local +X in world: [sin(heading), -cos(heading)] (points LEFT from forward × up)
    """
    robot_xy = robot_pos[:2] if len(robot_pos) > 2 else robot_pos
    target_xy = target_pos[:2] if len(target_pos) > 2 else target_pos
    
    # Vector from robot to target in world frame
    to_target = target_xy - robot_xy
    
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    
    # Project to_target onto robot's local axes
    # Local +X in world = [sin(heading), -cos(heading)] (points LEFT)
    # Local +Y in world = [cos(heading), sin(heading)] (points FORWARD)
    to_target_local_x = sin_h * to_target[0] - cos_h * to_target[1]  # LEFT component
    to_target_local_y = cos_h * to_target[0] + sin_h * to_target[1]  # FORWARD component
    
    # Bearing relative to forward (local +Y)
    # Negate local_x so that: +bearing = LEFT, -bearing = RIGHT
    bearing = np.arctan2(-to_target_local_x, to_target_local_y)
    
    return bearing


# =============================================================================
# MuJoCo Scene Markers
# =============================================================================

def add_chase_scene_markers(
    scene,
    robot_pos: np.ndarray,
    target_pos: np.ndarray,
    heading: float,
    bearing: Optional[float] = None,
    show_target: bool = True,
    show_forward: bool = True,
    show_robot_target_line: bool = True,
    show_robot_marker: bool = True,
    target_color: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    forward_color: Tuple[float, float, float, float] = (0.0, 1.0, 1.0, 1.0),
    line_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.9),
    robot_marker_color: Tuple[float, float, float, float] = (0.0, 0.5, 1.0, 1.0),
) -> None:
    """Add comprehensive chase visualization markers to a MuJoCo scene.
    
    This adds:
    1. Target marker (disc + sphere)
    2. Line from robot to target
    3. Robot's forward direction arrow
    4. Small marker at robot position
    
    Args:
        scene: MuJoCo mjvScene object (from renderer.scene).
        robot_pos: Robot's world position [x, y] or [x, y, z].
        target_pos: Target's world position [x, y].
        heading: Robot's heading angle in radians.
        bearing: Optional pre-computed bearing (not used for visualization, 
                 but could be used for bearing indicator in future).
        show_target: Whether to show target marker.
        show_forward: Whether to show forward direction arrow.
        show_robot_target_line: Whether to show line from robot to target.
        show_robot_marker: Whether to show robot position marker.
        target_color: RGBA color for target marker.
        forward_color: RGBA color for forward direction arrow.
        line_color: RGBA color for robot-target line.
        robot_marker_color: RGBA color for robot position marker.
    """
    from metamachine.utils.rendering import (
        add_ground_disc_marker,
        add_sphere_marker,
        add_ground_line_marker,
        add_ground_arrow_marker,
    )
    
    robot_xy = robot_pos[:2] if len(robot_pos) > 2 else robot_pos
    target_xy = target_pos[:2] if len(target_pos) > 2 else target_pos
    
    # 1. Target marker - disc and sphere
    if show_target:
        add_ground_disc_marker(
            scene,
            pos_xy=target_xy,
            radius=0.15,
            height=0.02,
            color=target_color,
            z_offset=0.01
        )
        add_sphere_marker(
            scene,
            pos=np.array([target_xy[0], target_xy[1], 0.15]),
            radius=0.08,
            color=(1.0, 0.5, 0.0, 1.0),  # Orange sphere on top
        )
    
    # 2. Line from robot to target
    if show_robot_target_line:
        add_ground_line_marker(
            scene,
            start_xy=robot_xy,
            end_xy=target_xy,
            radius=0.015,
            color=line_color,
            z_offset=0.03
        )
    
    # 3. Robot's forward direction arrow
    # Forward direction: heading is the angle of local +Y from world +X
    # So forward in world = [cos(heading), sin(heading)]
    if show_forward:
        forward_dir = compute_forward_direction(heading)
        add_ground_arrow_marker(
            scene,
            start_xy=robot_xy,
            direction_xy=forward_dir,
            length=0.6,
            shaft_radius=0.02,
            head_radius=0.05,
            head_length=0.1,
            color=forward_color,
            z_offset=0.04
        )
    
    # 4. Small marker at robot position
    if show_robot_marker:
        add_ground_disc_marker(
            scene,
            pos_xy=robot_xy,
            radius=0.08,
            height=0.02,
            color=robot_marker_color,
            z_offset=0.02
        )


def add_multi_ranging_markers(
    scene,
    module_positions: List[np.ndarray],
    target_pos: np.ndarray,
    module_colors: Optional[List[Tuple[float, float, float, float]]] = None,
) -> None:
    """Add multi-ranging visualization markers to a MuJoCo scene.
    
    Draws lines from each module position to the target, with distinct colors
    for each module.
    
    Args:
        scene: MuJoCo mjvScene object (from renderer.scene).
        module_positions: List of module world positions [x, y, z] for each module.
        target_pos: Target's world position [x, y].
        module_colors: Optional list of RGBA colors for each module line.
                      If None, uses default rainbow colors.
    """
    from metamachine.utils.rendering import (
        add_ground_line_marker,
        add_ground_disc_marker,
    )
    
    target_xy = target_pos[:2] if len(target_pos) > 2 else target_pos
    n_modules = len(module_positions)
    
    # Default colors: rainbow progression (red, green, blue for 3 modules)
    if module_colors is None:
        default_colors = [
            (1.0, 0.3, 0.3, 0.8),  # Red
            (0.3, 1.0, 0.3, 0.8),  # Green
            (0.3, 0.3, 1.0, 0.8),  # Blue
            (1.0, 1.0, 0.3, 0.8),  # Yellow
            (1.0, 0.3, 1.0, 0.8),  # Magenta
            (0.3, 1.0, 1.0, 0.8),  # Cyan
        ]
        module_colors = [default_colors[i % len(default_colors)] for i in range(n_modules)]
    
    for i, module_pos in enumerate(module_positions):
        module_xy = module_pos[:2] if len(module_pos) > 2 else module_pos
        color = module_colors[i]
        
        # Draw line from module to target
        add_ground_line_marker(
            scene,
            start_xy=module_xy,
            end_xy=target_xy,
            radius=0.01,
            color=color,
            z_offset=0.025 + i * 0.002  # Slight offset to prevent z-fighting
        )
        
        # Draw small disc at module position
        add_ground_disc_marker(
            scene,
            pos_xy=module_xy,
            radius=0.05,
            height=0.015,
            color=color,
            z_offset=0.01
        )


# =============================================================================
# Video Frame Overlay
# =============================================================================

def add_bearing_metrics_overlay(
    frame: np.ndarray,
    bearing: float,
    distance: float,
    step: int,
    heading: float,
    controller_idx: Optional[int] = None,
    controller_name: Optional[str] = None,
    bearing_source: str = "ground_truth",
    position: str = "right",
    start_y: int = 150,
    uncertainty_rad: Optional[float] = None,
    pred_vel_xy: Optional[Tuple[float, float]] = None,
    pred_yaw_rate: Optional[float] = None,
) -> np.ndarray:
    """Add bearing and distance metrics overlay to a video frame.
    
    Args:
        frame: Video frame (BGR format from OpenCV).
        bearing: Current bearing to target in radians.
        distance: Current distance to target in meters.
        step: Current step number.
        heading: Robot's heading angle in radians.
        controller_idx: Optional controller index.
        controller_name: Optional controller name.
        bearing_source: Source of bearing ("ground_truth" or "estimated").
        position: Position of overlay ("right" or "left").
        start_y: Y coordinate to start the overlay.
        uncertainty_rad: Optional predicted bearing std (radians).
        pred_vel_xy: Optional predicted local velocity (vx, vy).
        pred_yaw_rate: Optional predicted yaw rate (rad/s).
        
    Returns:
        Frame with metrics overlay added.
    """
    if not HAS_CV2:
        return frame
    
    # Bearing in degrees with direction indicator
    bearing_deg = np.degrees(bearing)
    heading_deg = np.degrees(heading)
    
    if bearing_deg > 5:
        direction = "LEFT"
    elif bearing_deg < -5:
        direction = "RIGHT"
    else:
        direction = "FRONT"
    
    # Build metrics list
    source_label = "GT" if bearing_source == "ground_truth" else "EST"
    metrics = [
        f"=== BEARING ({source_label}) ===",
        f"Bearing: {bearing_deg:+7.1f} deg",
        f"Direction: {direction}",
        f"Distance: {distance:.2f} m",
        f"Heading: {heading_deg:+7.1f} deg",
        f"Step: {step}",
    ]

    if uncertainty_rad is not None:
        metrics.append(f"Std: {np.degrees(uncertainty_rad):6.1f} deg")
    if pred_vel_xy is not None:
        try:
            vx, vy = pred_vel_xy
            metrics.append(f"Pred vxy: {vx:+.2f}, {vy:+.2f} m/s")
        except (TypeError, ValueError):
            pass
    if pred_yaw_rate is not None:
        metrics.append(f"Pred yaw: {np.degrees(pred_yaw_rate):+6.1f} deg/s")
    
    # Add controller info if provided
    if controller_idx is not None and controller_name is not None:
        metrics.append("")
        metrics.append("=== CONTROLLER ===")
        metrics.append(f"[{controller_idx}] {controller_name}")
    
    # Configure text appearance
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (0, 255, 255)  # Cyan text
    thickness = 2
    line_height = 25
    
    # Position calculation
    frame_width = frame.shape[1]
    if position == "right":
        start_x = frame_width - 280
    else:
        start_x = 20
    
    # Draw semi-transparent background
    overlay = frame.copy()
    bg_height = len(metrics) * line_height + 15
    bg_width = 270
    
    if position == "right":
        cv2.rectangle(
            overlay, 
            (start_x - 10, start_y - 20), 
            (frame_width - 10, start_y + bg_height - 10), 
            (0, 0, 0), 
            -1
        )
    else:
        cv2.rectangle(
            overlay, 
            (start_x - 10, start_y - 20), 
            (start_x + bg_width, start_y + bg_height - 10), 
            (0, 0, 0), 
            -1
        )
    
    frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    
    # Draw text
    for i, text in enumerate(metrics):
        y_pos = start_y + (i * line_height)
        cv2.putText(frame, text, (start_x, y_pos), font, font_scale, color, thickness)
    
    return frame


# =============================================================================
# Matplotlib Chase Video Utilities  
# =============================================================================

def draw_chase_frame_matplotlib(
    ax,
    robot_pos: np.ndarray,
    target_pos: np.ndarray,
    heading: float,
    bearing: float,
    controller_idx: int,
    controller_name: str,
    controller_names: List[str],
    controller_colors: np.ndarray,
    step: int,
    episode: int,
    distance: float,
    trajectory_history: Optional[List[Dict]] = None,
    bearing_source: str = "ground_truth",
    show_bearing_info: bool = True,
    anchor_pos: Optional[np.ndarray] = None,
    robot_anchor_distance: Optional[float] = None,
    anchor_target_distance: Optional[float] = None,
    x_center: float = 0.0,
    y_center: float = 0.0,
    max_range: float = 10.0,
) -> None:
    """Draw a single frame of the chase visualization in matplotlib.
    
    Args:
        ax: Matplotlib axes object.
        robot_pos: Robot's world position [x, y].
        target_pos: Target's world position [x, y].
        heading: Robot's heading angle in radians.
        bearing: Current bearing to target in radians.
        controller_idx: Current controller index.
        controller_name: Current controller name.
        controller_names: List of all controller names.
        controller_colors: Array of colors for each controller.
        step: Current step number.
        episode: Current episode number.
        distance: Current distance to target.
        trajectory_history: Optional list of previous frame data for trail.
        bearing_source: Source of bearing ("ground_truth" or "estimated").
        show_bearing_info: Whether to show bearing information.
        anchor_pos: Optional anchor position for triangulation mode.
        robot_anchor_distance: Optional robot-to-anchor distance.
        anchor_target_distance: Optional anchor-to-target distance.
        x_center: X center of the plot.
        y_center: Y center of the plot.
        max_range: Range of the plot (half-width).
    """
    ax.clear()
    ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    rx, ry = robot_pos[0], robot_pos[1]
    tx, ty = target_pos[0], target_pos[1]
    
    # Plot trajectory history with controller colors
    if trajectory_history:
        for i in range(len(trajectory_history) - 1):
            f1 = trajectory_history[i]
            f2 = trajectory_history[i + 1]
            color = controller_colors[f1['controller_idx']]
            ax.plot(
                [f1['robot_pos'][0], f2['robot_pos'][0]],
                [f1['robot_pos'][1], f2['robot_pos'][1]],
                color=color, alpha=0.5, linewidth=2
            )
        
        # Target trail
        target_trail_x = [f['target_pos'][0] for f in trajectory_history]
        target_trail_y = [f['target_pos'][1] for f in trajectory_history]
        ax.plot(target_trail_x, target_trail_y, 'r-', alpha=0.3, linewidth=1)
    
    # Plot robot with current controller color
    robot_color = controller_colors[controller_idx]
    ax.plot(rx, ry, 'o', color=robot_color, markersize=12)
    
    # Draw robot's forward direction arrow (CORRECT: using cos/sin of heading)
    # heading is the angle of local +Y from world +X, so forward = [cos(heading), sin(heading)]
    forward_dir = compute_forward_direction(heading)
    arrow_length = 0.3
    ax.arrow(
        rx, ry, 
        arrow_length * forward_dir[0], 
        arrow_length * forward_dir[1],
        head_width=0.15, head_length=0.1, 
        fc=robot_color, ec=robot_color
    )
    
    # Plot target
    ax.plot(tx, ty, 'rx', markersize=15, markeredgewidth=3)
    
    # Plot anchor if present
    if anchor_pos is not None:
        anc_x, anc_y = anchor_pos[0], anchor_pos[1]
        ax.plot(anc_x, anc_y, 's', color='purple', markersize=12, markeredgewidth=2, label='Anchor')
        # Draw lines showing triangulation
        ax.plot([rx, anc_x], [ry, anc_y], 'purple', alpha=0.3, linestyle=':', linewidth=1)
        ax.plot([anc_x, tx], [anc_y, ty], 'purple', alpha=0.3, linestyle=':', linewidth=1)
    
    # Draw line between robot and target
    ax.plot([rx, tx], [ry, ty], 'g--', alpha=0.5)
    
    # Build title
    bearing_deg = np.degrees(bearing)
    source_label = "GT" if bearing_source == "ground_truth" else "EST"
    
    if anchor_pos is not None and robot_anchor_distance is not None:
        title = (
            f"Policy Switch Chase (Triangulation) - Episode {episode}\n"
            f"Step {step}, R→T: {distance:.2f}m, "
            f"R→A: {robot_anchor_distance:.2f}m, A→T: {anchor_target_distance:.2f}m\n"
            f"Active: {controller_name} | Bearing ({source_label}): {bearing_deg:+.1f}°"
        )
    else:
        direction = "LEFT" if bearing_deg > 5 else ("RIGHT" if bearing_deg < -5 else "FRONT")
        title = (
            f"Policy Switch Chase - Episode {episode}\n"
            f"Step {step}, Distance: {distance:.2f}m\n"
            f"Active: {controller_name} | Bearing ({source_label}): {bearing_deg:+.1f}° ({direction})"
        )
    ax.set_title(title)
    
    # Legend for controllers with bearing info
    num_controllers = len(controller_names)
    for i, name in enumerate(controller_names):
        marker = '●' if i == controller_idx else '○'
        ax.text(
            x_center + max_range/2 - 0.5,
            y_center + max_range/2 - 0.5 - i * 0.3,
            f"{marker} {name}",
            color=controller_colors[i],
            fontsize=10,
            ha='right'
        )
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
