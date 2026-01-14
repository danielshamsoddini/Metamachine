#!/usr/bin/env python3
"""
Sim-to-Real Comprehensive State Analysis Tool

This script compares full state data from simulation and real robot
to diagnose behavioral differences beyond just actuation.

Analysis includes:
1. Actuation Analysis (joint tracking, commands vs positions)
2. Velocity Analysis (linear and angular velocities in body/world frames)
3. Orientation Analysis (quaternion, projected gravity)
4. Behavioral Metrics (turning rate, forward speed, etc.)
5. Frequency Domain Analysis
6. Episode-wise Statistics

Usage:
    python scripts/sim2real_state_analysis.py sim_state.npz real_state.npz
    python scripts/sim2real_state_analysis.py sim_state.npz real_state.npz --output analysis/
    python scripts/sim2real_state_analysis.py sim_state.npz real_state.npz --no-show

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr
from scipy.spatial.transform import Rotation


# =============================================================================
# Data Loading
# =============================================================================

def load_state_data(filepath: str) -> dict:
    """Load state data from .npz file."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    data = dict(np.load(filepath, allow_pickle=True))
    
    # Handle numpy object arrays for lists
    for key in ['joint_names', 'default_dof_pos']:
        if key in data and isinstance(data[key], np.ndarray):
            if data[key].ndim == 0:
                data[key] = data[key].item()
    
    return data


def get_episode_slices(episodes: np.ndarray) -> List[Tuple[int, int, int]]:
    """Get start and end indices for each episode.
    
    Returns:
        List of (episode_id, start_idx, end_idx) tuples
    """
    slices = []
    current_ep = episodes[0]
    start_idx = 0
    
    for i, ep in enumerate(episodes):
        if ep != current_ep:
            slices.append((current_ep, start_idx, i))
            current_ep = ep
            start_idx = i
    
    # Last episode
    slices.append((current_ep, start_idx, len(episodes)))
    
    return slices


# =============================================================================
# Quaternion Utilities
# =============================================================================

def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to euler angles (roll, pitch, yaw)."""
    # Convert from (w, x, y, z) to scipy format (x, y, z, w)
    if quat.ndim == 1:
        quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
        return Rotation.from_quat(quat_scipy).as_euler('xyz')
    else:
        quat_scipy = np.column_stack([quat[:, 1], quat[:, 2], quat[:, 3], quat[:, 0]])
        return Rotation.from_quat(quat_scipy).as_euler('xyz')


def compute_heading_from_quat(quat: np.ndarray) -> np.ndarray:
    """Extract heading (yaw) angle from quaternion time series."""
    euler = quat_to_euler(quat)
    return euler[:, 2]  # yaw


def unwrap_angle(angles: np.ndarray) -> np.ndarray:
    """Unwrap angle series to avoid discontinuities."""
    return np.unwrap(angles)


# =============================================================================
# Analysis Functions
# =============================================================================

def compute_actuation_metrics(data: dict) -> dict:
    """Compute actuation-level metrics."""
    commands = data['joint_commands']
    positions = data['dof_pos']
    velocities = data['dof_vel']
    timestamps = data['timestamps']
    
    num_joints = commands.shape[1]
    tracking_error = positions - commands
    
    dt = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 0.05
    
    metrics = {
        'rmse': [],
        'max_error': [],
        'mean_error': [],
        'std_error': [],
        'delay_samples': [],
        'delay_time_ms': [],
        'correlation': [],
        'vel_rms': [],
        'vel_max': [],
    }
    
    for j in range(num_joints):
        error = tracking_error[:, j]
        cmd = commands[:, j]
        pos = positions[:, j]
        vel = velocities[:, j]
        
        # Basic error metrics
        metrics['rmse'].append(np.sqrt(np.mean(error ** 2)))
        metrics['max_error'].append(np.max(np.abs(error)))
        metrics['mean_error'].append(np.mean(error))
        metrics['std_error'].append(np.std(error))
        
        # Velocity metrics
        metrics['vel_rms'].append(np.sqrt(np.mean(vel ** 2)))
        metrics['vel_max'].append(np.max(np.abs(vel)))
        
        # Estimate delay using cross-correlation
        if np.std(cmd) > 1e-6 and np.std(pos) > 1e-6:
            correlation = np.correlate(pos - np.mean(pos), cmd - np.mean(cmd), mode='full')
            delay_idx = np.argmax(correlation) - (len(cmd) - 1)
            metrics['delay_samples'].append(delay_idx)
            metrics['delay_time_ms'].append(delay_idx * dt * 1000)
            
            corr, _ = pearsonr(cmd, pos)
            metrics['correlation'].append(corr)
        else:
            metrics['delay_samples'].append(0)
            metrics['delay_time_ms'].append(0.0)
            metrics['correlation'].append(0.0)
    
    return metrics


def compute_velocity_metrics(data: dict) -> dict:
    """Compute velocity-related metrics (key for turning analysis)."""
    vel_body = data['vel_body']
    ang_vel_body = data['ang_vel_body']
    timestamps = data['timestamps']
    episodes = data['episode']
    
    dt = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 0.05
    
    # Body frame velocities
    # vel_body: [vx, vy, vz] in body frame (forward, left, up typically)
    # ang_vel_body: [wx, wy, wz] in body frame (roll rate, pitch rate, yaw rate)
    
    forward_vel = vel_body[:, 0]  # Assuming x is forward
    lateral_vel = vel_body[:, 1]  # Assuming y is lateral
    vertical_vel = vel_body[:, 2]
    
    roll_rate = ang_vel_body[:, 0]
    pitch_rate = ang_vel_body[:, 1]
    yaw_rate = ang_vel_body[:, 2]  # This is the turning rate!
    
    # Per-episode statistics
    episode_slices = get_episode_slices(episodes)
    
    episode_stats = []
    for ep_id, start, end in episode_slices:
        ep_forward = forward_vel[start:end]
        ep_yaw = yaw_rate[start:end]
        ep_time = timestamps[end-1] - timestamps[start]
        
        # Compute cumulative heading change
        ep_heading_change = np.sum(np.abs(ep_yaw)) * dt
        
        episode_stats.append({
            'episode': ep_id,
            'duration': ep_time,
            'mean_forward_vel': np.mean(ep_forward),
            'mean_yaw_rate': np.mean(ep_yaw),
            'mean_abs_yaw_rate': np.mean(np.abs(ep_yaw)),
            'max_yaw_rate': np.max(np.abs(ep_yaw)),
            'total_heading_change': ep_heading_change,
            'mean_forward_speed': np.mean(np.abs(ep_forward)),
        })
    
    metrics = {
        # Overall velocity stats
        'forward_vel_mean': np.mean(forward_vel),
        'forward_vel_std': np.std(forward_vel),
        'forward_vel_max': np.max(np.abs(forward_vel)),
        'lateral_vel_mean': np.mean(lateral_vel),
        'lateral_vel_std': np.std(lateral_vel),
        'vertical_vel_std': np.std(vertical_vel),
        
        # Angular velocity stats (KEY for turning)
        'yaw_rate_mean': np.mean(yaw_rate),
        'yaw_rate_abs_mean': np.mean(np.abs(yaw_rate)),
        'yaw_rate_std': np.std(yaw_rate),
        'yaw_rate_max': np.max(np.abs(yaw_rate)),
        'roll_rate_std': np.std(roll_rate),
        'pitch_rate_std': np.std(pitch_rate),
        
        # Time series for plotting
        'forward_vel': forward_vel,
        'lateral_vel': lateral_vel,
        'yaw_rate': yaw_rate,
        'roll_rate': roll_rate,
        'pitch_rate': pitch_rate,
        
        # Episode stats
        'episode_stats': episode_stats,
    }
    
    return metrics


def compute_orientation_metrics(data: dict) -> dict:
    """Compute orientation-related metrics."""
    quat = data['quat']
    projected_gravity = data['projected_gravity']
    timestamps = data['timestamps']
    
    # Convert quaternion to euler angles
    euler = quat_to_euler(quat)
    roll = euler[:, 0]
    pitch = euler[:, 1]
    yaw = euler[:, 2]
    
    # Unwrap yaw to track cumulative rotation
    yaw_unwrapped = unwrap_angle(yaw)
    
    # Total rotation over the recording
    total_rotation = yaw_unwrapped[-1] - yaw_unwrapped[0]
    duration = timestamps[-1] - timestamps[0]
    avg_turning_rate = total_rotation / duration if duration > 0 else 0
    
    metrics = {
        'roll_mean': np.mean(roll),
        'roll_std': np.std(roll),
        'roll_max': np.max(np.abs(roll)),
        'pitch_mean': np.mean(pitch),
        'pitch_std': np.std(pitch),
        'pitch_max': np.max(np.abs(pitch)),
        'yaw_range': np.max(yaw) - np.min(yaw),
        'total_rotation_rad': total_rotation,
        'total_rotation_deg': np.degrees(total_rotation),
        'avg_turning_rate_rad': avg_turning_rate,
        'avg_turning_rate_deg': np.degrees(avg_turning_rate),
        
        # Stability metrics from projected gravity
        'gravity_x_std': np.std(projected_gravity[:, 0]),
        'gravity_y_std': np.std(projected_gravity[:, 1]),
        'gravity_z_mean': np.mean(projected_gravity[:, 2]),
        
        # Time series
        'roll': roll,
        'pitch': pitch,
        'yaw': yaw,
        'yaw_unwrapped': yaw_unwrapped,
    }
    
    return metrics


def compute_behavioral_comparison(sim_data: dict, real_data: dict) -> dict:
    """Compute high-level behavioral comparison metrics."""
    
    sim_vel = compute_velocity_metrics(sim_data)
    real_vel = compute_velocity_metrics(real_data)
    
    sim_orient = compute_orientation_metrics(sim_data)
    real_orient = compute_orientation_metrics(real_data)
    
    # KEY METRIC: Turning rate ratio
    sim_yaw_rate = sim_vel['yaw_rate_abs_mean']
    real_yaw_rate = real_vel['yaw_rate_abs_mean']
    turning_ratio = sim_yaw_rate / real_yaw_rate if real_yaw_rate > 1e-6 else float('inf')
    
    # Forward velocity ratio
    sim_forward = np.abs(sim_vel['forward_vel_mean'])
    real_forward = np.abs(real_vel['forward_vel_mean'])
    forward_ratio = sim_forward / real_forward if real_forward > 1e-6 else float('inf')
    
    # Total rotation comparison
    sim_rotation = np.abs(sim_orient['total_rotation_deg'])
    real_rotation = np.abs(real_orient['total_rotation_deg'])
    rotation_ratio = sim_rotation / real_rotation if real_rotation > 1e-6 else float('inf')
    
    comparison = {
        'sim_velocity': sim_vel,
        'real_velocity': real_vel,
        'sim_orientation': sim_orient,
        'real_orientation': real_orient,
        
        # Key ratios
        'turning_rate_ratio': turning_ratio,
        'forward_velocity_ratio': forward_ratio,
        'total_rotation_ratio': rotation_ratio,
        
        # Absolute differences
        'yaw_rate_diff': sim_yaw_rate - real_yaw_rate,
        'forward_vel_diff': sim_forward - real_forward,
    }
    
    return comparison


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_actuation_comparison(sim_data: dict, real_data: dict, 
                              output_dir: Optional[str] = None) -> None:
    """Plot joint actuation comparison."""
    sim_times = sim_data['timestamps'] - sim_data['timestamps'][0]
    real_times = real_data['timestamps'] - real_data['timestamps'][0]
    
    sim_cmds = sim_data['joint_commands']
    sim_pos = sim_data['dof_pos']
    real_cmds = real_data['joint_commands']
    real_pos = real_data['dof_pos']
    
    num_joints = sim_cmds.shape[1]
    joint_names = sim_data.get('joint_names', [f'Joint {i}' for i in range(num_joints)])
    if isinstance(joint_names, np.ndarray):
        joint_names = joint_names.tolist()
    
    fig, axes = plt.subplots(num_joints, 2, figsize=(16, 3.5 * num_joints))
    if num_joints == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Sim vs Real: Joint Actuation Comparison', fontsize=14, fontweight='bold')
    
    for j in range(num_joints):
        # Simulation
        ax_sim = axes[j, 0]
        ax_sim.plot(sim_times, sim_cmds[:, j], 'b-', label='Command', linewidth=1, alpha=0.8)
        ax_sim.plot(sim_times, sim_pos[:, j], 'r-', label='Position', linewidth=1, alpha=0.8)
        ax_sim.fill_between(sim_times, sim_cmds[:, j], sim_pos[:, j], alpha=0.3, color='purple')
        ax_sim.set_ylabel(f'{joint_names[j]}\n(rad)', fontsize=10)
        ax_sim.legend(loc='upper right', fontsize=8)
        ax_sim.grid(True, alpha=0.3)
        if j == 0:
            ax_sim.set_title('SIMULATION', fontsize=12, fontweight='bold')
        if j == num_joints - 1:
            ax_sim.set_xlabel('Time (s)')
        
        # Real
        ax_real = axes[j, 1]
        ax_real.plot(real_times, real_cmds[:, j], 'b-', label='Command', linewidth=1, alpha=0.8)
        ax_real.plot(real_times, real_pos[:, j], 'r-', label='Position', linewidth=1, alpha=0.8)
        ax_real.fill_between(real_times, real_cmds[:, j], real_pos[:, j], alpha=0.3, color='purple')
        ax_real.legend(loc='upper right', fontsize=8)
        ax_real.grid(True, alpha=0.3)
        if j == 0:
            ax_real.set_title('REAL ROBOT', fontsize=12, fontweight='bold')
        if j == num_joints - 1:
            ax_real.set_xlabel('Time (s)')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(Path(output_dir) / 'actuation_comparison.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: {Path(output_dir) / 'actuation_comparison.png'}")


def plot_velocity_comparison(sim_data: dict, real_data: dict,
                             comparison: dict,
                             output_dir: Optional[str] = None) -> None:
    """Plot velocity comparison - KEY for turning analysis."""
    sim_times = sim_data['timestamps'] - sim_data['timestamps'][0]
    real_times = real_data['timestamps'] - real_data['timestamps'][0]
    
    sim_vel = comparison['sim_velocity']
    real_vel = comparison['real_velocity']
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    
    fig.suptitle('Sim vs Real: Velocity Comparison (KEY FOR TURNING ANALYSIS)', 
                 fontsize=14, fontweight='bold')
    
    # Row 1: Forward velocity
    axes[0, 0].plot(sim_times, sim_vel['forward_vel'], 'b-', linewidth=0.8)
    axes[0, 0].set_ylabel('Forward Vel\n(m/s)')
    axes[0, 0].set_title('SIMULATION', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(real_times, real_vel['forward_vel'], 'r-', linewidth=0.8)
    axes[0, 1].set_title('REAL ROBOT', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Row 2: Lateral velocity
    axes[1, 0].plot(sim_times, sim_vel['lateral_vel'], 'b-', linewidth=0.8)
    axes[1, 0].set_ylabel('Lateral Vel\n(m/s)')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(real_times, real_vel['lateral_vel'], 'r-', linewidth=0.8)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Row 3: YAW RATE (KEY!)
    axes[2, 0].plot(sim_times, sim_vel['yaw_rate'], 'b-', linewidth=0.8)
    axes[2, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[2, 0].set_ylabel('Yaw Rate\n(rad/s)', fontweight='bold', color='darkred')
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(real_times, real_vel['yaw_rate'], 'r-', linewidth=0.8)
    axes[2, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[2, 1].grid(True, alpha=0.3)
    
    # Add yaw rate statistics as text
    sim_yaw_text = f"Mean: {sim_vel['yaw_rate_abs_mean']:.3f} rad/s\nMax: {sim_vel['yaw_rate_max']:.3f} rad/s"
    real_yaw_text = f"Mean: {real_vel['yaw_rate_abs_mean']:.3f} rad/s\nMax: {real_vel['yaw_rate_max']:.3f} rad/s"
    axes[2, 0].text(0.98, 0.98, sim_yaw_text, transform=axes[2, 0].transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[2, 1].text(0.98, 0.98, real_yaw_text, transform=axes[2, 1].transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Row 4: Roll and Pitch rates
    axes[3, 0].plot(sim_times, sim_vel['roll_rate'], 'g-', linewidth=0.8, label='Roll rate', alpha=0.7)
    axes[3, 0].plot(sim_times, sim_vel['pitch_rate'], 'm-', linewidth=0.8, label='Pitch rate', alpha=0.7)
    axes[3, 0].set_ylabel('Roll/Pitch Rate\n(rad/s)')
    axes[3, 0].set_xlabel('Time (s)')
    axes[3, 0].legend(loc='upper right', fontsize=8)
    axes[3, 0].grid(True, alpha=0.3)
    
    axes[3, 1].plot(real_times, real_vel['roll_rate'], 'g-', linewidth=0.8, label='Roll rate', alpha=0.7)
    axes[3, 1].plot(real_times, real_vel['pitch_rate'], 'm-', linewidth=0.8, label='Pitch rate', alpha=0.7)
    axes[3, 1].set_xlabel('Time (s)')
    axes[3, 1].legend(loc='upper right', fontsize=8)
    axes[3, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(Path(output_dir) / 'velocity_comparison.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: {Path(output_dir) / 'velocity_comparison.png'}")


def plot_orientation_comparison(sim_data: dict, real_data: dict,
                                comparison: dict,
                                output_dir: Optional[str] = None) -> None:
    """Plot orientation comparison."""
    sim_times = sim_data['timestamps'] - sim_data['timestamps'][0]
    real_times = real_data['timestamps'] - real_data['timestamps'][0]
    
    sim_orient = comparison['sim_orientation']
    real_orient = comparison['real_orientation']
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 10))
    
    fig.suptitle('Sim vs Real: Orientation Comparison', fontsize=14, fontweight='bold')
    
    # Roll
    axes[0, 0].plot(sim_times, np.degrees(sim_orient['roll']), 'b-', linewidth=0.8)
    axes[0, 0].set_ylabel('Roll (deg)')
    axes[0, 0].set_title('SIMULATION', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(real_times, np.degrees(real_orient['roll']), 'r-', linewidth=0.8)
    axes[0, 1].set_title('REAL ROBOT', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Pitch
    axes[1, 0].plot(sim_times, np.degrees(sim_orient['pitch']), 'b-', linewidth=0.8)
    axes[1, 0].set_ylabel('Pitch (deg)')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(real_times, np.degrees(real_orient['pitch']), 'r-', linewidth=0.8)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Cumulative Yaw (unwrapped)
    axes[2, 0].plot(sim_times, np.degrees(sim_orient['yaw_unwrapped']), 'b-', linewidth=1.2)
    axes[2, 0].set_ylabel('Cumulative Yaw (deg)', fontweight='bold')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].grid(True, alpha=0.3)
    total_sim = sim_orient['total_rotation_deg']
    axes[2, 0].text(0.98, 0.02, f"Total: {total_sim:.1f}°", transform=axes[2, 0].transAxes,
                    fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    axes[2, 1].plot(real_times, np.degrees(real_orient['yaw_unwrapped']), 'r-', linewidth=1.2)
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].grid(True, alpha=0.3)
    total_real = real_orient['total_rotation_deg']
    axes[2, 1].text(0.98, 0.02, f"Total: {total_real:.1f}°", transform=axes[2, 1].transAxes,
                    fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(Path(output_dir) / 'orientation_comparison.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: {Path(output_dir) / 'orientation_comparison.png'}")


def plot_yaw_rate_histogram(comparison: dict, output_dir: Optional[str] = None) -> None:
    """Plot yaw rate distribution comparison."""
    sim_yaw = comparison['sim_velocity']['yaw_rate']
    real_yaw = comparison['real_velocity']['yaw_rate']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    fig.suptitle('Yaw Rate Distribution Comparison', fontsize=14, fontweight='bold')
    
    # Histograms
    bins = np.linspace(min(sim_yaw.min(), real_yaw.min()), 
                       max(sim_yaw.max(), real_yaw.max()), 50)
    
    axes[0].hist(sim_yaw, bins=bins, alpha=0.7, label='Simulation', color='blue', density=True)
    axes[0].hist(real_yaw, bins=bins, alpha=0.7, label='Real Robot', color='red', density=True)
    axes[0].set_xlabel('Yaw Rate (rad/s)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Yaw Rate Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot comparison
    axes[1].boxplot([sim_yaw, real_yaw], tick_labels=['Simulation', 'Real Robot'])
    axes[1].set_ylabel('Yaw Rate (rad/s)')
    axes[1].set_title('Yaw Rate Statistics')
    axes[1].grid(True, alpha=0.3)
    
    # Add ratio annotation
    ratio = comparison['turning_rate_ratio']
    axes[1].text(0.5, 0.98, f"Sim/Real Ratio: {ratio:.2f}x", transform=axes[1].transAxes,
                 fontsize=12, verticalalignment='top', horizontalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8), fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(Path(output_dir) / 'yaw_rate_distribution.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: {Path(output_dir) / 'yaw_rate_distribution.png'}")


def plot_action_analysis(sim_data: dict, real_data: dict,
                         output_dir: Optional[str] = None) -> None:
    """Plot raw action comparison to see if commands are similar."""
    sim_times = sim_data['timestamps'] - sim_data['timestamps'][0]
    real_times = real_data['timestamps'] - real_data['timestamps'][0]
    
    sim_actions = sim_data['actions']
    real_actions = real_data['actions']
    
    num_actions = sim_actions.shape[1]
    
    fig, axes = plt.subplots(num_actions, 2, figsize=(16, 3 * num_actions))
    if num_actions == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Sim vs Real: Raw Policy Actions', fontsize=14, fontweight='bold')
    
    for j in range(num_actions):
        axes[j, 0].plot(sim_times, sim_actions[:, j], 'b-', linewidth=0.8)
        axes[j, 0].set_ylabel(f'Action {j}')
        axes[j, 0].grid(True, alpha=0.3)
        if j == 0:
            axes[j, 0].set_title('SIMULATION', fontweight='bold')
        if j == num_actions - 1:
            axes[j, 0].set_xlabel('Time (s)')
        
        axes[j, 1].plot(real_times, real_actions[:, j], 'r-', linewidth=0.8)
        axes[j, 1].grid(True, alpha=0.3)
        if j == 0:
            axes[j, 1].set_title('REAL ROBOT', fontweight='bold')
        if j == num_actions - 1:
            axes[j, 1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(Path(output_dir) / 'action_comparison.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: {Path(output_dir) / 'action_comparison.png'}")


def plot_frequency_analysis(sim_data: dict, real_data: dict,
                            comparison: dict,
                            output_dir: Optional[str] = None) -> None:
    """Plot frequency domain analysis of yaw rate."""
    sim_times = sim_data['timestamps']
    real_times = real_data['timestamps']
    
    sim_yaw = comparison['sim_velocity']['yaw_rate']
    real_yaw = comparison['real_velocity']['yaw_rate']
    
    sim_dt = np.mean(np.diff(sim_times))
    real_dt = np.mean(np.diff(real_times))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    fig.suptitle('Frequency Domain Analysis', fontsize=14, fontweight='bold')
    
    # Yaw rate PSD
    f_sim, psd_sim = signal.welch(sim_yaw, fs=1/sim_dt, nperseg=min(512, len(sim_yaw)//2))
    f_real, psd_real = signal.welch(real_yaw, fs=1/real_dt, nperseg=min(512, len(real_yaw)//2))
    
    axes[0, 0].semilogy(f_sim, psd_sim, 'b-', label='Simulation', linewidth=1.5)
    axes[0, 0].semilogy(f_real, psd_real, 'r-', label='Real Robot', linewidth=1.5)
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('PSD')
    axes[0, 0].set_title('Yaw Rate Power Spectral Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, 20)
    
    # Forward velocity PSD
    sim_fwd = comparison['sim_velocity']['forward_vel']
    real_fwd = comparison['real_velocity']['forward_vel']
    
    f_sim_fwd, psd_sim_fwd = signal.welch(sim_fwd, fs=1/sim_dt, nperseg=min(512, len(sim_fwd)//2))
    f_real_fwd, psd_real_fwd = signal.welch(real_fwd, fs=1/real_dt, nperseg=min(512, len(real_fwd)//2))
    
    axes[0, 1].semilogy(f_sim_fwd, psd_sim_fwd, 'b-', label='Simulation', linewidth=1.5)
    axes[0, 1].semilogy(f_real_fwd, psd_real_fwd, 'r-', label='Real Robot', linewidth=1.5)
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('PSD')
    axes[0, 1].set_title('Forward Velocity Power Spectral Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 20)
    
    # Joint position PSD (first joint)
    sim_pos = sim_data['dof_pos'][:, 0]
    real_pos = real_data['dof_pos'][:, 0]
    
    f_sim_pos, psd_sim_pos = signal.welch(sim_pos, fs=1/sim_dt, nperseg=min(512, len(sim_pos)//2))
    f_real_pos, psd_real_pos = signal.welch(real_pos, fs=1/real_dt, nperseg=min(512, len(real_pos)//2))
    
    axes[1, 0].semilogy(f_sim_pos, psd_sim_pos, 'b-', label='Simulation', linewidth=1.5)
    axes[1, 0].semilogy(f_real_pos, psd_real_pos, 'r-', label='Real Robot', linewidth=1.5)
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('PSD')
    axes[1, 0].set_title('Joint 0 Position Power Spectral Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 20)
    
    # Action PSD (first action)
    sim_act = sim_data['actions'][:, 0]
    real_act = real_data['actions'][:, 0]
    
    f_sim_act, psd_sim_act = signal.welch(sim_act, fs=1/sim_dt, nperseg=min(512, len(sim_act)//2))
    f_real_act, psd_real_act = signal.welch(real_act, fs=1/real_dt, nperseg=min(512, len(real_act)//2))
    
    axes[1, 1].semilogy(f_sim_act, psd_sim_act, 'b-', label='Simulation', linewidth=1.5)
    axes[1, 1].semilogy(f_real_act, psd_real_act, 'r-', label='Real Robot', linewidth=1.5)
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('PSD')
    axes[1, 1].set_title('Action 0 Power Spectral Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 20)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(Path(output_dir) / 'frequency_analysis.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: {Path(output_dir) / 'frequency_analysis.png'}")


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(sim_data: dict, real_data: dict, comparison: dict,
                    sim_actuation: dict, real_actuation: dict,
                    output_dir: Optional[str] = None) -> str:
    """Generate comprehensive analysis report."""
    
    report_lines = []
    
    def add(line=""):
        report_lines.append(line)
    
    add("=" * 80)
    add("SIM-TO-REAL COMPREHENSIVE STATE ANALYSIS REPORT")
    add("=" * 80)
    add(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add()
    
    # Data summary
    add("-" * 80)
    add("DATA SUMMARY")
    add("-" * 80)
    sim_duration = sim_data['timestamps'][-1] - sim_data['timestamps'][0]
    real_duration = real_data['timestamps'][-1] - real_data['timestamps'][0]
    sim_dt = np.mean(np.diff(sim_data['timestamps']))
    real_dt = np.mean(np.diff(real_data['timestamps']))
    
    add(f"  Simulation:")
    add(f"    Duration: {sim_duration:.2f} s")
    add(f"    Steps: {len(sim_data['timestamps'])}")
    add(f"    Control frequency: {1/sim_dt:.1f} Hz (dt={sim_dt*1000:.1f}ms)")
    add(f"    Episodes: {len(np.unique(sim_data['episode']))}")
    add()
    add(f"  Real Robot:")
    add(f"    Duration: {real_duration:.2f} s")
    add(f"    Steps: {len(real_data['timestamps'])}")
    add(f"    Control frequency: {1/real_dt:.1f} Hz (dt={real_dt*1000:.1f}ms)")
    add(f"    Episodes: {len(np.unique(real_data['episode']))}")
    add()
    
    # KEY FINDINGS - Behavioral Gap
    add("=" * 80)
    add("KEY FINDINGS: BEHAVIORAL GAP ANALYSIS")
    add("=" * 80)
    add()
    
    turning_ratio = comparison['turning_rate_ratio']
    forward_ratio = comparison['forward_velocity_ratio']
    rotation_ratio = comparison['total_rotation_ratio']
    
    sim_vel = comparison['sim_velocity']
    real_vel = comparison['real_velocity']
    sim_orient = comparison['sim_orientation']
    real_orient = comparison['real_orientation']
    
    add("  ┌─────────────────────────────────────────────────────────────────────┐")
    add(f"  │  TURNING RATE RATIO (Sim/Real): {turning_ratio:.2f}x                           │")
    add("  └─────────────────────────────────────────────────────────────────────┘")
    add()
    add(f"  This means simulation turns {turning_ratio:.1f}x {'FASTER' if turning_ratio > 1 else 'SLOWER'} than real robot!")
    add()
    
    add("  Yaw Rate Statistics:")
    add(f"    ├── Simulation:")
    add(f"    │     Mean |yaw_rate|: {sim_vel['yaw_rate_abs_mean']:.4f} rad/s ({np.degrees(sim_vel['yaw_rate_abs_mean']):.2f} deg/s)")
    add(f"    │     Max |yaw_rate|:  {sim_vel['yaw_rate_max']:.4f} rad/s ({np.degrees(sim_vel['yaw_rate_max']):.2f} deg/s)")
    add(f"    │     Std yaw_rate:    {sim_vel['yaw_rate_std']:.4f} rad/s")
    add(f"    └── Real Robot:")
    add(f"          Mean |yaw_rate|: {real_vel['yaw_rate_abs_mean']:.4f} rad/s ({np.degrees(real_vel['yaw_rate_abs_mean']):.2f} deg/s)")
    add(f"          Max |yaw_rate|:  {real_vel['yaw_rate_max']:.4f} rad/s ({np.degrees(real_vel['yaw_rate_max']):.2f} deg/s)")
    add(f"          Std yaw_rate:    {real_vel['yaw_rate_std']:.4f} rad/s")
    add()
    
    add("  Total Rotation (Cumulative Heading Change):")
    add(f"    ├── Simulation: {sim_orient['total_rotation_deg']:.1f}°")
    add(f"    └── Real Robot: {real_orient['total_rotation_deg']:.1f}°")
    add(f"    Ratio: {rotation_ratio:.2f}x")
    add()
    
    add("  Forward Velocity Statistics:")
    add(f"    ├── Simulation:")
    add(f"    │     Mean: {sim_vel['forward_vel_mean']:.4f} m/s")
    add(f"    │     Max:  {sim_vel['forward_vel_max']:.4f} m/s")
    add(f"    └── Real Robot:")
    add(f"          Mean: {real_vel['forward_vel_mean']:.4f} m/s")
    add(f"          Max:  {real_vel['forward_vel_max']:.4f} m/s")
    add(f"    Ratio: {forward_ratio:.2f}x")
    add()
    
    # Actuation Analysis
    add("-" * 80)
    add("ACTUATION ANALYSIS (Joint Tracking)")
    add("-" * 80)
    add()
    
    joint_names = sim_data.get('joint_names', [f'Joint {i}' for i in range(sim_data['dof_pos'].shape[1])])
    if isinstance(joint_names, np.ndarray):
        joint_names = joint_names.tolist()
    num_joints = len(joint_names)
    
    add(f"  {'Joint':<15} {'Sim RMSE':>12} {'Real RMSE':>12} {'Ratio':>10} {'Sim Delay':>12} {'Real Delay':>12}")
    add(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*10} {'-'*12} {'-'*12}")
    
    for j in range(num_joints):
        name = joint_names[j]
        sim_rmse = sim_actuation['rmse'][j]
        real_rmse = real_actuation['rmse'][j]
        ratio = real_rmse / sim_rmse if sim_rmse > 0 else float('inf')
        sim_delay = sim_actuation['delay_time_ms'][j]
        real_delay = real_actuation['delay_time_ms'][j]
        
        add(f"  {name:<15} {sim_rmse:>12.5f} {real_rmse:>12.5f} {ratio:>9.2f}x {sim_delay:>11.1f}ms {real_delay:>11.1f}ms")
    
    add()
    avg_sim_rmse = np.mean(sim_actuation['rmse'])
    avg_real_rmse = np.mean(real_actuation['rmse'])
    actuation_ratio = avg_real_rmse / avg_sim_rmse if avg_sim_rmse > 0 else float('inf')
    add(f"  Average RMSE - Sim: {avg_sim_rmse:.5f} rad, Real: {avg_real_rmse:.5f} rad")
    add(f"  Actuation Gap Ratio: {actuation_ratio:.2f}x")
    add()
    
    # Stability Analysis
    add("-" * 80)
    add("STABILITY ANALYSIS")
    add("-" * 80)
    add()
    
    add("  Roll/Pitch Variation (body stability):")
    add(f"    ├── Simulation:")
    add(f"    │     Roll std:  {np.degrees(sim_orient['roll_std']):.2f}°")
    add(f"    │     Pitch std: {np.degrees(sim_orient['pitch_std']):.2f}°")
    add(f"    └── Real Robot:")
    add(f"          Roll std:  {np.degrees(real_orient['roll_std']):.2f}°")
    add(f"          Pitch std: {np.degrees(real_orient['pitch_std']):.2f}°")
    add()
    
    # Diagnosis
    add("=" * 80)
    add("DIAGNOSIS & ROOT CAUSE ANALYSIS")
    add("=" * 80)
    add()
    
    issues_found = []
    
    # Check turning rate discrepancy
    if abs(turning_ratio - 1.0) > 0.3:
        if turning_ratio > 1.0:
            issues_found.append(f"⚠️  MAJOR: Simulation turns {turning_ratio:.1f}x FASTER than real robot")
            add(f"  ⚠️  MAJOR ISSUE: Turning Rate Mismatch")
            add(f"      Simulation yaw rate is {turning_ratio:.1f}x higher than real robot.")
            add()
            add("      Possible causes:")
            add("      1. Ground friction coefficient too low in simulation")
            add("      2. Robot mass/inertia underestimated in sim URDF/MJCF")
            add("      3. Motor torque limits not modeled in simulation")
            add("      4. Foot-ground contact model too ideal (no slip resistance)")
            add("      5. Real robot may have mechanical friction in joints")
            add()
        else:
            issues_found.append(f"⚠️  Simulation turns {1/turning_ratio:.1f}x SLOWER than real robot")
            add(f"  ⚠️  ISSUE: Turning Rate Mismatch")
            add(f"      Real robot yaw rate is {1/turning_ratio:.1f}x higher than simulation.")
            add()
    
    # Check actuation gap
    if actuation_ratio > 2.0:
        issues_found.append(f"⚠️  Real robot tracking error is {actuation_ratio:.1f}x worse than sim")
        add(f"  ⚠️  ISSUE: Significant Actuation Gap")
        add(f"      Real robot joint tracking is {actuation_ratio:.1f}x worse than simulation.")
        add()
        add("      Possible causes:")
        add("      1. PD gains in simulation are too high (unrealistic tracking)")
        add("      2. Communication latency not modeled in simulation")
        add("      3. Motor bandwidth limitations on real robot")
        add()
    
    # Check delay difference
    avg_sim_delay = np.mean(sim_actuation['delay_time_ms'])
    avg_real_delay = np.mean(real_actuation['delay_time_ms'])
    if abs(avg_real_delay - avg_sim_delay) > 10:
        issues_found.append(f"⚠️  Delay mismatch: Sim={avg_sim_delay:.1f}ms, Real={avg_real_delay:.1f}ms")
        add(f"  ⚠️  ISSUE: Latency Mismatch")
        add(f"      Sim delay: {avg_sim_delay:.1f}ms, Real delay: {avg_real_delay:.1f}ms")
        add()
    
    if not issues_found:
        add("  ✓ No major issues detected!")
    
    add()
    
    # Recommendations
    add("=" * 80)
    add("RECOMMENDATIONS")
    add("=" * 80)
    add()
    
    if turning_ratio > 1.5:
        add("  🔧 TO REDUCE TURNING RATE IN SIMULATION:")
        add()
        add("     1. Increase ground friction coefficient:")
        add("        - In MJCF: <geom friction=\"1.0 0.005 0.0001\"/>")
        add("        - Try friction values: 0.8 → 1.0 → 1.5")
        add()
        add("     2. Increase robot inertia (if underestimated):")
        add("        - Check URDF/MJCF inertia tensors match real robot")
        add("        - Increase overall mass by ~10-20%")
        add()
        add("     3. Add motor dynamics/limits:")
        add("        - Limit max torque: forcerange=\"-X X\"")
        add("        - Add motor bandwidth limiting (filter)")
        add()
        add("     4. Add turning resistance:")
        add("        - Increase ground damping")
        add("        - Add rotational friction")
        add()
    elif turning_ratio < 0.7:
        add("  🔧 TO INCREASE TURNING RATE IN SIMULATION:")
        add()
        add("     1. Decrease ground friction coefficient")
        add("     2. Decrease robot mass/inertia if overestimated")
        add("     3. Check if contact model is too stiff")
        add()
    
    if actuation_ratio > 1.5:
        add("  🔧 TO MATCH ACTUATION PERFORMANCE:")
        add()
        add("     1. Add actuation noise/delay in simulation:")
        add("        - action_noise_std: 0.01-0.05")
        add("        - latency_scheme: 'constant' with delay_ms: 10-30")
        add()
        add("     2. Reduce PD gains in simulation:")
        add("        - Current gains may be unrealistically high")
        add("        - Try reducing kp by 20-50%")
        add()
    
    add()
    add("=" * 80)
    add("END OF REPORT")
    add("=" * 80)
    
    report = "\n".join(report_lines)
    
    # Print to console
    print(report)
    
    # Save to file
    if output_dir:
        report_path = Path(output_dir) / 'analysis_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\n  Report saved to: {report_path}")
    
    return report


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Sim-to-Real Comprehensive State Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/sim2real_state_analysis.py sim_state.npz real_state.npz
    python scripts/sim2real_state_analysis.py sim_state.npz real_state.npz --output results/
    python scripts/sim2real_state_analysis.py sim_state.npz real_state.npz --no-show
        """
    )
    
    parser.add_argument('sim_file', type=str, help='Simulation state file (.npz)')
    parser.add_argument('real_file', type=str, help='Real robot state file (.npz)')
    parser.add_argument('--output', '-o', type=str, default='sim2real_state_analysis',
                        help='Output directory for plots and report')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots (only save)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Sim-to-Real Comprehensive State Analysis")
    print("=" * 60)
    print(f"\nLoading data...")
    print(f"  Sim file:  {args.sim_file}")
    print(f"  Real file: {args.real_file}")
    
    # Load data
    sim_data = load_state_data(args.sim_file)
    real_data = load_state_data(args.real_file)
    
    print(f"\nData loaded successfully!")
    print(f"  Sim steps:  {len(sim_data['timestamps'])}")
    print(f"  Real steps: {len(real_data['timestamps'])}")
    
    # Compute metrics
    print(f"\nComputing metrics...")
    sim_actuation = compute_actuation_metrics(sim_data)
    real_actuation = compute_actuation_metrics(real_data)
    comparison = compute_behavioral_comparison(sim_data, real_data)
    
    # Generate plots
    print(f"\nGenerating plots...")
    
    if not args.no_show:
        plt.ion()  # Interactive mode
    
    plot_actuation_comparison(sim_data, real_data, output_dir)
    plot_velocity_comparison(sim_data, real_data, comparison, output_dir)
    plot_orientation_comparison(sim_data, real_data, comparison, output_dir)
    plot_yaw_rate_histogram(comparison, output_dir)
    plot_action_analysis(sim_data, real_data, output_dir)
    plot_frequency_analysis(sim_data, real_data, comparison, output_dir)
    
    # Generate report
    print(f"\n" + "-" * 60)
    generate_report(sim_data, real_data, comparison, sim_actuation, real_actuation, output_dir)
    
    print(f"\n✓ Analysis complete! Results saved to: {output_dir}/")
    
    if not args.no_show:
        plt.show(block=True)


if __name__ == '__main__':
    main()
