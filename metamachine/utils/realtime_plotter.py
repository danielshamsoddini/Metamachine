"""Real-time plotting utilities for joint tracking analysis.

This module provides utilities for real-time visualization of:
- Actions (policy outputs)
- Joint position commands (action + default offset)
- Actual joint positions (from simulation or real robot)

Useful for reality gap analysis to compare commanded vs actual joint positions.

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

import threading
import time
from collections import deque
from typing import Optional

import numpy as np


class RealtimeJointPlotter:
    """Real-time plotter for joint tracking analysis.
    
    Plots actions, joint commands (action + offset), and actual joint positions
    in real-time for analyzing tracking performance and reality gap.
    
    Usage:
        plotter = RealtimeJointPlotter(num_joints=3, joint_names=["j0", "j1", "j2"])
        plotter.start()
        
        # In your control loop:
        plotter.update(
            actions=policy_action,
            joint_commands=action + default_offset,
            joint_positions=actual_joint_pos
        )
        
        # When done:
        plotter.stop()
    """
    
    def __init__(
        self,
        num_joints: int,
        joint_names: Optional[list[str]] = None,
        default_dof_pos: Optional[np.ndarray] = None,
        history_length: int = 200,
        update_interval: float = 0.05,
        figsize: tuple[int, int] = (14, 8),
        title: str = "Joint Tracking Analysis",
    ):
        """Initialize the real-time plotter.
        
        Args:
            num_joints: Number of joints to plot
            joint_names: Names for each joint (optional)
            default_dof_pos: Default joint positions (offset added to actions)
            history_length: Number of timesteps to show in the plot
            update_interval: Minimum time between plot updates (seconds)
            figsize: Figure size (width, height)
            title: Plot title
        """
        self.num_joints = num_joints
        self.joint_names = joint_names or [f"Joint {i}" for i in range(num_joints)]
        self.default_dof_pos = default_dof_pos if default_dof_pos is not None else np.zeros(num_joints)
        self.history_length = history_length
        self.update_interval = update_interval
        self.figsize = figsize
        self.title = title
        
        # Data buffers (thread-safe with deque)
        self.time_history = deque(maxlen=history_length)
        self.action_history = deque(maxlen=history_length)
        self.command_history = deque(maxlen=history_length)
        self.position_history = deque(maxlen=history_length)
        
        # Threading control
        self._running = False
        self._plot_thread: Optional[threading.Thread] = None
        self._data_lock = threading.Lock()
        self._last_update_time = 0.0
        self._start_time = 0.0
        self._step_count = 0
        
        # Plot objects (initialized in _plot_loop)
        self._fig = None
        self._axes = None
        self._lines = None
        
    def start(self) -> None:
        """Start the real-time plotting thread."""
        if self._running:
            return
            
        self._running = True
        self._start_time = time.time()
        self._step_count = 0
        
        # Clear history
        self.time_history.clear()
        self.action_history.clear()
        self.command_history.clear()
        self.position_history.clear()
        
        # Start plot thread
        self._plot_thread = threading.Thread(target=self._plot_loop, daemon=True)
        self._plot_thread.start()
        print(f"[RealtimeJointPlotter] Started with {self.num_joints} joints")
        
    def stop(self) -> None:
        """Stop the real-time plotting thread."""
        self._running = False
        if self._plot_thread is not None:
            self._plot_thread.join(timeout=1.0)
            self._plot_thread = None
        print("[RealtimeJointPlotter] Stopped")
        
    def update(
        self,
        actions: np.ndarray,
        joint_positions: np.ndarray,
        joint_commands: Optional[np.ndarray] = None,
    ) -> None:
        """Update the plot with new data.
        
        Args:
            actions: Raw actions from policy (before adding offset)
            joint_positions: Actual joint positions from robot/simulation
            joint_commands: Commanded joint positions (action + offset).
                           If None, computed as actions + default_dof_pos
        """
        if not self._running:
            return
            
        # Compute joint commands if not provided
        if joint_commands is None:
            joint_commands = actions + self.default_dof_pos
            
        self._step_count += 1
        current_time = time.time() - self._start_time
        
        with self._data_lock:
            self.time_history.append(current_time)
            self.action_history.append(actions.copy())
            self.command_history.append(joint_commands.copy())
            self.position_history.append(joint_positions.copy())
            
    def _plot_loop(self) -> None:
        """Main plotting loop (runs in separate thread)."""
        import matplotlib
        
        # Try to use TkAgg for interactive plots, fall back to other backends
        backend = None
        for candidate_backend in ['TkAgg', 'Qt5Agg', 'GTK3Agg', 'WXAgg']:
            try:
                matplotlib.use(candidate_backend)
                backend = candidate_backend
                break
            except Exception:
                continue
        
        if backend is None:
            print("[RealtimeJointPlotter] Warning: No interactive backend available. "
                  "Real-time plotting disabled.")
            self._running = False
            return
            
        import matplotlib.pyplot as plt
        
        print(f"[RealtimeJointPlotter] Using matplotlib backend: {backend}")
        
        # Set up the figure
        plt.ion()  # Enable interactive mode
        
        # Use single column layout (N rows x 1 col) for longer, easier-to-read subplots
        nrows = self.num_joints
        ncols = 1
        
        # Adjust figure size for vertical layout (width, height per joint)
        fig_width = self.figsize[0]
        fig_height = max(self.figsize[1], 2.5 * self.num_joints)  # At least 2.5 inches per joint
        
        self._fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
        self._fig.suptitle(self.title, fontsize=14)
        self._axes = axes.flatten()
        
        # Initialize lines for each joint
        self._lines = []
        for i in range(self.num_joints):
            ax = self._axes[i]
            ax.set_title(self.joint_names[i], fontsize=10)
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('Position (rad)', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Create lines for: action, command, actual position
            line_action, = ax.plot([], [], 'b--', label='Action', alpha=0.6, linewidth=1)
            line_command, = ax.plot([], [], 'g-', label='Command', linewidth=2)
            line_position, = ax.plot([], [], 'r-', label='Actual', linewidth=2)
            
            self._lines.append({
                'action': line_action,
                'command': line_command,
                'position': line_position,
            })
            
            ax.legend(loc='upper right', fontsize=7)
            
        # Hide unused subplots
        for i in range(self.num_joints, len(self._axes)):
            self._axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.show(block=False)
        
        # Main update loop
        while self._running:
            try:
                self._update_plot()
                plt.pause(self.update_interval)
            except Exception as e:
                print(f"[RealtimeJointPlotter] Error: {e}")
                break
                
        plt.ioff()
        plt.close(self._fig)
        
    def _update_plot(self) -> None:
        """Update the plot with current data."""
        with self._data_lock:
            if len(self.time_history) < 2:
                return
                
            times = np.array(self.time_history)
            actions = np.array(self.action_history)
            commands = np.array(self.command_history)
            positions = np.array(self.position_history)
            
        # Update each joint's lines
        for i in range(self.num_joints):
            ax = self._axes[i]
            lines = self._lines[i]
            
            lines['action'].set_data(times, actions[:, i])
            lines['command'].set_data(times, commands[:, i])
            lines['position'].set_data(times, positions[:, i])
            
            # Auto-scale axes
            ax.relim()
            ax.autoscale_view()
            
            # Keep consistent x-axis range showing recent history
            if len(times) > 10:
                x_min = max(0, times[-1] - self.history_length * 0.05)
                x_max = times[-1] + 0.5
                ax.set_xlim(x_min, x_max)
                
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()


class JointTrackingLogger:
    """Logger for recording joint tracking data for offline analysis.
    
    Records actions, commands, and actual positions for post-hoc analysis
    of tracking performance and reality gap.
    """
    
    def __init__(
        self,
        num_joints: int,
        joint_names: Optional[list[str]] = None,
        default_dof_pos: Optional[np.ndarray] = None,
    ):
        """Initialize the logger.
        
        Args:
            num_joints: Number of joints
            joint_names: Names for each joint
            default_dof_pos: Default joint positions
        """
        self.num_joints = num_joints
        self.joint_names = joint_names or [f"Joint {i}" for i in range(num_joints)]
        self.default_dof_pos = default_dof_pos if default_dof_pos is not None else np.zeros(num_joints)
        
        self._data = {
            'timestamps': [],
            'actions': [],
            'commands': [],
            'positions': [],
        }
        self._start_time = None
        
    def reset(self) -> None:
        """Reset the logger."""
        self._data = {
            'timestamps': [],
            'actions': [],
            'commands': [],
            'positions': [],
        }
        self._start_time = None
        
    def log(
        self,
        actions: np.ndarray,
        joint_positions: np.ndarray,
        joint_commands: Optional[np.ndarray] = None,
    ) -> None:
        """Log a single timestep.
        
        Args:
            actions: Raw actions from policy
            joint_positions: Actual joint positions
            joint_commands: Commanded positions (computed if None)
        """
        if self._start_time is None:
            self._start_time = time.time()
            
        if joint_commands is None:
            joint_commands = actions + self.default_dof_pos
            
        timestamp = time.time() - self._start_time
        self._data['timestamps'].append(timestamp)
        self._data['actions'].append(actions.copy())
        self._data['commands'].append(joint_commands.copy())
        self._data['positions'].append(joint_positions.copy())
        
    def get_data(self) -> dict:
        """Get logged data as numpy arrays.
        
        Returns:
            Dictionary with 'timestamps', 'actions', 'commands', 'positions'
        """
        return {
            'timestamps': np.array(self._data['timestamps']),
            'actions': np.array(self._data['actions']),
            'commands': np.array(self._data['commands']),
            'positions': np.array(self._data['positions']),
            'joint_names': self.joint_names,
            'default_dof_pos': self.default_dof_pos,
        }
        
    def save(self, filepath: str) -> None:
        """Save logged data to file.
        
        Args:
            filepath: Path to save data (supports .npz, .pkl)
        """
        import os
        data = self.get_data()
        
        if filepath.endswith('.npz'):
            np.savez(filepath, **data)
        elif filepath.endswith('.pkl'):
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        else:
            # Default to npz
            np.savez(filepath + '.npz', **data)
            
        print(f"[JointTrackingLogger] Saved {len(self._data['timestamps'])} timesteps to {filepath}")
        
    def plot_summary(
        self,
        figsize: tuple[int, int] = (14, 10),
        save_path: Optional[str] = None,
    ) -> None:
        """Generate summary plots of the recorded data.
        
        Args:
            figsize: Figure size (width, height) - height will be auto-adjusted
            save_path: Path to save figure (optional)
        """
        import matplotlib.pyplot as plt
        
        data = self.get_data()
        times = data['timestamps']
        actions = data['actions']
        commands = data['commands']
        positions = data['positions']
        
        if len(times) == 0:
            print("[JointTrackingLogger] No data to plot")
            return
            
        # Calculate tracking error
        tracking_error = positions - commands
        
        # Use vertical layout: 2 rows per joint (position + error), 1 column
        nrows = self.num_joints * 2  # 2 subplots per joint (position + error)
        ncols = 1
        
        # Adjust figure size for vertical layout
        fig_width = figsize[0]
        fig_height = max(figsize[1], 2.5 * self.num_joints)  # At least 2.5 inches per joint pair
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
        fig.suptitle('Joint Tracking Analysis Summary', fontsize=14)
        
        for i in range(self.num_joints):
            # Position tracking plot (row 2*i)
            ax_pos = axes[i * 2, 0]
            ax_pos.plot(times, actions[:, i], 'b--', label='Action', alpha=0.6)
            ax_pos.plot(times, commands[:, i], 'g-', label='Command', linewidth=1.5)
            ax_pos.plot(times, positions[:, i], 'r-', label='Actual', linewidth=1.5)
            ax_pos.set_title(f'{self.joint_names[i]} - Position', fontsize=10)
            ax_pos.set_xlabel('Time (s)', fontsize=8)
            ax_pos.set_ylabel('Position (rad)', fontsize=8)
            ax_pos.legend(loc='upper right', fontsize=7)
            ax_pos.grid(True, alpha=0.3)
            
            # Tracking error plot (row 2*i + 1)
            ax_err = axes[i * 2 + 1, 0]
            ax_err.plot(times, tracking_error[:, i], 'k-', linewidth=1)
            ax_err.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax_err.fill_between(times, tracking_error[:, i], 0, alpha=0.3)
            ax_err.set_title(f'{self.joint_names[i]} - Tracking Error', fontsize=10)
            ax_err.set_xlabel('Time (s)', fontsize=8)
            ax_err.set_ylabel('Error (rad)', fontsize=8)
            ax_err.grid(True, alpha=0.3)
            
            # Add error statistics
            rmse = np.sqrt(np.mean(tracking_error[:, i] ** 2))
            max_err = np.max(np.abs(tracking_error[:, i]))
            ax_err.text(
                0.02, 0.98,
                f'RMSE: {rmse:.4f}\nMax: {max_err:.4f}',
                transform=ax_err.transAxes,
                fontsize=7,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
                    
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[JointTrackingLogger] Saved figure to {save_path}")
            
        plt.show()


def create_joint_plotter_from_env(
    env,
    history_length: int = 200,
    update_interval: float = 0.05,
) -> RealtimeJointPlotter:
    """Create a RealtimeJointPlotter from an environment.
    
    Args:
        env: MetaMachine environment (sim or real)
        history_length: Number of timesteps to display
        update_interval: Plot update interval
        
    Returns:
        Configured RealtimeJointPlotter instance
    """
    # Get num_joints and default_dof_pos from environment
    if hasattr(env, 'num_joint'):
        num_joints = env.num_joint
    elif hasattr(env, 'action_space'):
        num_joints = env.action_space.shape[0]
    else:
        raise ValueError("Cannot determine number of joints from environment")
        
    # Get default DOF positions
    if hasattr(env, 'default_dof_pos'):
        default_dof_pos = np.array(env.default_dof_pos)
    elif hasattr(env, 'cfg') and hasattr(env.cfg, 'control'):
        default_dof_pos = np.array(env.cfg.control.get('default_dof_pos', np.zeros(num_joints)))
    else:
        default_dof_pos = np.zeros(num_joints)
        
    # Get joint names if available
    joint_names = None
    if hasattr(env, 'joint_names'):
        joint_names = env.joint_names
        
    return RealtimeJointPlotter(
        num_joints=num_joints,
        joint_names=joint_names,
        default_dof_pos=default_dof_pos,
        history_length=history_length,
        update_interval=update_interval,
    )


def create_joint_logger_from_env(env) -> JointTrackingLogger:
    """Create a JointTrackingLogger from an environment.
    
    Args:
        env: MetaMachine environment (sim or real)
        
    Returns:
        Configured JointTrackingLogger instance
    """
    # Get num_joints and default_dof_pos from environment
    if hasattr(env, 'num_joint'):
        num_joints = env.num_joint
    elif hasattr(env, 'action_space'):
        num_joints = env.action_space.shape[0]
    else:
        raise ValueError("Cannot determine number of joints from environment")
        
    # Get default DOF positions
    if hasattr(env, 'default_dof_pos'):
        default_dof_pos = np.array(env.default_dof_pos)
    elif hasattr(env, 'cfg') and hasattr(env.cfg, 'control'):
        default_dof_pos = np.array(env.cfg.control.get('default_dof_pos', np.zeros(num_joints)))
    else:
        default_dof_pos = np.zeros(num_joints)
        
    # Get joint names if available
    joint_names = None
    if hasattr(env, 'joint_names'):
        joint_names = env.joint_names
        
    return JointTrackingLogger(
        num_joints=num_joints,
        joint_names=joint_names,
        default_dof_pos=default_dof_pos,
    )


class StateLogger:
    """Comprehensive state logger for sim-to-real behavior analysis.
    
    Records all relevant state data for comparing simulation and real robot
    behaviors beyond just joint tracking. Useful for diagnosing why policies
    behave differently in sim vs real.
    
    Recorded data includes:
    - Joint positions and velocities (dof_pos, dof_vel)
    - Actions and joint commands
    - Body orientation (quaternion, projected gravity)
    - Body velocities (linear and angular)
    - IMU data if available
    - Rewards and episode info
    """
    
    def __init__(
        self,
        num_joints: int,
        joint_names: Optional[list[str]] = None,
        default_dof_pos: Optional[np.ndarray] = None,
    ):
        """Initialize the state logger.
        
        Args:
            num_joints: Number of joints
            joint_names: Names for each joint
            default_dof_pos: Default joint positions
        """
        self.num_joints = num_joints
        self.joint_names = joint_names or [f"Joint {i}" for i in range(num_joints)]
        self.default_dof_pos = default_dof_pos if default_dof_pos is not None else np.zeros(num_joints)
        
        self._data = {
            # Timestamps
            'timestamps': [],
            'step_count': [],
            'episode': [],
            
            # Actions
            'actions': [],
            'joint_commands': [],
            
            # Joint state
            'dof_pos': [],
            'dof_vel': [],
            
            # Body state
            'quat': [],               # Orientation quaternion [x, y, z, w]
            'ang_vel_body': [],       # Angular velocity in body frame
            'vel_body': [],           # Linear velocity in body frame
            'projected_gravity': [],  # Gravity projected to body frame
            
            # Position (sim only, but useful for analysis)
            'pos_world': [],
            'vel_world': [],
            'ang_vel_world': [],
            
            # Rewards
            'rewards': [],
            'reward_components': [],
            
            # Additional data
            'observations': [],
            'info': [],
        }
        self._start_time = None
        self._step_count = 0
        self._episode = 0
        
    def reset(self, new_episode: bool = True) -> None:
        """Reset the logger, optionally starting a new episode.
        
        Args:
            new_episode: If True, increment episode counter
        """
        if new_episode:
            self._episode += 1
        self._step_count = 0
        
    def clear(self) -> None:
        """Clear all logged data."""
        for key in self._data:
            self._data[key] = []
        self._start_time = None
        self._step_count = 0
        self._episode = 0
        
    def log(
        self,
        action: np.ndarray,
        obs: np.ndarray,
        reward: float,
        info: dict,
        env,
        joint_command: Optional[np.ndarray] = None,
    ) -> None:
        """Log a single timestep of state data.
        
        Args:
            action: Action taken
            obs: Observation received
            reward: Reward received
            info: Info dict from environment
            env: Environment instance (to extract additional state)
            joint_command: Joint command (computed if None)
        """
        if self._start_time is None:
            self._start_time = time.time()
            
        timestamp = time.time() - self._start_time
        self._step_count += 1
        
        # Compute joint command if not provided
        if joint_command is None:
            joint_command = action + self.default_dof_pos
        
        # Basic data
        self._data['timestamps'].append(timestamp)
        self._data['step_count'].append(self._step_count)
        self._data['episode'].append(self._episode)
        self._data['actions'].append(action.copy())
        self._data['joint_commands'].append(joint_command.copy())
        self._data['observations'].append(obs.copy())
        self._data['rewards'].append(reward)
        
        # Extract state from environment
        state_data = self._extract_state_from_env(env)
        for key, value in state_data.items():
            if key in self._data:
                self._data[key].append(value)
        
        # Reward components from info
        if 'reward_components' in info:
            self._data['reward_components'].append(info['reward_components'].copy())
        else:
            self._data['reward_components'].append({})
            
        # Store simplified info
        self._data['info'].append({
            k: v for k, v in info.items() 
            if isinstance(v, (int, float, bool, str)) or 
            (isinstance(v, np.ndarray) and v.size < 20)
        })
    
    def _extract_state_from_env(self, env) -> dict:
        """Extract state data from environment.
        
        Args:
            env: Environment instance
            
        Returns:
            Dictionary of state values
        """
        state_data = {}
        
        # Try to get joint state
        if hasattr(env, 'state') and hasattr(env.state, 'dof_pos'):
            state_data['dof_pos'] = np.array(env.state.dof_pos).copy()
            if hasattr(env.state, 'dof_vel'):
                state_data['dof_vel'] = np.array(env.state.dof_vel).copy()
        elif hasattr(env, 'observable_data'):
            if 'dof_pos' in env.observable_data:
                state_data['dof_pos'] = np.array(env.observable_data['dof_pos']).copy()
            if 'dof_vel' in env.observable_data:
                state_data['dof_vel'] = np.array(env.observable_data['dof_vel']).copy()
        elif hasattr(env, 'data') and hasattr(env, 'joint_idx'):
            # MuJoCo simulation
            state_data['dof_pos'] = env.data.qpos[env.model.jnt_qposadr[env.joint_idx]].copy()
            state_data['dof_vel'] = env.data.qvel[env.model.jnt_dofadr[env.joint_idx]].copy()
        
        # Try to get orientation
        if hasattr(env, 'state') and hasattr(env.state, 'quat'):
            state_data['quat'] = np.array(env.state.quat).copy()
        elif hasattr(env, 'observable_data') and 'quat' in env.observable_data:
            state_data['quat'] = np.array(env.observable_data['quat']).copy()
        elif hasattr(env, 'data'):
            # MuJoCo - convert wxyz to xyzw
            quat_wxyz = env.data.qpos[3:7]
            state_data['quat'] = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        
        # Try to get angular velocity
        if hasattr(env, 'state') and hasattr(env.state, 'ang_vel_body'):
            state_data['ang_vel_body'] = np.array(env.state.ang_vel_body).copy()
        elif hasattr(env, 'observable_data') and 'ang_vel_body' in env.observable_data:
            state_data['ang_vel_body'] = np.array(env.observable_data['ang_vel_body']).copy()
        elif hasattr(env, 'data'):
            state_data['ang_vel_body'] = env.data.qvel[3:6].copy()
        
        # Try to get linear velocity
        if hasattr(env, 'state') and hasattr(env.state, 'vel_body'):
            state_data['vel_body'] = np.array(env.state.vel_body).copy()
        elif hasattr(env, 'observable_data') and 'vel_body' in env.observable_data:
            state_data['vel_body'] = np.array(env.observable_data['vel_body']).copy()
        
        # Try to get projected gravity
        if hasattr(env, 'state') and hasattr(env.state, 'projected_gravity'):
            state_data['projected_gravity'] = np.array(env.state.projected_gravity).copy()
        elif hasattr(env, 'state') and hasattr(env.state, 'derived') and hasattr(env.state.derived, 'projected_gravity'):
            state_data['projected_gravity'] = np.array(env.state.derived.projected_gravity).copy()
        
        # Simulation-specific: world frame data
        if hasattr(env, 'data'):
            state_data['pos_world'] = env.data.qpos[:3].copy()
            state_data['vel_world'] = env.data.qvel[:3].copy()
            state_data['ang_vel_world'] = env.data.qvel[3:6].copy()
        
        return state_data
    
    def get_data(self) -> dict:
        """Get logged data as numpy arrays.
        
        Returns:
            Dictionary with all logged data as numpy arrays
        """
        data = {}
        for key, values in self._data.items():
            if len(values) == 0:
                continue
            if key in ['reward_components', 'info']:
                # Keep as list for dict data
                data[key] = values
            elif isinstance(values[0], np.ndarray):
                data[key] = np.array(values)
            elif isinstance(values[0], (int, float)):
                data[key] = np.array(values)
            else:
                data[key] = values
                
        # Add metadata
        data['joint_names'] = self.joint_names
        data['default_dof_pos'] = self.default_dof_pos
        data['num_joints'] = self.num_joints
        
        return data
    
    def save(self, filepath: str) -> None:
        """Save logged data to file.
        
        Args:
            filepath: Path to save data (.npz or .pkl)
        """
        data = self.get_data()
        
        if filepath.endswith('.pkl'):
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        else:
            # For npz, we need to handle dict data specially
            if not filepath.endswith('.npz'):
                filepath = filepath + '.npz'
            
            # Convert reward_components and info to pickle-able format
            import pickle
            save_data = {}
            for key, value in data.items():
                if key in ['reward_components', 'info']:
                    save_data[key] = np.array([pickle.dumps(v) for v in value], dtype=object)
                else:
                    save_data[key] = value
            
            np.savez(filepath, **save_data)
            
        num_steps = len(self._data['timestamps'])
        print(f"[StateLogger] Saved {num_steps} timesteps to {filepath}")
        
    @staticmethod
    def load(filepath: str) -> dict:
        """Load state data from file.
        
        Args:
            filepath: Path to load data from
            
        Returns:
            Dictionary of logged data
        """
        if filepath.endswith('.pkl'):
            import pickle
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            import pickle
            data = dict(np.load(filepath, allow_pickle=True))
            
            # Decode pickled dict data
            for key in ['reward_components', 'info']:
                if key in data and len(data[key]) > 0:
                    try:
                        data[key] = [pickle.loads(v) for v in data[key]]
                    except:
                        pass  # Keep as is if unpickling fails
                        
            # Handle numpy object arrays
            for key in ['joint_names', 'default_dof_pos']:
                if key in data and isinstance(data[key], np.ndarray):
                    if data[key].ndim == 0:
                        data[key] = data[key].item()
                        
            return data


def create_state_logger_from_env(env) -> StateLogger:
    """Create a StateLogger from an environment.
    
    Args:
        env: MetaMachine environment (sim or real)
        
    Returns:
        Configured StateLogger instance
    """
    # Get num_joints and default_dof_pos from environment
    if hasattr(env, 'num_joint'):
        num_joints = env.num_joint
    elif hasattr(env, 'action_space'):
        num_joints = env.action_space.shape[0]
    else:
        raise ValueError("Cannot determine number of joints from environment")
        
    # Get default DOF positions
    if hasattr(env, 'default_dof_pos'):
        default_dof_pos = np.array(env.default_dof_pos)
    elif hasattr(env, 'action_processor') and hasattr(env.action_processor, 'default_dof_pos'):
        default_dof_pos = np.array(env.action_processor.default_dof_pos)
    elif hasattr(env, 'cfg') and hasattr(env.cfg, 'control'):
        default = env.cfg.control.get('default_dof_pos', 0)
        if isinstance(default, (list, np.ndarray)):
            default_dof_pos = np.array(default)
        else:
            default_dof_pos = np.full(num_joints, default)
    else:
        default_dof_pos = np.zeros(num_joints)
        
    # Get joint names if available
    joint_names = None
    if hasattr(env, 'joint_names'):
        joint_names = env.joint_names
        
    return StateLogger(
        num_joints=num_joints,
        joint_names=joint_names,
        default_dof_pos=default_dof_pos,
    )
