"""
Bearing Estimation Utilities for MetaMachine

This module provides reusable components for bearing estimation using
distance triangulation with multiple locomotion policies.

Features:
- MultiPolicyBearingCollector: Collects bearing data with random policy switching
- BearingEstimatorNetwork: Neural network with optional policy indicator input
- BearingEstimatorRunner: Inference-time bearing estimation

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn


__all__ = [
    "MultiPolicyBearingCollector",
    "BearingEstimatorV3",
    "BearingEstimatorRunner",
    "BearingAugmentedPolicySwitchEnv",
    "BearingAugmentedConfig",
]


# =============================================================================
# Multi-Policy Data Collector
# =============================================================================

class MultiPolicyBearingCollector:
    """
    Collects simulation data for bearing estimator training with multiple policies.
    
    The robot randomly switches between policies at random intervals while
    collecting distance and bearing data. A one-hot policy indicator is saved
    alongside each sample.
    
    Output format:
    - distance_history: (N, history_length)
    - delta_history: (N, history_length) 
    - bearing: (N,) ground truth bearing in radians
    - policy_index: (N,) integer policy index for each sample
    - policy_onehot: (N, num_policies) one-hot encoding of active policy
    - locomotion_obs_history: (N, history_length, obs_dim) [optional]
    
    Sign convention:
    - Positive bearing = target to the LEFT
    - Negative bearing = target to the RIGHT
    - 0 = target directly in front
    """
    
    def __init__(
        self,
        locomotion_log_dirs: List[str],
        checkpoint: str = "latest",
        history_length: int = 30,
        target_spawn_range: Tuple[float, float] = (0.5, 3.5),
        distance_noise_std: float = 0.03,
        max_steps_per_episode: int = 1000,
        policy_switch_interval: Tuple[int, int] = (100, 500),
        device: str = "auto",
        include_locomotion_obs: bool = False,
        action_scale: float = 1.0,
        policy_names: Optional[List[str]] = None,
    ):
        """
        Initialize multi-policy data collector.
        
        Args:
            locomotion_log_dirs: List of paths to locomotion policy log directories
            checkpoint: Checkpoint to load from each directory
            history_length: History length for bearing estimator
            target_spawn_range: (min, max) distance for target spawn
            distance_noise_std: Std dev of distance sensor noise
            max_steps_per_episode: Max steps per episode
            policy_switch_interval: (min, max) steps between policy switches
            device: Device for policy inference
            include_locomotion_obs: Whether to collect locomotion observations
            action_scale: Scale factor for policy actions
            policy_names: Optional names for each policy (default: directory names)
        """
        self.locomotion_log_dirs = locomotion_log_dirs
        self.num_policies = len(locomotion_log_dirs)
        self.checkpoint = checkpoint
        self.history_length = history_length
        self.target_spawn_range = target_spawn_range
        self.distance_noise_std = distance_noise_std
        self.max_steps_per_episode = max_steps_per_episode
        self.policy_switch_interval = policy_switch_interval
        self.device = device
        self.include_locomotion_obs = include_locomotion_obs
        self.action_scale = action_scale
        
        # Set policy names
        if policy_names is not None:
            assert len(policy_names) == self.num_policies
            self.policy_names = policy_names
        else:
            self.policy_names = [Path(d).name for d in locomotion_log_dirs]
        
        # Will be initialized when collecting
        self.locomotion_env = None
        self.policy_runner = None
        self.single_obs_dim = None
        
        # All collected episodes
        self.all_episodes = []
        
    def _setup(self, render_mode: str = "none"):
        """Setup environment and load all policies.
        
        Args:
            render_mode: Render mode for environment ("none", "mp4", "human")
        """
        from metamachine.environments.configs.config_registry import ConfigRegistry
        from metamachine.utils.policy_runner import (
            PolicyRunner, 
            find_checkpoint_path, 
            load_policy_standalone
        )
        
        # Load config from first policy
        config_path = Path(self.locomotion_log_dirs[0]) / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        self.locomotion_cfg = ConfigRegistry.create_from_file(str(config_path))
        
        # Configure rendering
        if render_mode == "none":
            self.locomotion_cfg.simulation.render = False
            self.locomotion_cfg.simulation.render_mode = "none"
        else:
            self.locomotion_cfg.simulation.render = True
            self.locomotion_cfg.simulation.render_mode = render_mode
        
        # Create environment
        from metamachine.environments.env_sim import MetaMachine
        self.locomotion_env = MetaMachine(self.locomotion_cfg)
        
        # Calculate single-timestep observation dimension
        include_history_steps = self.locomotion_cfg.observation.include_history_steps
        full_obs_dim = self.locomotion_env.observation_space.shape[0]
        self.single_obs_dim = full_obs_dim // include_history_steps
        
        # Load all policies
        self.policy_runner = PolicyRunner()
        obs_dim = self.locomotion_env.observation_space.shape[0]
        act_dim = self.locomotion_env.action_space.shape[0]
        
        print(f"[MultiPolicyCollector] Loading {self.num_policies} policies...")
        for i, log_dir in enumerate(self.locomotion_log_dirs):
            checkpoint_path = find_checkpoint_path(Path(log_dir), self.checkpoint)
            if checkpoint_path is None:
                raise FileNotFoundError(f"No checkpoint found in {log_dir}")
            
            policy, _, _ = load_policy_standalone(
                checkpoint_path,
                obs_dim=obs_dim,
                act_dim=act_dim,
                device=self.device,
                verbose=False
            )
            
            self.policy_runner.add_policy(
                policy=policy,
                name=self.policy_names[i],
                log_dir=log_dir,
                obs_dim=obs_dim,
                act_dim=act_dim
            )
            print(f"  [{i}] {self.policy_names[i]}: {Path(checkpoint_path).name}")
        
        print(f"[MultiPolicyCollector] Setup complete")
        print(f"  Full observation dim: {obs_dim} (stacked)")
        print(f"  Single-timestep obs dim: {self.single_obs_dim}")
        print(f"  Action dim: {act_dim}")
        print(f"  Policy switch interval: {self.policy_switch_interval}")
        
    def _get_single_timestep_obs(self) -> np.ndarray:
        """Get single-timestep observation (without history stacking)."""
        if hasattr(self.locomotion_env, 'state'):
            single_obs = self.locomotion_env.state._construct_observation()
            return np.asarray(single_obs).flatten()
        return np.zeros(self.single_obs_dim)
    
    def _get_robot_position(self) -> np.ndarray:
        """Get robot's world position."""
        if hasattr(self.locomotion_env, 'state'):
            return self.locomotion_env.state.raw.pos_world.copy()
        return np.zeros(3)
    
    def _get_robot_heading(self) -> float:
        """Get robot's heading angle (forward is local Y axis)."""
        if hasattr(self.locomotion_env, 'state'):
            heading = self.locomotion_env.state.derived.heading
            if hasattr(heading, '__len__'):
                return float(heading[0])
            return float(heading)
        return 0.0
    
    def _get_distance_to_target(self, target_pos: np.ndarray) -> float:
        """Get distance from robot to target."""
        robot_pos = self._get_robot_position()
        return float(np.linalg.norm(target_pos - robot_pos[:2]))
    
    def _get_bearing_to_target(self, target_pos: np.ndarray) -> float:
        """
        Get relative bearing to target (in robot's local frame).
        
        Returns:
            Bearing in radians, range [-pi, pi]
            0 = target directly in front (local +Y direction)
            >0 = target to the LEFT
            <0 = target to the RIGHT
        """
        robot_pos = self._get_robot_position()
        heading = self._get_robot_heading()
        
        # Vector from robot to target in world frame
        to_target = target_pos - robot_pos[:2]
        
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        
        # Project to_target onto robot's local axes
        to_target_local_x = sin_h * to_target[0] - cos_h * to_target[1]  # LEFT
        to_target_local_y = cos_h * to_target[0] + sin_h * to_target[1]  # FORWARD
        
        bearing = np.arctan2(-to_target_local_x, to_target_local_y)
        return bearing
    
    def _spawn_target(self) -> np.ndarray:
        """Spawn target at random position around robot."""
        robot_pos = self._get_robot_position()
        
        distance = np.random.uniform(self.target_spawn_range[0], self.target_spawn_range[1])
        angle = np.random.uniform(0, 2 * np.pi)
        
        target_pos = np.array([
            robot_pos[0] + distance * np.cos(angle),
            robot_pos[1] + distance * np.sin(angle)
        ])
        
        return target_pos
    
    def collect_episode(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Collect one episode of data with random policy switching.
        
        Returns:
            Episode data dictionary, or None if not enough samples
        """
        # Reset environment
        obs, _ = self.locomotion_env.reset()
        
        # Spawn target
        target_pos = self._spawn_target()
        
        # Initialize current policy randomly
        current_policy_idx = np.random.randint(0, self.num_policies)
        self.policy_runner.select_policy(current_policy_idx)
        
        # Initialize steps until next switch
        steps_until_switch = np.random.randint(
            self.policy_switch_interval[0], 
            self.policy_switch_interval[1]
        )
        
        # Initialize buffers
        distances = []
        bearings = []
        policy_indices = []
        locomotion_obs = []
        
        # Run episode
        for step in range(self.max_steps_per_episode):
            # Check if we should switch policies
            if steps_until_switch <= 0:
                current_policy_idx = np.random.randint(0, self.num_policies)
                self.policy_runner.select_policy(current_policy_idx)
                steps_until_switch = np.random.randint(
                    self.policy_switch_interval[0], 
                    self.policy_switch_interval[1]
                )
            
            # Record data
            distance = self._get_distance_to_target(target_pos)
            bearing = self._get_bearing_to_target(target_pos)
            
            # Add noise to distance
            noisy_distance = distance + np.random.normal(0, self.distance_noise_std)
            noisy_distance = max(0.01, noisy_distance)
            
            distances.append(noisy_distance)
            bearings.append(bearing)
            policy_indices.append(current_policy_idx)
            
            # Collect single-timestep observation
            if self.include_locomotion_obs:
                single_obs = self._get_single_timestep_obs()
                locomotion_obs.append(single_obs.copy())
            
            # Get action from current policy
            action, _ = self.policy_runner.predict(obs, deterministic=True)
            action = np.asarray(action).flatten() * self.action_scale
            
            # Step environment
            obs, reward, terminated, truncated, info = self.locomotion_env.step(action)
            
            steps_until_switch -= 1
            
            if terminated:
                break
        
        # Check if we have enough samples
        if len(distances) < self.history_length + 1:
            print(f"[MultiPolicyCollector] Episode too short ({len(distances)} samples), skipping")
            return None
        
        # Convert to arrays
        distances = np.array(distances, dtype=np.float32)
        bearings = np.array(bearings, dtype=np.float32)
        policy_indices = np.array(policy_indices, dtype=np.int64)
        
        # Create sliding window samples
        num_samples = len(distances) - self.history_length
        
        distance_history = np.zeros((num_samples, self.history_length), dtype=np.float32)
        delta_history = np.zeros((num_samples, self.history_length), dtype=np.float32)
        target_bearings = np.zeros(num_samples, dtype=np.float32)
        sample_policy_indices = np.zeros(num_samples, dtype=np.int64)
        policy_onehot = np.zeros((num_samples, self.num_policies), dtype=np.float32)
        
        for i in range(num_samples):
            start_idx = i
            end_idx = i + self.history_length
            
            distance_history[i] = distances[start_idx:end_idx]
            
            # Compute delta distances
            deltas = np.diff(distances[start_idx:end_idx + 1])
            delta_history[i] = -deltas  # Negate: getting closer = positive
            
            # Target bearing at end of window
            target_bearings[i] = bearings[end_idx - 1]
            
            # Policy at end of window (current policy when predicting)
            sample_policy_indices[i] = policy_indices[end_idx - 1]
            policy_onehot[i, policy_indices[end_idx - 1]] = 1.0
        
        episode_data = {
            'distance_history': distance_history,
            'delta_history': delta_history,
            'bearing': target_bearings,
            'policy_index': sample_policy_indices,
            'policy_onehot': policy_onehot,
            'full_distances': distances,
            'full_bearings': bearings,
            'full_policy_indices': policy_indices,
            'target_position': target_pos,
        }
        
        # Include locomotion observations if enabled
        if self.include_locomotion_obs and locomotion_obs:
            loco_obs_array = np.array(locomotion_obs, dtype=np.float32)
            
            locomotion_obs_history = np.zeros(
                (num_samples, self.history_length, self.single_obs_dim), dtype=np.float32
            )
            
            for i in range(num_samples):
                start_idx = i
                end_idx = i + self.history_length
                locomotion_obs_history[i] = loco_obs_array[start_idx:end_idx]
            
            episode_data['locomotion_obs_history'] = locomotion_obs_history
            episode_data['single_obs_dim'] = self.single_obs_dim
        
        # Store episode
        self.all_episodes.append(episode_data)
        
        return episode_data
    
    def collect_data(self, num_episodes: int = 100, verbose: bool = True) -> Dict[str, np.ndarray]:
        """
        Collect data from multiple episodes.
        
        Args:
            num_episodes: Number of episodes to collect
            verbose: Print progress
            
        Returns:
            Combined data dictionary
        """
        if self.locomotion_env is None:
            self._setup()
        
        self.all_episodes = []  # Reset
        
        for ep in range(num_episodes):
            if verbose and (ep + 1) % 10 == 0:
                print(f"  Collecting episode {ep + 1}/{num_episodes}")
            
            self.collect_episode()
        
        return self.get_combined_data()
    
    def get_combined_data(self) -> Dict[str, np.ndarray]:
        """Get all collected data combined."""
        if not self.all_episodes:
            return {}
        
        combined = {
            'distance_history': np.concatenate([ep['distance_history'] for ep in self.all_episodes]),
            'delta_history': np.concatenate([ep['delta_history'] for ep in self.all_episodes]),
            'bearing': np.concatenate([ep['bearing'] for ep in self.all_episodes]),
            'policy_index': np.concatenate([ep['policy_index'] for ep in self.all_episodes]),
            'policy_onehot': np.concatenate([ep['policy_onehot'] for ep in self.all_episodes]),
        }
        
        # Include locomotion observations if available
        if 'locomotion_obs_history' in self.all_episodes[0]:
            combined['locomotion_obs_history'] = np.concatenate(
                [ep['locomotion_obs_history'] for ep in self.all_episodes]
            )
            combined['single_obs_dim'] = self.all_episodes[0]['single_obs_dim']
        
        return combined
    
    def save_data(self, save_path: str):
        """Save all collected data to file."""
        combined = self.get_combined_data()
        if not combined:
            print("[MultiPolicyCollector] No data to save!")
            return
        
        # Add metadata
        combined['history_length'] = self.history_length
        combined['distance_noise_std'] = self.distance_noise_std
        combined['target_spawn_range_min'] = self.target_spawn_range[0]
        combined['target_spawn_range_max'] = self.target_spawn_range[1]
        combined['num_episodes'] = len(self.all_episodes)
        combined['num_policies'] = self.num_policies
        combined['policy_names'] = np.array(self.policy_names)
        combined['source'] = 'simulation_multipolicy'
        
        np.savez(save_path, **combined)
        print(f"[MultiPolicyCollector] Data saved to: {save_path}")
        print(f"  Total samples: {len(combined['bearing'])}")
        print(f"  Episodes: {len(self.all_episodes)}")
        print(f"  Policies: {self.num_policies} ({self.policy_names})")
        
        # Print policy distribution
        policy_counts = np.bincount(combined['policy_index'], minlength=self.num_policies)
        print(f"  Policy distribution:")
        for i, (name, count) in enumerate(zip(self.policy_names, policy_counts)):
            pct = 100 * count / len(combined['policy_index'])
            print(f"    [{i}] {name}: {count} ({pct:.1f}%)")
        
        if 'locomotion_obs_history' in combined:
            print(f"  Locomotion obs dim: {combined['single_obs_dim']}")
    
    # =========================================================================
    # Rendering Methods
    # =========================================================================
    
    def _add_bearing_visualization_to_scene(
        self, 
        scene, 
        target_pos: np.ndarray, 
        robot_pos: np.ndarray,
        heading: float,
    ) -> None:
        """Add comprehensive bearing visualization to the MuJoCo scene."""
        from metamachine.utils.rendering import (
            add_ground_disc_marker, 
            add_sphere_marker, 
            add_arrow_marker
        )
        
        robot_xy = robot_pos[:2]
        
        # 1. Target marker
        add_ground_disc_marker(
            scene,
            pos_xy=target_pos,
            radius=0.3,
            height=0.02,
            color=(0.0, 1.0, 0.0, 1.0),
            z_offset=0.01
        )
        
        add_sphere_marker(
            scene,
            pos=np.array([target_pos[0], target_pos[1], 0.15]),
            radius=0.1,
            color=(1.0, 0.2, 0.0, 1.0),
        )
        
        # 2. Arrow from robot to target
        direction = target_pos - robot_xy
        length = float(np.linalg.norm(direction))
        if length > 0.1:
            dir_normalized = direction / length
            arrow_length = min(length * 0.8, 2.0)
            
            add_arrow_marker(
                scene,
                start=np.array([robot_xy[0], robot_xy[1], 0.08]),
                direction=np.array([dir_normalized[0], dir_normalized[1], 0]),
                length=arrow_length,
                radius=0.03,
                color=(1.0, 1.0, 0.0, 1.0),
            )
        
        # 3. Heading indicator (forward direction)
        # heading = angle of local +Y from world +X, counterclockwise
        # So forward in world = [cos(heading), sin(heading)]
        heading_dir = np.array([np.cos(heading), np.sin(heading)])
        add_arrow_marker(
            scene,
            start=np.array([robot_xy[0], robot_xy[1], 0.06]),
            direction=np.array([heading_dir[0], heading_dir[1], 0]),
            length=0.5,
            radius=0.02,
            color=(0.0, 1.0, 1.0, 1.0),
        )
        
        # 4. Robot position marker
        add_ground_disc_marker(
            scene,
            pos_xy=robot_xy,
            radius=0.08,
            height=0.02,
            color=(0.0, 0.5, 1.0, 1.0),
            z_offset=0.02
        )
    
    def _add_bearing_metrics_overlay(
        self,
        frame: np.ndarray,
        bearing: float,
        distance: float,
        step: int,
        heading: float,
        policy_idx: int,
        policy_name: str,
    ) -> np.ndarray:
        """Add bearing, distance, and policy metrics overlay to frame."""
        import cv2
        
        # Bearing direction indicator
        bearing_deg = np.degrees(bearing)
        heading_deg = np.degrees(heading)
        if bearing_deg > 5:
            direction = "LEFT"
        elif bearing_deg < -5:
            direction = "RIGHT"
        else:
            direction = "FRONT"
        
        metrics = [
            "=== BEARING INFO ===",
            f"Bearing: {bearing_deg:+7.1f} deg",
            f"Direction: {direction}",
            f"Distance: {distance:.2f} m",
            f"Heading: {heading_deg:+7.1f} deg",
            f"Step: {step}",
            "",
            "=== POLICY INFO ===",
            f"Policy: [{policy_idx}] {policy_name}",
        ]
        
        # Configure text appearance
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 255)  # Cyan
        thickness = 2
        line_height = 25
        
        # Position at right side of frame
        frame_width = frame.shape[1]
        start_x = frame_width - 280
        start_y = 150
        
        # Draw semi-transparent background
        overlay = frame.copy()
        bg_height = len(metrics) * line_height + 15
        cv2.rectangle(overlay, (start_x - 10, start_y - 20), 
                      (frame_width - 10, start_y + bg_height - 10), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        
        for i, text in enumerate(metrics):
            y_pos = start_y + (i * line_height)
            cv2.putText(frame, text, (start_x, y_pos), font, font_scale, color, thickness)
        
        return frame
    
    def _capture_frame_with_target(
        self, 
        target_pos: np.ndarray, 
        bearing: float, 
        distance: float, 
        step: int,
        heading: float,
        policy_idx: int,
        policy_name: str,
    ) -> Optional[np.ndarray]:
        """Capture a frame with target marker, bearing info, and policy indicator."""
        import cv2
        import mujoco
        
        env = self.locomotion_env
        
        if env.render_mode != "mp4" or not env.recording_active:
            return None
        
        renderer = env._create_egl_renderer()
        
        if renderer == "synthetic":
            width, height = env.render_size
            frame = env._create_synthetic_frame(width, height)
        else:
            camera_id = getattr(env, "preferred_camera_id", -1)
            renderer.update_scene(env.data, camera=camera_id)
            
            # Add bearing visualization to scene
            robot_pos = self._get_robot_position()
            self._add_bearing_visualization_to_scene(
                renderer.scene, 
                target_pos, 
                robot_pos,
                heading,
            )
            
            pixels = renderer.render()
            frame = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        
        # Add standard metrics overlay from env
        frame = env._add_metrics_overlay(frame)
        
        # Add bearing and policy metrics overlay
        frame = self._add_bearing_metrics_overlay(
            frame, bearing, distance, step, heading, policy_idx, policy_name
        )
        
        env.video_frames.append(frame)
        return frame
    
    def collect_episode_with_render(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Collect one episode with rendering enabled.
        
        Returns:
            Episode data dictionary, or None if not enough samples
        """
        # Reset environment
        obs, _ = self.locomotion_env.reset()
        
        # Start video recording
        if self.locomotion_env.render_mode == "mp4":
            self.locomotion_env.start_video_recording()
        
        # Spawn target
        target_pos = self._spawn_target()
        
        # Initialize current policy randomly
        current_policy_idx = np.random.randint(0, self.num_policies)
        current_policy_name = self.policy_names[current_policy_idx]
        self.policy_runner.select_policy(current_policy_idx)
        
        # Initialize steps until next switch
        steps_until_switch = np.random.randint(
            self.policy_switch_interval[0], 
            self.policy_switch_interval[1]
        )
        
        # Initialize buffers
        distances = []
        bearings = []
        policy_indices = []
        locomotion_obs = []
        
        # Run episode
        for step in range(self.max_steps_per_episode):
            # Check if we should switch policies
            if steps_until_switch <= 0:
                current_policy_idx = np.random.randint(0, self.num_policies)
                current_policy_name = self.policy_names[current_policy_idx]
                self.policy_runner.select_policy(current_policy_idx)
                steps_until_switch = np.random.randint(
                    self.policy_switch_interval[0], 
                    self.policy_switch_interval[1]
                )
            
            # Record data
            distance = self._get_distance_to_target(target_pos)
            bearing = self._get_bearing_to_target(target_pos)
            heading = self._get_robot_heading()
            
            # Add noise to distance
            noisy_distance = distance + np.random.normal(0, self.distance_noise_std)
            noisy_distance = max(0.01, noisy_distance)
            
            distances.append(noisy_distance)
            bearings.append(bearing)
            policy_indices.append(current_policy_idx)
            
            # Collect single-timestep observation
            if self.include_locomotion_obs:
                single_obs = self._get_single_timestep_obs()
                locomotion_obs.append(single_obs.copy())
            
            # Get action from current policy
            action, _ = self.policy_runner.predict(obs, deterministic=True)
            action = np.asarray(action).flatten() * self.action_scale
            
            # Temporarily disable recording to prevent env's default frame capture
            if self.locomotion_env.render_mode == "mp4":
                self.locomotion_env.recording_active = False
            
            # Step environment
            obs, reward, terminated, truncated, info = self.locomotion_env.step(action)
            
            # Re-enable recording and capture custom frame AFTER step (physics updated)
            if self.locomotion_env.render_mode == "mp4":
                self.locomotion_env.recording_active = True
                self._capture_frame_with_target(
                    target_pos, bearing, distance, step, heading, 
                    current_policy_idx, current_policy_name
                )
            
            steps_until_switch -= 1
            
            if terminated:
                break
        
        # Check if we have enough samples
        if len(distances) < self.history_length + 1:
            print(f"[MultiPolicyCollector] Episode too short ({len(distances)} samples), skipping")
            return None
        
        # Convert to arrays
        distances = np.array(distances, dtype=np.float32)
        bearings = np.array(bearings, dtype=np.float32)
        policy_indices = np.array(policy_indices, dtype=np.int64)
        
        # Create sliding window samples
        num_samples = len(distances) - self.history_length
        
        distance_history = np.zeros((num_samples, self.history_length), dtype=np.float32)
        delta_history = np.zeros((num_samples, self.history_length), dtype=np.float32)
        target_bearings = np.zeros(num_samples, dtype=np.float32)
        sample_policy_indices = np.zeros(num_samples, dtype=np.int64)
        policy_onehot = np.zeros((num_samples, self.num_policies), dtype=np.float32)
        
        for i in range(num_samples):
            start_idx = i
            end_idx = i + self.history_length
            
            distance_history[i] = distances[start_idx:end_idx]
            
            # Compute delta distances
            deltas = np.diff(distances[start_idx:end_idx + 1])
            delta_history[i] = -deltas  # Negate: getting closer = positive
            
            # Target bearing at end of window
            target_bearings[i] = bearings[end_idx - 1]
            
            # Policy at end of window
            sample_policy_indices[i] = policy_indices[end_idx - 1]
            policy_onehot[i, policy_indices[end_idx - 1]] = 1.0
        
        episode_data = {
            'distance_history': distance_history,
            'delta_history': delta_history,
            'bearing': target_bearings,
            'policy_index': sample_policy_indices,
            'policy_onehot': policy_onehot,
            'full_distances': distances,
            'full_bearings': bearings,
            'full_policy_indices': policy_indices,
            'target_position': target_pos,
        }
        
        return episode_data
    
    def render_example_episode(self, video_path: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Render a single example episode with visualization for verification.
        
        This creates a video showing:
        - Target position as a marker on the ground
        - Bearing and distance information in the overlay
        - Current policy indicator in the overlay
        - Policy switching during the episode
        
        Args:
            video_path: Path to save the video
            
        Returns:
            Episode data dictionary
        """
        print(f"[MultiPolicyCollector] Rendering example episode...")
        print(f"  Video will be saved to: {video_path}")
        print(f"  Policies: {self.policy_names}")
        
        # Close existing environment if any
        if self.locomotion_env is not None:
            self.locomotion_env.close()
            self.locomotion_env = None
        
        # Setup with rendering enabled
        self._setup(render_mode="mp4")
        
        # Configure video path
        self.locomotion_env.video_path = str(Path(video_path).parent)
        video_name = Path(video_path).stem
        self.locomotion_env.video_name_pattern = video_name
        
        # Collect one episode with rendering
        episode_data = self.collect_episode_with_render()
        
        if episode_data is not None:
            print(f"[MultiPolicyCollector] Example episode rendered successfully!")
            print(f"  Steps: {len(episode_data['full_bearings'])}")
            print(f"  Bearing range: [{np.degrees(episode_data['full_bearings'].min()):.1f}°, "
                  f"{np.degrees(episode_data['full_bearings'].max()):.1f}°]")
            
            # Print policy usage summary
            policy_counts = np.bincount(
                episode_data['full_policy_indices'], minlength=self.num_policies
            )
            print(f"  Policy usage:")
            for i, (name, count) in enumerate(zip(self.policy_names, policy_counts)):
                print(f"    [{i}] {name}: {count} steps")
        
        return episode_data
    
    def close(self):
        """Cleanup."""
        if self.locomotion_env is not None:
            self.locomotion_env.close()


# =============================================================================
# Bearing Estimator Network V3 (with policy indicator)
# =============================================================================

class BearingEstimatorV3(nn.Module):
    """
    Bearing estimator with optional policy indicator input.
    
    This extends BearingEstimatorV2 by:
    1. Accepting a one-hot policy indicator as additional input
    2. Conditioning the prediction on which policy is currently running
    
    This allows the network to learn policy-specific bearing estimation,
    accounting for different motion patterns of different policies.
    """
    
    def __init__(
        self,
        history_length: int = 30,
        num_policies: int = 0,  # 0 = no policy conditioning
        hidden_dims: List[int] = [256, 256, 128],
        use_lstm: bool = True,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        locomotion_obs_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.history_length = history_length
        self.num_policies = num_policies
        self.use_lstm = use_lstm
        self.locomotion_obs_dim = locomotion_obs_dim
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        
        # Feature encoder (processes each timestep's input)
        input_per_step = 2 + locomotion_obs_dim  # distance + delta + loco_obs
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_per_step, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=64,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0,
                bidirectional=False,
            )
            head_input = lstm_hidden
        else:
            self.flatten = nn.Flatten()
            self.mlp = nn.Sequential(
                nn.Linear(64 * history_length, hidden_dims[0]),
                nn.LayerNorm(hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for i in range(len(hidden_dims) - 1):
                self.mlp.add_module(f"linear_{i+1}", nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                self.mlp.add_module(f"norm_{i+1}", nn.LayerNorm(hidden_dims[i+1]))
                self.mlp.add_module(f"relu_{i+1}", nn.ReLU())
                self.mlp.add_module(f"dropout_{i+1}", nn.Dropout(dropout))
            head_input = hidden_dims[-1]
        
        # Add policy indicator to head input
        if num_policies > 0:
            head_input += num_policies
        
        # Output head: predict sin, cos
        self.head = nn.Sequential(
            nn.Linear(head_input, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # sin, cos
        )
        
        # Hidden state for inference (LSTM only)
        self._hidden = None
    
    def reset_hidden(self, batch_size: int = 1):
        """Reset LSTM hidden state for new sequence inference."""
        if self.use_lstm:
            device = next(self.parameters()).device
            self._hidden = (
                torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device),
                torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device),
            )
    
    def forward(
        self,
        distance_history: torch.Tensor,
        delta_history: torch.Tensor,
        locomotion_obs_history: Optional[torch.Tensor] = None,
        policy_indicator: Optional[torch.Tensor] = None,
        use_hidden: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            distance_history: (batch, seq_len, history_length) or (batch, history_length)
            delta_history: Same shape as distance_history
            locomotion_obs_history: Optional locomotion observations
            policy_indicator: (batch, [seq_len,] num_policies) one-hot or (batch, [seq_len]) index
            use_hidden: If True and LSTM, use and update internal hidden state
            
        Returns:
            (batch, [seq_len,] 2) predicted sin/cos
        """
        is_sequence = distance_history.dim() == 3
        
        if not is_sequence:
            distance_history = distance_history.unsqueeze(1)
            delta_history = delta_history.unsqueeze(1)
            if locomotion_obs_history is not None:
                locomotion_obs_history = locomotion_obs_history.unsqueeze(1)
            if policy_indicator is not None and policy_indicator.dim() == 1:
                policy_indicator = policy_indicator.unsqueeze(1)
            elif policy_indicator is not None and policy_indicator.dim() == 2:
                policy_indicator = policy_indicator.unsqueeze(1)
        
        batch_size, seq_len, hist_len = distance_history.shape
        
        # Reshape for feature encoding
        dist_flat = distance_history.view(batch_size * seq_len, hist_len, 1)
        delta_flat = delta_history.view(batch_size * seq_len, hist_len, 1)
        
        if locomotion_obs_history is not None and self.locomotion_obs_dim > 0:
            loco_flat = locomotion_obs_history.view(batch_size * seq_len, hist_len, self.locomotion_obs_dim)
            features = torch.cat([dist_flat, delta_flat, loco_flat], dim=-1)
        else:
            features = torch.cat([dist_flat, delta_flat], dim=-1)
        
        # Encode features
        encoded = self.feature_encoder(features)
        
        if self.use_lstm:
            if use_hidden and self._hidden is not None:
                if self._hidden[0].size(1) != batch_size * seq_len:
                    self.reset_hidden(batch_size * seq_len)
                lstm_out, self._hidden = self.lstm(encoded, self._hidden)
            else:
                lstm_out, _ = self.lstm(encoded)
            
            lstm_last = lstm_out[:, -1, :]  # (batch * seq_len, lstm_hidden)
            pre_head = lstm_last
        else:
            flat = self.flatten(encoded)
            mlp_out = self.mlp(flat)
            pre_head = mlp_out
        
        # Add policy indicator if provided
        if self.num_policies > 0 and policy_indicator is not None:
            # Handle different input formats
            if policy_indicator.dim() == 2 and policy_indicator.size(-1) != self.num_policies:
                # It's an index tensor, convert to one-hot
                policy_indicator = policy_indicator.view(-1)
                policy_onehot = torch.zeros(batch_size * seq_len, self.num_policies, 
                                           device=policy_indicator.device)
                policy_onehot.scatter_(1, policy_indicator.unsqueeze(-1), 1.0)
            elif policy_indicator.dim() == 3:
                # (batch, seq_len, num_policies) -> (batch * seq_len, num_policies)
                policy_onehot = policy_indicator.view(batch_size * seq_len, self.num_policies)
            else:
                policy_onehot = policy_indicator.view(batch_size * seq_len, self.num_policies)
            
            pre_head = torch.cat([pre_head, policy_onehot], dim=-1)
        
        # Output head
        output = self.head(pre_head)
        
        # Reshape back
        output = output.view(batch_size, seq_len, 2)
        
        if not is_sequence:
            output = output.squeeze(1)
        
        # Normalize to unit circle
        output = output / (torch.norm(output, dim=-1, keepdim=True) + 1e-8)
        
        return output
    
    def predict_bearing(
        self,
        distance_history: torch.Tensor,
        delta_history: torch.Tensor,
        locomotion_obs_history: Optional[torch.Tensor] = None,
        policy_indicator: Optional[torch.Tensor] = None,
        use_hidden: bool = False,
    ) -> torch.Tensor:
        """Predict bearing angle in radians."""
        sin_cos = self.forward(
            distance_history, delta_history, 
            locomotion_obs_history, policy_indicator,
            use_hidden
        )
        if sin_cos.dim() == 3:
            bearing = torch.atan2(sin_cos[:, :, 0], sin_cos[:, :, 1])
        else:
            bearing = torch.atan2(sin_cos[:, 0], sin_cos[:, 1])
        return bearing


# =============================================================================
# Runtime Bearing Estimator with Policy Indicator
# =============================================================================

class BearingEstimatorRunner:
    """
    Runs bearing estimation on real robot with support for multiple policies.
    
    This class:
    1. Maintains history buffers
    2. Accepts current policy index for policy-conditioned prediction
    3. Supports Kalman filtering for smoothing
    """
    
    def __init__(
        self,
        estimator_log_dir: str,
        checkpoint: str = "bearing_estimator.pt",
        device: str = "cpu",
        use_kalman: bool = True,
        kalman_measurement_noise: float = 0.3,
        bearing_offset_deg: float = 0.0,
    ):
        """Initialize the bearing estimator runner."""
        self.device = device
        self.estimator_log_dir = Path(estimator_log_dir)
        self.use_kalman = use_kalman
        self.kalman_measurement_noise = kalman_measurement_noise
        self.bearing_offset_rad = np.radians(bearing_offset_deg)
        
        # Load the model
        self._load_model(checkpoint)
        
        # Initialize history buffers
        self.distance_history = np.zeros(self.history_length, dtype=np.float32)
        self.prev_distance = 0.0
        self.last_distance = 0.0
        self.history_filled = False
        self.samples_received = 0
        
        # Locomotion obs history
        if self.include_locomotion_obs and self.locomotion_obs_dim > 0:
            self.locomotion_obs_history = np.zeros(
                (self.history_length, self.locomotion_obs_dim), dtype=np.float32
            )
        else:
            self.locomotion_obs_history = None
        
        # Kalman filter
        if self.model_version == 'v3' and self.use_kalman:
            self._init_kalman_filter()
        else:
            self.kalman = None
        
        # Tracking
        self.current_bearing = 0.0
        self.current_bearing_raw = 0.0
        self.bearing_history = []
        self.bearing_history_raw = []
        self.current_policy_idx = 0
    
    def _init_kalman_filter(self):
        """Initialize Kalman filter for bearing smoothing."""
        self.kalman = {
            'x': np.array([0.0, 0.0]),
            'P': np.array([[1.0, 0.0], [0.0, 1.0]]),
            'dt': 0.05,
        }
        process_noise_bearing = 0.1
        process_noise_rate = 0.5
        self.kalman['Q'] = np.array([
            [process_noise_bearing * self.kalman['dt'], 0.0],
            [0.0, process_noise_rate * self.kalman['dt']],
        ])
        self.kalman['R'] = np.array([[self.kalman_measurement_noise]])
        self.kalman['F'] = np.array([
            [1.0, self.kalman['dt']],
            [0.0, 1.0],
        ])
        self.kalman['H'] = np.array([[1.0, 0.0]])
    
    def _kalman_filter(self, measurement: float) -> float:
        """Apply Kalman filter to smooth bearing prediction."""
        if self.kalman is None:
            return measurement
        
        x_pred = self.kalman['F'] @ self.kalman['x']
        x_pred[0] = np.arctan2(np.sin(x_pred[0]), np.cos(x_pred[0]))
        P_pred = self.kalman['F'] @ self.kalman['P'] @ self.kalman['F'].T + self.kalman['Q']
        
        y = measurement - x_pred[0]
        y = np.arctan2(np.sin(y), np.cos(y))
        
        S = self.kalman['H'] @ P_pred @ self.kalman['H'].T + self.kalman['R']
        K = P_pred @ self.kalman['H'].T @ np.linalg.inv(S)
        
        self.kalman['x'] = x_pred + K.flatten() * y
        self.kalman['x'][0] = np.arctan2(np.sin(self.kalman['x'][0]), np.cos(self.kalman['x'][0]))
        self.kalman['P'] = (np.eye(2) - K @ self.kalman['H']) @ P_pred
        
        return self.kalman['x'][0]
    
    def _load_model(self, checkpoint: str) -> None:
        """Load the trained model."""
        checkpoint_path = self.estimator_log_dir / checkpoint
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"[BearingEstimator] Loading model from: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model_version = checkpoint_data.get('model_version', 'v1')
        print(f"  Model version: {self.model_version}")
        
        self.history_length = checkpoint_data['history_length']
        self.use_lstm = checkpoint_data.get('use_lstm', False)
        self.locomotion_obs_dim = checkpoint_data.get('locomotion_obs_dim', 0)
        self.include_locomotion_obs = checkpoint_data.get('include_locomotion_obs', False)
        self.num_policies = checkpoint_data.get('num_policies', 0)
        self.policy_names = checkpoint_data.get('policy_names', [])
        
        print(f"  History length: {self.history_length}")
        print(f"  Architecture: {'LSTM' if self.use_lstm else 'MLP'}")
        print(f"  Include locomotion obs: {self.include_locomotion_obs}")
        print(f"  Num policies: {self.num_policies}")
        if self.policy_names:
            print(f"  Policy names: {self.policy_names}")
        
        # Create model
        if self.model_version == 'v3':
            self.model = BearingEstimatorV3(
                history_length=self.history_length,
                num_policies=self.num_policies,
                hidden_dims=[256, 256, 128],
                use_lstm=self.use_lstm,
                lstm_hidden=128,
                lstm_layers=2,
                locomotion_obs_dim=self.locomotion_obs_dim,
                dropout=0.0,
            )
        else:
            # Fall back to V2 or V1
            # Import from train_bearing_estimator_v2 if needed
            raise ValueError(f"Model version {self.model_version} not supported in this runner. "
                           f"Use the runner from train_bearing_estimator_v2.py for V1/V2 models.")
        
        self.model.load_state_dict(checkpoint_data['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        if self.use_lstm:
            self.model.reset_hidden(1)
        
        print(f"  Model loaded successfully!")
    
    def reset(self, initial_distance: float = 5.0) -> None:
        """Reset the estimator state."""
        self.distance_history[:] = initial_distance
        self.prev_distance = initial_distance
        self.last_distance = initial_distance
        self.history_filled = False
        self.samples_received = 0
        self.current_bearing = 0.0
        self.current_bearing_raw = 0.0
        self.bearing_history = []
        self.bearing_history_raw = []
        
        if self.locomotion_obs_history is not None:
            self.locomotion_obs_history[:] = 0.0
        
        if self.kalman is not None:
            self.kalman['x'] = np.array([0.0, 0.0])
            self.kalman['P'] = np.array([[1.0, 0.0], [0.0, 0.1]])
        
        if self.use_lstm:
            self.model.reset_hidden(1)
    
    def set_policy(self, policy_idx: int) -> None:
        """Set the current policy index for prediction."""
        if self.num_policies > 0:
            self.current_policy_idx = policy_idx % self.num_policies
    
    def update(
        self, 
        distance: float, 
        locomotion_obs: Optional[np.ndarray] = None,
        policy_idx: Optional[int] = None,
    ) -> float:
        """
        Update with a new distance measurement and estimate bearing.
        
        Args:
            distance: Current distance to target
            locomotion_obs: Single-timestep locomotion observation
            policy_idx: Current policy index (overrides self.current_policy_idx)
            
        Returns:
            Estimated bearing in radians
        """
        if policy_idx is not None:
            self.current_policy_idx = policy_idx
        
        self.samples_received += 1
        
        if self.samples_received > self.history_length:
            self.prev_distance = self.distance_history[0]
        
        self.distance_history = np.roll(self.distance_history, -1)
        self.distance_history[-1] = distance
        
        # IMPORTANT: Always roll locomotion_obs_history to keep it aligned with distance_history
        # Even if we don't have a new observation, we need to maintain sync
        if self.locomotion_obs_history is not None:
            self.locomotion_obs_history = np.roll(self.locomotion_obs_history, -1, axis=0)
            if locomotion_obs is not None:
                # Update with the new observation
                self.locomotion_obs_history[-1] = locomotion_obs.flatten()[:self.locomotion_obs_dim]
            # If locomotion_obs is None, we keep the rolled value (which was from -2 position)
            # This maintains alignment but may use stale data - acceptable for intermediate steps
        
        if self.samples_received >= self.history_length:
            self.history_filled = True
        
        if not self.history_filled:
            self.last_distance = distance
            self.current_bearing = 0.0
            return 0.0
        
        if self.samples_received == self.history_length:
            self.prev_distance = self.distance_history[0]
            self.last_distance = distance
            self.current_bearing = 0.0
            return 0.0
        
        # Compute delta history
        distances_with_prev = np.concatenate([[self.prev_distance], self.distance_history])
        raw_diff = np.diff(distances_with_prev)
        delta_history = -raw_diff
        
        # Run inference
        with torch.no_grad():
            dist_tensor = torch.from_numpy(self.distance_history.astype(np.float32)).unsqueeze(0).to(self.device)
            delta_tensor = torch.from_numpy(delta_history.astype(np.float32)).unsqueeze(0).to(self.device)
            
            loco_tensor = None
            if self.locomotion_obs_history is not None and self.include_locomotion_obs:
                # Shape must be (batch, history_length, locomotion_obs_dim) = (1, 30, 15)
                # NOT flattened to (1, 450)!
                loco_tensor = torch.from_numpy(
                    self.locomotion_obs_history.astype(np.float32)
                ).unsqueeze(0).to(self.device)
            
            # Policy indicator
            policy_tensor = None
            if self.num_policies > 0:
                policy_onehot = np.zeros(self.num_policies, dtype=np.float32)
                policy_onehot[self.current_policy_idx] = 1.0
                policy_tensor = torch.from_numpy(policy_onehot).unsqueeze(0).to(self.device)
            
            bearing_raw = self.model.predict_bearing(
                dist_tensor, delta_tensor, loco_tensor, policy_tensor,
                use_hidden=self.use_lstm
            ).item()
        
        bearing_raw = bearing_raw + self.bearing_offset_rad
        bearing_raw = np.arctan2(np.sin(bearing_raw), np.cos(bearing_raw))
        
        self.last_distance = distance
        self.current_bearing_raw = bearing_raw
        
        if self.kalman is not None:
            bearing = self._kalman_filter(bearing_raw)
        else:
            bearing = bearing_raw
        
        self.current_bearing = bearing
        self.bearing_history.append(bearing)
        self.bearing_history_raw.append(bearing_raw)
        
        if len(self.bearing_history) > 1000:
            self.bearing_history = self.bearing_history[-500:]
            self.bearing_history_raw = self.bearing_history_raw[-500:]
        
        return bearing
    
    def get_bearing_degrees(self, filtered: bool = True) -> float:
        """Get current estimated bearing in degrees."""
        if filtered:
            return np.degrees(self.current_bearing)
        else:
            return np.degrees(self.current_bearing_raw)
    
    def is_ready(self) -> bool:
        """Check if enough samples have been collected."""
        return self.samples_received > self.history_length
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status for display."""
        return {
            'bearing_rad': self.current_bearing,
            'bearing_deg': np.degrees(self.current_bearing),
            'bearing_raw_rad': self.current_bearing_raw,
            'bearing_raw_deg': np.degrees(self.current_bearing_raw),
            'distance': self.last_distance,
            'history_filled': self.history_filled,
            'samples_received': self.samples_received,
            'history_length': self.history_length,
            'current_policy_idx': self.current_policy_idx,
            'num_policies': self.num_policies,
            'kalman_enabled': self.kalman is not None,
        }


# =============================================================================
# Bearing-Augmented Environment Configuration
# =============================================================================


@dataclass
class BearingAugmentedConfig:
    """Configuration for bearing-augmented policy switch environment.
    
    This extends PolicySwitchConfig with bearing estimation settings.
    
    Attributes:
        # Bearing source
        use_ground_truth_bearing: If True, use ground-truth bearing. If False, use estimator.
        bearing_estimator_log_dir: Path to trained bearing estimator log directory
        bearing_estimator_checkpoint: Checkpoint filename (default: bearing_estimator.pt)
        
        # Estimator settings
        estimator_use_kalman: Whether to use Kalman filter for smoothing
        estimator_kalman_noise: Kalman filter measurement noise
        estimator_bearing_offset_deg: Bearing offset correction in degrees
        estimator_include_locomotion_obs: Whether estimator uses locomotion obs
        estimator_history_length: History length for estimator (auto-loaded if None)
        
        # Observation settings
        bearing_obs_format: How to encode bearing - 'sin_cos' or 'angle'
        include_intra_planning_bearings: Whether to include bearing estimates for each
            of the N intra-planning steps (produces N bearing values instead of just 1)
        
        # Visualization settings
        enable_bearing_visualization: Whether to enable bearing visualization
        enable_mujoco_overlay: Add bearing text overlay to MuJoCo video frames
        enable_mujoco_markers: Add 3D markers (target, forward dir) to MuJoCo scene
        enable_chase_bearing: Add bearing info to matplotlib chase video
    """
    # Bearing source
    use_ground_truth_bearing: bool = False
    bearing_estimator_log_dir: Optional[str] = None
    bearing_estimator_checkpoint: str = "bearing_estimator.pt"
    
    # Estimator settings
    estimator_use_kalman: bool = True
    estimator_kalman_noise: float = 0.3
    estimator_bearing_offset_deg: float = 0.0
    estimator_include_locomotion_obs: bool = False
    estimator_history_length: Optional[int] = None  # Auto-loaded from checkpoint
    
    # Observation settings
    bearing_obs_format: str = "sin_cos"  # 'sin_cos' or 'angle'
    include_intra_planning_bearings: bool = False  # Include bearing for each of N locomotion steps
    
    # Visualization settings
    enable_bearing_visualization: bool = True
    enable_mujoco_overlay: bool = True
    enable_mujoco_markers: bool = True
    enable_chase_bearing: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BearingAugmentedConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# =============================================================================
# Bearing-Augmented Policy Switch Environment
# =============================================================================


class BearingAugmentedPolicySwitchEnv(gym.Wrapper):
    """
    Wrapper that augments HierarchicalPolicySwitchEnv with bearing estimation.
    
    This wrapper adds predicted or ground-truth bearing (as sin/cos) to the
    observation space, enabling the high-level planning agent to use directional
    information when making controller selection decisions.
    
    The bearing can come from:
    1. Ground-truth: Calculated from robot pose and target position (for comparison)
    2. Estimator: A trained BearingEstimatorV3 that predicts bearing from distance history
    
    Example:
        >>> from metamachine.utils.bearing_estimation import (
        ...     BearingAugmentedPolicySwitchEnv, BearingAugmentedConfig
        ... )
        >>> from planning.policy_switch_env import (
        ...     HierarchicalPolicySwitchEnv, PolicySwitchConfig
        ... )
        >>> 
        >>> # Create base environment
        >>> base_config = PolicySwitchConfig(
        ...     locomotion_checkpoint_dirs=["logs/ctrl1", "logs/ctrl2", "logs/ctrl3"],
        ...     target_mode="moving",
        ... )
        >>> base_env = HierarchicalPolicySwitchEnv(base_config)
        >>> 
        >>> # Wrap with bearing augmentation (using estimator)
        >>> bearing_config = BearingAugmentedConfig(
        ...     use_ground_truth_bearing=False,
        ...     bearing_estimator_log_dir="logs/bearing_estimator",
        ... )
        >>> env = BearingAugmentedPolicySwitchEnv(base_env, bearing_config)
        >>> 
        >>> obs, info = env.reset()
        >>> # obs now includes [sin(bearing), cos(bearing)] at the end
    """
    
    def __init__(
        self,
        env: gym.Env,
        bearing_config: BearingAugmentedConfig,
        device: str = "cpu",
    ):
        """Initialize the bearing-augmented environment.
        
        Args:
            env: Base HierarchicalPolicySwitchEnv instance
            bearing_config: Bearing augmentation configuration
            device: Device for bearing estimator inference
        """
        super().__init__(env)
        
        self.bearing_config = bearing_config
        self.device = device
        
        # Store reference to unwrapped env for accessing internal state
        self._base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
        
        # Get number of controllers from base env
        if hasattr(self._base_env, 'num_controllers'):
            self.num_controllers = self._base_env.num_controllers
        else:
            self.num_controllers = self._base_env.action_space.n
        
        # Setup bearing source
        self.use_ground_truth = bearing_config.use_ground_truth_bearing
        self.bearing_estimator = None
        
        if not self.use_ground_truth:
            self._setup_bearing_estimator()
        
        # Check if base env uses intra-planning observations (multiple distance samples per step)
        has_intra_planning = (
            hasattr(self._base_env, 'config') and
            getattr(self._base_env.config, 'include_intra_planning_obs', False)
        )
        locomotion_steps = getattr(self._base_env.config, 'locomotion_steps_per_planning', 1) if hasattr(self._base_env, 'config') else 1
        
        # Track whether we should include intra-planning bearings (one for each locomotion step)
        self.include_intra_planning_bearings = (
            bearing_config.include_intra_planning_bearings and 
            has_intra_planning and 
            locomotion_steps > 1
        )
        self.locomotion_steps_per_planning = locomotion_steps
        
        # Initialize intra-planning bearings buffer
        if self.include_intra_planning_bearings:
            self.intra_planning_bearings = np.zeros(locomotion_steps, dtype=np.float32)
            self.intra_planning_bearings_sin_cos = np.zeros((locomotion_steps, 2), dtype=np.float32)
        else:
            self.intra_planning_bearings = None
            self.intra_planning_bearings_sin_cos = None
        
        # Determine bearing observation dimension
        if bearing_config.bearing_obs_format == "sin_cos":
            if self.include_intra_planning_bearings:
                # N bearings * 2 (sin, cos each)
                self.bearing_obs_dim = locomotion_steps * 2
            else:
                self.bearing_obs_dim = 2  # [sin(bearing), cos(bearing)]
        else:
            if self.include_intra_planning_bearings:
                self.bearing_obs_dim = locomotion_steps  # N bearings
            else:
                self.bearing_obs_dim = 1  # [bearing_angle]
        
        # Extend observation space
        base_obs_shape = env.observation_space.shape
        new_obs_dim = base_obs_shape[0] + self.bearing_obs_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(new_obs_dim,),
            dtype=np.float32
        )
        
        print(f"[BearingAugmentedEnv] Initialized")
        print(f"  Bearing source: {'ground-truth' if self.use_ground_truth else 'estimator'}")
        print(f"  Bearing obs format: {bearing_config.bearing_obs_format}")
        print(f"  Original obs dim: {base_obs_shape[0]}")
        print(f"  Augmented obs dim: {new_obs_dim}")
        if not self.use_ground_truth and has_intra_planning and locomotion_steps > 1:
            print(f"  Intra-planning distances: {locomotion_steps} samples per planning step")
            print(f"  (All {locomotion_steps} distance measurements will be fed to estimator)")
        if self.include_intra_planning_bearings:
            print(f"  Include intra-planning bearings: {locomotion_steps} bearing estimates per planning step")
        
        # Current bearing state
        self.current_bearing = 0.0
        self.current_bearing_sin_cos = np.array([0.0, 1.0])  # [sin(0), cos(0)]
        
        # Distance history for estimator (if not using ground truth)
        if not self.use_ground_truth and self.bearing_estimator is not None:
            self.history_length = self.bearing_estimator.history_length
        else:
            self.history_length = 30  # Default, not used for ground truth
        
        # Visualization state
        self._enable_mujoco_overlay = False
        self._enable_mujoco_markers = False
        self._enable_chase_bearing = False
        
        # Enable visualization if configured
        if bearing_config.enable_bearing_visualization:
            self.enable_bearing_visualization(
                enable_mujoco_overlay=bearing_config.enable_mujoco_overlay,
                enable_mujoco_markers=bearing_config.enable_mujoco_markers,
                enable_chase_bearing=bearing_config.enable_chase_bearing,
            )
    
    def _setup_bearing_estimator(self) -> None:
        """Setup the bearing estimator model."""
        if self.bearing_config.bearing_estimator_log_dir is None:
            raise ValueError(
                "bearing_estimator_log_dir must be specified when use_ground_truth_bearing=False"
            )
        
        print(f"[BearingAugmentedEnv] Loading bearing estimator...")
        self.bearing_estimator = BearingEstimatorRunner(
            estimator_log_dir=self.bearing_config.bearing_estimator_log_dir,
            checkpoint=self.bearing_config.bearing_estimator_checkpoint,
            device=self.device,
            use_kalman=self.bearing_config.estimator_use_kalman,
            kalman_measurement_noise=self.bearing_config.estimator_kalman_noise,
            bearing_offset_deg=self.bearing_config.estimator_bearing_offset_deg,
        )
        print(f"  Estimator loaded successfully")
    
    def _get_ground_truth_bearing(self) -> float:
        """Calculate ground-truth bearing from robot pose and target position.
        
        Returns:
            Bearing in radians, range [-pi, pi]
            0 = target directly in front (local +Y direction)
            >0 = target to the LEFT
            <0 = target to the RIGHT
        """
        # Get robot position and heading
        robot_pos = self._base_env._get_robot_position()
        target_pos = self._base_env.target_pos
        
        # Get robot heading
        if hasattr(self._base_env, 'locomotion_env') and hasattr(self._base_env.locomotion_env, 'state'):
            heading = self._base_env.locomotion_env.state.derived.heading
            if hasattr(heading, '__len__'):
                heading = float(heading[0])
            else:
                heading = float(heading)
        else:
            heading = 0.0
        
        # Vector from robot to target in world frame
        to_target = target_pos - robot_pos[:2]
        
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        
        # Project to_target onto robot's local axes
        to_target_local_x = sin_h * to_target[0] - cos_h * to_target[1]  # LEFT
        to_target_local_y = cos_h * to_target[0] + sin_h * to_target[1]  # FORWARD
        
        bearing = np.arctan2(-to_target_local_x, to_target_local_y)
        return bearing
    
    def _get_estimated_bearing(self, distance: float, locomotion_obs: Optional[np.ndarray] = None) -> float:
        """Get bearing from the trained estimator.
        
        Args:
            distance: Current distance to target
            locomotion_obs: Single-timestep locomotion observation
            
        Returns:
            Estimated bearing in radians
        """
        if self.bearing_estimator is None:
            return 0.0
        
        # Get current controller index
        if hasattr(self._base_env, 'current_controller_idx'):
            policy_idx = self._base_env.current_controller_idx
        else:
            policy_idx = 0
        
        # Update estimator with new distance measurement
        bearing = self.bearing_estimator.update(
            distance=distance,
            locomotion_obs=locomotion_obs,
            policy_idx=policy_idx,
        )
        
        return bearing
    
    def _get_single_timestep_locomotion_obs(self) -> Optional[np.ndarray]:
        """Get single-timestep locomotion observation for estimator."""
        if not self.bearing_config.estimator_include_locomotion_obs:
            return None
        
        if hasattr(self._base_env, 'locomotion_env') and hasattr(self._base_env.locomotion_env, 'state'):
            single_obs = self._base_env.locomotion_env.state._construct_observation()
            return np.asarray(single_obs).flatten()
        return None
    
    def _update_bearing(self, info: Dict) -> None:
        """Update current bearing estimate.
        
        When locomotion_steps_per_planning > 1, this feeds ALL intra-planning
        distance measurements to the estimator, so no information is wasted.
        The final bearing returned is based on the full distance history.
        
        Args:
            info: Info dict from base environment step
        """
        if self.use_ground_truth:
            # For ground-truth, we just compute the final bearing
            self.current_bearing = self._get_ground_truth_bearing()
        else:
            # For estimator: feed ALL N intra-planning distance measurements
            # to properly update the distance history
            self._update_bearing_with_intra_planning_distances(info)
        
        # Update sin/cos representation
        self.current_bearing_sin_cos = np.array([
            np.sin(self.current_bearing),
            np.cos(self.current_bearing)
        ], dtype=np.float32)
    
    def _update_bearing_with_intra_planning_distances(self, info: Dict) -> None:
        """Update bearing estimator with all N intra-planning distance measurements.
        
        When locomotion_steps_per_planning > 1, the base environment collects
        N distance measurements during each planning step. Instead of only using
        the final distance, we feed all N distances to the estimator to build
        up a proper distance history.
        
        If the base environment also collects intra-planning locomotion observations
        (include_intra_planning_loco_obs=True), we use those for each step.
        Otherwise, we only use the final locomotion observation.
        
        If include_intra_planning_bearings is True, we also store each of the N
        bearing estimates in self.intra_planning_bearings for use in the observation.
        
        Args:
            info: Info dict from base environment step
        """
        if self.bearing_estimator is None:
            self.current_bearing = 0.0
            return
        
        # Get current controller index
        if hasattr(self._base_env, 'current_controller_idx'):
            policy_idx = self._base_env.current_controller_idx
        else:
            policy_idx = 0
        
        # Check if base env has intra-planning distance observations
        has_intra_planning = (
            hasattr(self._base_env, 'intra_planning_distances') and 
            hasattr(self._base_env, 'config') and
            getattr(self._base_env.config, 'include_intra_planning_obs', False)
        )
        
        # Check if base env has intra-planning locomotion observations
        has_intra_loco_obs = (
            hasattr(self._base_env, 'intra_planning_loco_obs') and
            self._base_env.intra_planning_loco_obs is not None and
            hasattr(self._base_env, 'config') and
            getattr(self._base_env.config, 'include_intra_planning_loco_obs', False)
        )
        
        if has_intra_planning:
            # Feed all N intra-planning distances to the estimator
            intra_distances = self._base_env.intra_planning_distances
            n_steps = len(intra_distances)
            
            # Get locomotion obs for each step if available, otherwise only last step
            if has_intra_loco_obs:
                intra_loco_obs = self._base_env.intra_planning_loco_obs
            else:
                # Fall back to only using the final locomotion observation
                final_loco_obs = self._get_single_timestep_locomotion_obs()
                intra_loco_obs = None
            
            # Update estimator with each distance measurement
            for i, dist in enumerate(intra_distances):
                if has_intra_loco_obs:
                    # Use the locomotion observation from this specific step
                    loco_obs_for_step = intra_loco_obs[i]
                else:
                    # Only pass locomotion_obs for the last step
                    loco_obs_for_step = final_loco_obs if (i == n_steps - 1) else None
                
                bearing = self.bearing_estimator.update(
                    distance=float(dist),
                    locomotion_obs=loco_obs_for_step,
                    policy_idx=policy_idx,
                )
                
                # Store intra-planning bearing if configured
                if self.include_intra_planning_bearings and self.intra_planning_bearings is not None:
                    self.intra_planning_bearings[i] = bearing
                    self.intra_planning_bearings_sin_cos[i, 0] = np.sin(bearing)
                    self.intra_planning_bearings_sin_cos[i, 1] = np.cos(bearing)
            
            # Final bearing is from the last update
            self.current_bearing = bearing
        else:
            # Fallback: single distance measurement (backwards compatible)
            distance = info.get('distance', self._base_env.last_distance)
            locomotion_obs = self._get_single_timestep_locomotion_obs()
            self.current_bearing = self._get_estimated_bearing(distance, locomotion_obs)
    
    def _augment_observation(self, obs: np.ndarray) -> np.ndarray:
        """Augment observation with bearing information.
        
        If include_intra_planning_bearings is True and we have N locomotion steps
        per planning step, we include N bearing estimates instead of just 1.
        
        Args:
            obs: Original observation from base environment
            
        Returns:
            Augmented observation with bearing appended
        """
        if self.include_intra_planning_bearings and self.intra_planning_bearings is not None:
            # Include all N intra-planning bearings
            if self.bearing_config.bearing_obs_format == "sin_cos":
                # Flatten (N, 2) to (N*2,)
                bearing_obs = self.intra_planning_bearings_sin_cos.flatten()
            else:
                bearing_obs = self.intra_planning_bearings.copy()
        else:
            # Include only the final bearing
            if self.bearing_config.bearing_obs_format == "sin_cos":
                bearing_obs = self.current_bearing_sin_cos
            else:
                bearing_obs = np.array([self.current_bearing], dtype=np.float32)
        
        return np.concatenate([obs, bearing_obs]).astype(np.float32)
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset the environment.
        
        Returns:
            observation: Augmented initial observation
            info: Additional information including bearing
        """
        obs, info = self.env.reset(**kwargs)
        
        # Reset bearing estimator if using one
        if self.bearing_estimator is not None:
            initial_distance = info.get('distance', 2.0)
            self.bearing_estimator.reset(initial_distance=initial_distance)
        
        # Update bearing (will be 0 initially for estimator, or actual for ground truth)
        self._update_bearing(info)
        
        # Augment observation
        augmented_obs = self._augment_observation(obs)
        
        # Add bearing info to info dict
        info['bearing'] = self.current_bearing
        info['bearing_deg'] = np.degrees(self.current_bearing)
        info['bearing_sin_cos'] = self.current_bearing_sin_cos.copy()
        info['bearing_source'] = 'ground_truth' if self.use_ground_truth else 'estimator'
        
        return augmented_obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step.
        
        Args:
            action: Controller index to select
            
        Returns:
            observation: Augmented observation with bearing
            reward: Reward from base environment
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional information including bearing
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update bearing after step
        self._update_bearing(info)
        
        # Augment observation
        augmented_obs = self._augment_observation(obs)
        
        # Add bearing info to info dict
        info['bearing'] = self.current_bearing
        info['bearing_deg'] = np.degrees(self.current_bearing)
        info['bearing_sin_cos'] = self.current_bearing_sin_cos.copy()
        info['bearing_source'] = 'ground_truth' if self.use_ground_truth else 'estimator'
        
        # Add estimator status if using estimator
        if self.bearing_estimator is not None:
            info['estimator_ready'] = self.bearing_estimator.is_ready()
        
        return augmented_obs, reward, terminated, truncated, info
    
    def get_bearing_info(self) -> Dict[str, Any]:
        """Get current bearing information.
        
        Returns:
            Dictionary with bearing status
        """
        info = {
            'bearing_rad': self.current_bearing,
            'bearing_deg': np.degrees(self.current_bearing),
            'bearing_sin': self.current_bearing_sin_cos[0],
            'bearing_cos': self.current_bearing_sin_cos[1],
            'source': 'ground_truth' if self.use_ground_truth else 'estimator',
        }
        
        if self.bearing_estimator is not None:
            info['estimator_status'] = self.bearing_estimator.get_status()
        
        return info
    
    # =========================================================================
    # Visualization Methods
    # =========================================================================
    
    def enable_bearing_visualization(
        self,
        enable_mujoco_overlay: bool = True,
        enable_mujoco_markers: bool = True,
        enable_chase_bearing: bool = True,
    ) -> None:
        """Enable bearing visualization in videos.
        
        This hooks into the video recording systems to add:
        - Bearing metrics overlay on MuJoCo frames
        - Chase scene markers (target, forward direction, robot-target line)
        - Bearing info in matplotlib chase video
        
        Args:
            enable_mujoco_overlay: Add bearing text overlay to MuJoCo frames
            enable_mujoco_markers: Add 3D markers in MuJoCo scene
            enable_chase_bearing: Add bearing info to matplotlib chase video
        """
        self._enable_mujoco_overlay = enable_mujoco_overlay
        self._enable_mujoco_markers = enable_mujoco_markers
        self._enable_chase_bearing = enable_chase_bearing
        
        # Hook into locomotion env frame capture
        if enable_mujoco_overlay or enable_mujoco_markers:
            self._hook_locomotion_video()
        
        # Override base env's _record_frame to include bearing
        if enable_chase_bearing:
            self._hook_chase_video()
        
        print(f"[BearingAugmentedEnv] Visualization enabled")
        print(f"  MuJoCo overlay: {enable_mujoco_overlay}")
        print(f"  MuJoCo markers: {enable_mujoco_markers}")
        print(f"  Chase bearing: {enable_chase_bearing}")
    
    def _hook_locomotion_video(self) -> None:
        """Hook into locomotion env to add bearing visualization to frames.
        
        Uses the MetaMachine callback API when available, falls back to
        monkey-patching for backwards compatibility.
        """
        if not hasattr(self._base_env, 'locomotion_env'):
            print("[Warning] Cannot hook locomotion video - no locomotion_env")
            return
        
        loco_env = self._base_env.locomotion_env
        bearing_env = self  # Store reference for closures
        
        # Try to use the new callback API first (cleaner approach)
        use_callback_api = hasattr(loco_env, 'register_pre_render_callback')
        
        if use_callback_api:
            # Use the clean callback API
            if self._enable_mujoco_markers:
                def pre_render_callback(renderer, data):
                    bearing_env._add_markers_to_renderer_scene(renderer)
                loco_env.register_pre_render_callback(pre_render_callback)
                print("  Registered pre-render callback (for 3D markers)")
            
            if self._enable_mujoco_overlay:
                def overlay_callback(frame):
                    return bearing_env._create_bearing_overlay(frame)
                loco_env.register_frame_overlay_callback(overlay_callback)
                print("  Registered frame overlay callback (for text overlay)")
        else:
            # Fall back to monkey-patching for older MetaMachine versions
            import types
            
            # Hook 1: Replace _capture_frame_egl to add 3D markers before rendering
            if self._enable_mujoco_markers and hasattr(loco_env, '_capture_frame_egl'):
                original_capture = loco_env._capture_frame_egl
                
                def capture_frame_with_markers(self_loco) -> np.ndarray:
                    """Capture frame with 3D markers added to scene."""
                    import cv2
                    
                    if self_loco.render_mode != "mp4" or not self_loco.recording_active:
                        return None
                    
                    renderer = self_loco._create_egl_renderer()
                    
                    if renderer == "synthetic":
                        width, height = self_loco.render_size
                        frame = self_loco._create_synthetic_frame(width, height)
                    else:
                        camera_id = getattr(self_loco, "preferred_camera_id", -1)
                        renderer.update_scene(self_loco.data, camera=camera_id)
                        
                        # Add 3D markers to scene AFTER update_scene but BEFORE render
                        bearing_env._add_markers_to_renderer_scene(renderer)
                        
                        pixels = renderer.render()
                        frame = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                    
                    frame = self_loco._add_metrics_overlay(frame)
                    self_loco.video_frames.append(frame)
                    return frame
                
                loco_env._capture_frame_egl = types.MethodType(capture_frame_with_markers, loco_env)
                print("  Hooked into locomotion env _capture_frame_egl (for 3D markers)")
            
            # Hook 2: Replace _add_metrics_overlay to add bearing text overlay
            if self._enable_mujoco_overlay and hasattr(loco_env, '_add_metrics_overlay'):
                original_add_metrics = loco_env._add_metrics_overlay
                
                def add_metrics_with_bearing(self_loco, frame: np.ndarray) -> np.ndarray:
                    """Enhanced metrics overlay that includes bearing information."""
                    # First call the original overlay
                    frame = original_add_metrics(frame)
                    # Then add bearing overlay
                    return bearing_env._create_bearing_overlay(frame)
                
                loco_env._add_metrics_overlay = types.MethodType(add_metrics_with_bearing, loco_env)
                print("  Hooked into locomotion env _add_metrics_overlay (for text overlay)")
    
    def _create_bearing_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Create bearing metrics overlay on a frame.
        
        This is called at each frame capture time (potentially every locomotion step).
        For accurate visualization, we compute the bearing in real-time rather than
        using the cached value which only updates per planning step.
        
        Args:
            frame: Video frame in BGR format
            
        Returns:
            Frame with bearing overlay added
        """
        try:
            from metamachine.utils.chase_visualization import add_bearing_metrics_overlay
            
            heading = self._get_robot_heading()
            step = self._base_env.step_count if hasattr(self._base_env, 'step_count') else 0
            
            controller_idx = None
            controller_name = None
            if hasattr(self._base_env, 'current_controller_idx'):
                controller_idx = self._base_env.current_controller_idx
                controller_name = self._base_env.controller_names[controller_idx]
            
            # Compute real-time bearing for visualization
            # This gives accurate per-frame bearing rather than per-planning-step
            if self.use_ground_truth:
                bearing = self._get_ground_truth_bearing()
                bearing_source = 'ground_truth'
            else:
                # For estimator: compute bearing from current state
                # Get current distance (not noisy, for visualization)
                if hasattr(self._base_env, '_get_distance_to_target'):
                    current_distance = self._base_env._get_distance_to_target()
                elif hasattr(self._base_env, 'last_distance'):
                    current_distance = self._base_env.last_distance
                else:
                    current_distance = 0.0
                
                # Get locomotion obs for this frame
                locomotion_obs = self._get_single_timestep_locomotion_obs()
                
                # Update the estimator with current data and get bearing
                # Note: This updates the estimator history, which is correct since
                # each frame capture represents a new timestep
                if self.bearing_estimator is not None:
                    policy_idx = controller_idx if controller_idx is not None else 0
                    bearing = self.bearing_estimator.update(
                        distance=float(current_distance),
                        locomotion_obs=locomotion_obs,
                        policy_idx=policy_idx,
                    )
                    # Also update cached bearing for consistency
                    self.current_bearing = bearing
                    self.current_bearing_sin_cos = np.array([
                        np.sin(bearing), np.cos(bearing)
                    ], dtype=np.float32)
                else:
                    bearing = self.current_bearing
                bearing_source = 'estimated'
            
            # Get distance for display
            distance = self._base_env.last_distance if hasattr(self._base_env, 'last_distance') else 0.0
            
            frame = add_bearing_metrics_overlay(
                frame=frame,
                bearing=bearing,
                distance=distance,
                step=step,
                heading=heading,
                controller_idx=controller_idx,
                controller_name=controller_name,
                bearing_source=bearing_source,
                position="right",  # Put on right side to not overlap with default metrics
            )
        except Exception as e:
            pass  # Silently fail if overlay doesn't work
        
        return frame
    
    def _add_markers_to_renderer_scene(self, renderer) -> None:
        """Add 3D visualization markers to renderer scene.
        
        Called between update_scene and render to add markers that will
        appear in the rendered frame.
        
        We compute the bearing in real-time here so the 3D markers reflect
        the current state, not the cached per-planning-step value.
        """
        if renderer is None or renderer == "synthetic":
            return
        
        scene = renderer.scene
        if scene is None:
            return
        
        try:
            from metamachine.utils.chase_visualization import add_chase_scene_markers
            
            robot_pos = self._base_env._get_robot_position()
            target_pos = self._base_env.target_pos
            heading = self._get_robot_heading()
            
            # Compute real-time bearing for 3D markers
            if self.use_ground_truth:
                bearing = self._get_ground_truth_bearing()
            else:
                # For estimator, use current cached bearing
                # (will be updated by overlay callback right after render)
                bearing = self.current_bearing
            
            add_chase_scene_markers(
                scene=scene,
                robot_pos=robot_pos,
                target_pos=target_pos,
                heading=heading,
                bearing=bearing,
                show_target=True,
                show_forward=True,
                show_robot_target_line=True,
                show_robot_marker=False,  # Robot already visible
            )
        except Exception as e:
            # Silently fail - don't spam errors in video recording
            pass

    def _get_robot_heading(self) -> float:
        """Get robot's current heading angle."""
        if hasattr(self._base_env, 'locomotion_env') and hasattr(self._base_env.locomotion_env, 'state'):
            heading = self._base_env.locomotion_env.state.derived.heading
            if hasattr(heading, '__len__'):
                return float(heading[0])
            return float(heading)
        return 0.0
    
    def _hook_chase_video(self) -> None:
        """Hook into base env to add bearing info to chase video frames."""
        if not hasattr(self._base_env, '_record_frame'):
            print("  [Warning] Base env has no _record_frame method")
            return
        
        # Save original _record_frame
        self._original_record_frame = self._base_env._record_frame
        
        # Replace with version that adds bearing
        def record_frame_with_bearing(controller_idx: int) -> None:
            # Call original
            self._original_record_frame(controller_idx)
            
            # Add bearing info to last frame
            if self._base_env.trajectory_frames:
                self._base_env.trajectory_frames[-1]['bearing'] = self.current_bearing
                self._base_env.trajectory_frames[-1]['bearing_source'] = (
                    'ground_truth' if self.use_ground_truth else 'estimated'
                )
        
        self._base_env._record_frame = record_frame_with_bearing
        
        # Also hook _save_chase_video to use bearing info
        self._hook_save_chase_video()
        print("  Hooked into chase video recording")
    
    def _hook_save_chase_video(self) -> None:
        """Hook _save_chase_video to display bearing info."""
        if not hasattr(self._base_env, '_save_chase_video'):
            return
        
        self._original_save_chase_video = self._base_env._save_chase_video
        
        def save_chase_video_with_bearing() -> None:
            """Save chase video with bearing info in title/annotations."""
            import os
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            import numpy as np
            
            if not self._base_env.trajectory_frames:
                return
            
            video_path = os.path.join(
                self._base_env.log_dir,
                f"policy_switch_episode_{self._base_env.episode_count}.mp4"
            )
            
            print(f"[Video] Saving policy switch video with bearing to: {video_path}")
            print(f"  Total frames: {len(self._base_env.trajectory_frames)}")
            
            try:
                import matplotlib
                matplotlib.use('Agg')
                
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Determine plot bounds
                all_x = [f['robot_pos'][0] for f in self._base_env.trajectory_frames] + \
                        [f['target_pos'][0] for f in self._base_env.trajectory_frames]
                all_y = [f['robot_pos'][1] for f in self._base_env.trajectory_frames] + \
                        [f['target_pos'][1] for f in self._base_env.trajectory_frames]
                
                margin = 1.0
                x_min, x_max = min(all_x) - margin, max(all_x) + margin
                y_min, y_max = min(all_y) - margin, max(all_y) + margin
                
                x_range = x_max - x_min
                y_range = y_max - y_min
                max_range = max(x_range, y_range)
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                
                # Colors for controllers
                colors = plt.cm.tab10(np.linspace(0, 1, self._base_env.num_controllers))
                
                def animate(frame_idx):
                    frame = self._base_env.trajectory_frames[frame_idx]
                    bearing = frame.get('bearing', 0.0)
                    bearing_source = frame.get('bearing_source', 'ground_truth')
                    
                    from metamachine.utils.chase_visualization import draw_chase_frame_matplotlib
                    
                    draw_chase_frame_matplotlib(
                        ax=ax,
                        robot_pos=np.array(frame['robot_pos']),
                        target_pos=np.array(frame['target_pos']),
                        heading=frame['robot_heading'],
                        bearing=bearing,
                        controller_idx=frame['controller_idx'],
                        controller_name=frame['controller_name'],
                        controller_names=self._base_env.controller_names,
                        controller_colors=colors,
                        step=frame['step'],
                        episode=frame['episode'],
                        distance=frame['distance'],
                        trajectory_history=self._base_env.trajectory_frames[:frame_idx+1],
                        bearing_source=bearing_source,
                        show_bearing_info=True,
                        anchor_pos=frame.get('anchor_pos'),
                        robot_anchor_distance=frame.get('robot_anchor_distance'),
                        anchor_target_distance=frame.get('anchor_target_distance'),
                        x_center=x_center,
                        y_center=y_center,
                        max_range=max_range,
                    )
                
                fps = self._base_env.config.chase_video_fps
                interval = 1000 // fps
                
                anim = animation.FuncAnimation(
                    fig, animate,
                    frames=len(self._base_env.trajectory_frames),
                    interval=interval,
                    blit=False
                )
                
                anim.save(video_path, writer='ffmpeg', fps=fps, bitrate=1800)
                print(f"  Video saved successfully!")
                
            except Exception as e:
                print(f"  Error saving video: {e}")
                import traceback
                traceback.print_exc()
            finally:
                plt.close(fig)
            
            self._base_env.trajectory_frames = []
        
        self._base_env._save_chase_video = save_chase_video_with_bearing
