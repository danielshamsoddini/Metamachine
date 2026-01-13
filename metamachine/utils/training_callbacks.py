"""
Training Callbacks for Stable-Baselines3 with Dashboard Integration.

This module provides callbacks for tracking training progress with integration
to the Rich dashboard used in real robot experiments.

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>
Licensed under the Apache License, Version 2.0
"""

from typing import List, Optional

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    BaseCallback = object  # Placeholder


class SB3TrainingProgressCallback(BaseCallback):
    """SB3-compatible callback that tracks training progress with dashboard integration.
    
    This callback:
    - Tracks episode rewards and lengths
    - Updates the RLDashboard training progress display (if available)
    - Logs progress to a CSV file
    - Logs periodic updates to the dashboard log panel
    
    Usage:
        ```python
        from metamachine.utils.training_callbacks import SB3TrainingProgressCallback
        
        callback = SB3TrainingProgressCallback(
            log_file="logs/training_progress.csv",
            log_interval=5,
            total_timesteps=100000,
        )
        
        model.learn(total_timesteps=100000, callback=callback)
        ```
    """
    
    def __init__(
        self,
        log_file: str = None,
        log_interval: int = 10,
        total_timesteps: int = 100000,
        verbose: int = 0
    ):
        """Initialize the training progress callback.
        
        Args:
            log_file: Path to CSV file for logging progress
            log_interval: Log to dashboard every N episodes
            total_timesteps: Total timesteps for progress bar calculation
            verbose: Verbosity level
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for this callback")
        
        super().__init__(verbose)
        self.log_file = log_file
        self.log_interval = log_interval
        self.total_timesteps = total_timesteps
        
        # Episode tracking
        self.episode_count = 0
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # Dashboard reference
        self._dashboard = None
        self._dashboard_initialized = False
        
        # File logging
        if self.log_file:
            self.file_handle = open(self.log_file, 'w')
            self.file_handle.write("episode,timesteps,reward,length,avg_reward_10\n")
            self.file_handle.flush()
        else:
            self.file_handle = None
    
    def _init_dashboard(self):
        """Initialize dashboard training progress display."""
        if self._dashboard_initialized:
            return
        
        try:
            dashboard = self._get_dashboard()
            if dashboard is not None:
                if hasattr(dashboard, 'enable_training_progress'):
                    dashboard.enable_training_progress(self.total_timesteps)
                    self._dashboard = dashboard
                    self._dashboard_initialized = True
                    if self.verbose > 0:
                        print(f"[Training] Dashboard progress enabled: {self.total_timesteps:,} timesteps")
                elif self.verbose > 0:
                    print(f"[Training] Dashboard found but no enable_training_progress method")
            elif self.verbose > 0:
                print("[Training] Dashboard not found in environment chain")
        except Exception as e:
            if self.verbose > 0:
                print(f"[Training] Error initializing dashboard: {e}")
    
    def _get_dashboard(self):
        """Get the dashboard from the environment.
        
        Navigates through SB3's VecEnv and Gym wrappers to find the dashboard.
        """
        try:
            env = self.training_env
            
            # Navigate through VecEnv wrapper (DummyVecEnv, SubprocVecEnv, etc.)
            if hasattr(env, 'envs') and len(env.envs) > 0:
                env = env.envs[0]
            
            # Navigate through Monitor and other Gym wrappers
            while hasattr(env, 'env'):
                env = env.env
            
            # Check for real_env attribute (custom env wrapping RealMetaMachine)
            if hasattr(env, 'real_env'):
                real_env = env.real_env
                if hasattr(real_env, 'dashboard') and real_env.dashboard is not None:
                    return real_env.dashboard
            
            # Maybe the env itself has a dashboard
            if hasattr(env, 'dashboard') and env.dashboard is not None:
                return env.dashboard
                
        except Exception as e:
            if self.verbose > 0:
                print(f"[Training] Error getting dashboard: {e}")
        return None
    
    def _on_training_start(self) -> None:
        """Called at the start of training."""
        if self.verbose > 0:
            print(f"[Training] _on_training_start called, training_env type: {type(self.training_env).__name__}")
        self._init_dashboard()
    
    def _on_step(self) -> bool:
        """Called at each step."""
        # Initialize dashboard if not done yet
        if not self._dashboard_initialized:
            self._init_dashboard()
        
        # Update dashboard progress periodically (every 100 steps to reduce overhead)
        if self._dashboard is not None and self.num_timesteps % 100 == 0:
            try:
                avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0
                self._dashboard.update_training_progress(
                    timesteps=self.num_timesteps,
                    episodes=self.episode_count,
                    avg_reward=avg_reward,
                )
            except Exception:
                pass
        
        # Track episode progress
        self.current_episode_length += 1
        
        # Check if episode ended
        dones = self.locals.get('dones', self.locals.get('done', [False]))
        infos = self.locals.get('infos', self.locals.get('info', [{}]))
        
        if isinstance(dones, bool):
            dones = [dones]
        if isinstance(infos, dict):
            infos = [infos]
        
        for i, done in enumerate(dones):
            if done:
                self._on_episode_end(infos[i] if i < len(infos) else {})
        
        return True
    
    def _on_episode_end(self, info: dict):
        """Handle episode completion."""
        self.episode_count += 1
        
        # Get episode reward from info (Monitor wrapper adds this)
        ep_reward = info.get('episode', {}).get('r', self.current_episode_reward)
        ep_length = info.get('episode', {}).get('l', self.current_episode_length)
        
        self.episode_rewards.append(ep_reward)
        self.episode_lengths.append(ep_length)
        
        # Calculate moving average
        recent = self.episode_rewards[-10:]
        avg_reward = np.mean(recent)
        
        # Write to log file
        if self.file_handle:
            self.file_handle.write(
                f"{self.episode_count},{self.num_timesteps},{ep_reward:.2f},{ep_length},{avg_reward:.2f}\n"
            )
            self.file_handle.flush()
        
        # Update dashboard training progress with episode info
        if self._dashboard is not None:
            try:
                self._dashboard.update_training_progress(
                    timesteps=self.num_timesteps,
                    episodes=self.episode_count,
                    episode_reward=ep_reward,
                    avg_reward=avg_reward,
                )
            except Exception:
                pass
        
        # Log to dashboard (periodic)
        if self.episode_count % self.log_interval == 0:
            self._log_to_dashboard(
                f"📊 Ep {self.episode_count}: R={ep_reward:.1f} (avg10={avg_reward:.1f}) L={ep_length}"
            )
        
        # Reset counters
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def _log_to_dashboard(self, msg: str, level: str = "success"):
        """Log message to the dashboard."""
        try:
            env = self.training_env
            while hasattr(env, 'envs'):
                env = env.envs[0]
            while hasattr(env, 'env'):
                env = env.env
            if hasattr(env, 'real_env') and hasattr(env.real_env, '_dashboard_log'):
                env.real_env._dashboard_log(msg, level)
        except Exception:
            pass  # Dashboard logging is best-effort
    
    def _on_training_end(self):
        """Called at the end of training."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
        
        # Disable dashboard training progress
        if self._dashboard is not None:
            try:
                self._dashboard.disable_training_progress()
            except Exception:
                pass
        
        if self.episode_rewards:
            summary = (
                f"Training complete: {self.episode_count} episodes, "
                f"Avg reward: {np.mean(self.episode_rewards):.2f}"
            )
            self._log_to_dashboard(f"✅ {summary}")
    
    def get_summary(self) -> str:
        """Get a summary of training progress."""
        if not self.episode_rewards:
            return "No episodes completed"
        
        return (
            f"Episodes: {self.episode_count}, "
            f"Avg Reward: {np.mean(self.episode_rewards):.2f}, "
            f"Avg Length: {np.mean(self.episode_lengths):.1f}, "
            f"Last 10 Avg: {np.mean(self.episode_rewards[-10:]):.2f}"
        )

