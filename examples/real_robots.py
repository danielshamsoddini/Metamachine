#!/usr/bin/env python3
"""
Real Robot Control Example

This script demonstrates how to control real robots using the RealMetaMachine
environment with capybarish for ESP32 communication.

Features:
- Loads config with expected module_ids for ordered action mapping
- Auto-discovers and validates ESP32 modules (both active and sensor-only)
- Supports passive sensor modules (e.g., dedicated distance sensors)
- Configurable global state sources (main IMU, goal distance)
- Optional Rich dashboard for real-time monitoring
- Optional MuJoCo viewer to visualize robot orientation in real-time
- Multi-model support for A/B testing and comparison
- Keyboard controls: e=enable, d=disable, r=restart, c=calibrate, q=quit
- Balance mode (--balance): PID control to hold target theta
- Calibration mode (--calibration): record sensor sweeps and balance-point data to JSON

Usage:
    # Basic usage with default config
    python examples/real_robots.py
    
    # With custom config
    python examples/real_robots.py --config path/to/config.yaml
    
    # With sinusoidal test motion
    python examples/real_robots.py --test-motion
    
    # With MuJoCo viewer (shows robot orientation from IMU)
    python examples/real_robots.py --viewer
    python examples/real_robots.py --test-motion --viewer
    python examples/real_robots.py --log-dir logs/experiment --viewer
    
    # Load and run a trained policy
    python examples/real_robots.py --policy path/to/policy.pt
    
    # Load from training log with custom module IDs
    python examples/real_robots.py --log-dir logs/experiment --module-ids 5 21 16
    
    # Load MULTIPLE models for comparison (switch with , and . keys)
    python examples/real_robots.py -L logs/baseline logs/improved \\
        --module-ids 5 21 16

Before running:
    1. Configure ESP32 modules with their module_ids (0, 1, 2, ...)
    2. Configure ESP32 to send data to this computer's IP on port 6666
    3. Ensure the module_ids in config match your physical modules

Keyboard Controls:
    Motor:    e=enable, d=disable, r=restart, c=calibrate, q=quit
    Commands: 0-9=select/set, []=prev/next, +/-=adjust
              R=resample, k=toggle keyboard mode, i=info
    Models:   , (comma)=prev model, . (period)=next model, /=list models

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>
Licensed under the Apache License, Version 2.0
"""

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Default config path
DEFAULT_CONFIG = str(
    PROJECT_ROOT / "metamachine" / "environments" / "configs" / 
    "default_configs" / "wheel.yaml"
)


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    # Restore terminal cursor immediately
    sys.stdout.write('\033[?25h')  # Show cursor
    sys.stdout.write('\033[0m')    # Reset attributes
    sys.stdout.flush()
    try:
        with open('/dev/tty', 'w') as tty:
            tty.write('\033[?25h\033[0m')
            tty.flush()
    except Exception:
        pass
    print("\n[Signal] Received interrupt. Shutting down...")
    sys.exit(0)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real Robot Control with RealMetaMachine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default config
    python real_robots.py
    
    # Run with custom config
    python real_robots.py --config my_robot.yaml
    
    # Run sinusoidal test motion
    python real_robots.py --test-motion --amplitude 0.5 --frequency 0.3
    
    # Run with MuJoCo viewer (visualize robot orientation)
    python real_robots.py --viewer
    python real_robots.py --test-motion --viewer
    
    # Run with trained policy
    python real_robots.py --policy logs/experiment/policy.pt
    
    # Load from training log with custom module IDs
    python real_robots.py --log-dir logs/20251228_223556l_lego_tripod --module-ids 5 21 16
    
    # Load from training log with viewer
    python real_robots.py --log-dir logs/experiment --module-ids 5 21 16 --viewer
    
    # Load from training log with sensor modules
    python real_robots.py --log-dir logs/experiment --module-ids 5 21 16 --sensor-module-ids 100
    
    # Run balance mode (PID control at target theta; default from calibration)
    python real_robots.py --balance
    python real_robots.py --balance -b --target-theta -118 --kp 0.05 --duration 60
    
    # Run calibration mode (record sweeps and balance-point data)
    python real_robots.py --calibration
    python real_robots.py --calibration --calibration-output my_cal.json
    
    # Load MULTIPLE models for comparison (switch with , and . keys)
    python real_robots.py -L logs/baseline logs/improved logs/experimental \\
        --module-ids 5 21 16
    
    # Load multiple models with custom display names
    python real_robots.py -L logs/run1 logs/run2 logs/run3 \\
        --model-names "Baseline" "NewReward" "Latest" \\
        --module-ids 5 21 16
    
Keyboard Controls (during operation):
    Motor:    e=enable, d=disable, r=restart, c=calibrate, q=quit
    Commands: 0-9=select/set, []=prev/next, +/-=adjust
              R=resample, k=toggle keyboard mode, i=info
    Models:   , (comma)=prev model, . (period)=next model, /=list models
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to robot configuration YAML file"
    )
    
    parser.add_argument(
        "--test-motion",
        action="store_true",
        help="Run sinusoidal test motion (no policy required)"
    )
    
    parser.add_argument(
        "--amplitude", "-a",
        type=float,
        default=0.5,
        help="Sinusoidal motion amplitude (radians)"
    )
    
    parser.add_argument(
        "--frequency", "-f",
        type=float,
        default=0.3,
        help="Sinusoidal motion frequency (Hz)"
    )
    
    parser.add_argument(
        "--policy", "-p",
        type=str,
        default=None,
        help="Path to trained policy checkpoint"
    )
    
    parser.add_argument(
        "--log-dir", "-l",
        type=str,
        default=None,
        help="Path to training log directory (loads config and checkpoint from there)"
    )
    
    parser.add_argument(
        "--log-dirs", "-L",
        type=str,
        nargs="+",
        default=None,
        help="Multiple log directories to load policies from (switch with ,./)"
    )
    
    parser.add_argument(
        "--model-names",
        type=str,
        nargs="+",
        default=None,
        help="Display names for each model (default: directory names)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Checkpoint to load from log-dir: 'latest', 'final', 'best', or step number"
    )
    
    parser.add_argument(
        "--module-ids",
        type=int,
        nargs="+",
        default=None,
        help="Module IDs for real robot (e.g., --module-ids 5 21 16). Overrides config."
    )
    
    parser.add_argument(
        "--sensor-module-ids",
        type=int,
        nargs="+",
        default=None,
        help="Sensor-only module IDs (e.g., --sensor-module-ids 100). Overrides config."
    )
    
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable Rich dashboard"
    )
    
    parser.add_argument(
        "--duration", "-t",
        type=float,
        default=None,
        help="Run duration in seconds (None = run until quit)"
    )
    
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Enable MuJoCo viewer to visualize robot orientation in real-time"
    )
    
    parser.add_argument(
        "--viewer-config",
        type=str,
        default=None,
        help="Config for viewer simulation (uses real robot config if not specified)"
    )
    
    # Balance mode (PID control to hold target theta)
    parser.add_argument(
        "--balance", "-b",
        action="store_true",
        help="Run balance mode: PID control to maintain target theta (use with --target-theta)"
    )
    parser.add_argument(
        "--target-theta",
        type=float,
        default=-117.57,
        help="Target theta in degrees for balance mode (default: -117.57, from calibration)"
    )
    parser.add_argument(
        "--kp", type=float, default=0.035,
        help="PID proportional gain for balance mode (lower = less overshoot/oscillation)"
    )
    parser.add_argument(
        "--ki", type=float, default=0.005, help="PID integral gain for balance mode"
    )
    parser.add_argument(
        "--kd", type=float, default=0.01,
        help="PID derivative gain for balance mode (higher = more damping, less oscillation)"
    )
    parser.add_argument(
        "--error-scale",
        type=float,
        default=15.0,
        help="Tanh error scale in degrees for balance mode (linear region width)"
    )
    parser.add_argument(
        "--near-balance-scale",
        type=float,
        default=15.0,
        help="Degrees: scale down action when |error| < this (smaller action near balance)"
    )
    parser.add_argument(
        "--near-balance-min-gain",
        type=float,
        default=0.25,
        help="Minimum gain near balance (0–1); avoids zero action at exact balance"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=2.5,
        help="Max change in PID output per second (lower = smoother, less oscillation)"
    )
    parser.add_argument(
        "--output-limit",
        type=float,
        default=0.5,
        help="Max PID output magnitude (lower = gentler response)"
    )
    
    # Outer position-hold loop
    parser.add_argument(
        "--no-position-hold",
        action="store_true",
        help="Disable outer position-hold loop (only inner balance PID)"
    )
    parser.add_argument(
        "--pos-kp", type=float, default=0.3,
        help="Outer loop P gain: degrees of theta offset per radian of wheel displacement"
    )
    parser.add_argument(
        "--pos-kd", type=float, default=0.15,
        help="Outer loop D gain: damping on wheel velocity"
    )
    parser.add_argument(
        "--pos-max-theta", type=float, default=4.0,
        help="Max theta offset (degrees) the outer position loop can apply"
    )
    
    # Calibration mode (record sweeps and balance-point snapshots to JSON)
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Run calibration mode: record sensor sweeps and balance-point data to JSON"
    )
    parser.add_argument(
        "--calibration-output",
        type=str,
        default=None,
        help="Output JSON path for calibration data (default: calibration_YYYYMMDD_HHMMSS.json)"
    )
    
    return parser.parse_args()


def load_config(config_path: str):
    """Load configuration from YAML file or config name."""
    from metamachine.environments.configs.config_registry import ConfigRegistry
    
    if os.path.exists(config_path):
        cfg = ConfigRegistry.create_from_file(config_path)
    else:
        cfg = ConfigRegistry.create_from_name(config_path)
    
    return cfg


def create_environment(cfg):
    """Create the real robot environment."""
    from metamachine.environments.env_real import RealMetaMachine
    
    # Ensure mode is set to real
    if cfg.environment.get("mode", "sim") != "real":
        print("[Warning] Config mode is not 'real'. Forcing mode=real.")
        cfg.environment.mode = "real"
    
    env = RealMetaMachine(cfg)
    return env


def load_policy(policy_path: str, env):
    """Load a trained policy checkpoint."""
    import torch
    
    if not os.path.exists(policy_path):
        print(f"Error: Policy file not found: {policy_path}")
        return None
    
    print(f"Loading policy from: {policy_path}")
    
    try:
        # Try loading as a CrossQ/JAX policy
        checkpoint = torch.load(policy_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "actor" in checkpoint:
            actor = checkpoint["actor"]
        elif "policy" in checkpoint:
            actor = checkpoint["policy"]
        else:
            print("[Warning] Unknown checkpoint format. Keys:", list(checkpoint.keys()))
            return None
        
        print("Policy loaded successfully!")
        return actor
        
    except Exception as e:
        print(f"Error loading policy: {e}")
        return None


# =============================================================================
# Run Functions
# =============================================================================

def run_sinusoidal_test(env, amplitude: float, frequency: float, duration: float = None):
    """Run sinusoidal test motion.
    
    Args:
        env: RealMetaMachine environment
        amplitude: Motion amplitude in radians
        frequency: Motion frequency in Hz
        duration: Duration in seconds (None = run until interrupt)
    """
    print("\n" + "=" * 60)
    print("Sinusoidal Test Motion")
    print("=" * 60)
    print(f"  Amplitude: {amplitude} rad")
    print(f"  Frequency: {frequency} Hz")
    print(f"  Duration: {'infinite' if duration is None else f'{duration}s'}")
    print("=" * 60)
    print("\nPress 'e' to enable motors, 'q' to quit")
    
    # Reset environment
    obs, info = env.reset()
    
    start_time = time.time()
    step_count = 0
    
    try:
        while True:
            elapsed = time.time() - start_time
            
            # Check duration
            if duration is not None and elapsed >= duration:
                print(f"\n[Done] Reached duration limit ({duration}s)")
                break
            
            # Generate sinusoidal action
            num_actions = env.action_space.shape[0]
            phase = 2 * np.pi * frequency * elapsed
            
            # Create action with phase offset per motor for gait-like motion
            action = np.zeros(num_actions)
            for i in range(num_actions):
                phase_offset = (2 * np.pi * i) / num_actions
                action[i] = amplitude * np.sin(phase + phase_offset)
            
            # Execute step
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            
            # Print status periodically
            if step_count % 100 == 0:
                print(f"\r[Step {step_count}] Time: {elapsed:.1f}s, "
                      f"Reward: {reward:.4f}", end="", flush=True)
            
            # Check for episode end
            if done or truncated:
                print(f"\n[Episode ended] Done={done}, Truncated={truncated}")
                obs, info = env.reset()
    
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    
    finally:
        elapsed = time.time() - start_time
        print(f"\n\nTest completed: {step_count} steps in {elapsed:.1f}s")


def run_policy(env, policy, duration: float = None):
    """Run a trained policy (PyTorch model).
    
    Args:
        env: RealMetaMachine environment
        policy: Loaded policy model (PyTorch)
        duration: Duration in seconds (None = run until interrupt)
    """
    import torch
    
    print("\n" + "=" * 60)
    print("Running Trained Policy")
    print("=" * 60)
    print(f"  Duration: {'infinite' if duration is None else f'{duration}s'}")
    print("=" * 60)
    print("\nPress 'e' to enable motors, 'q' to quit")
    
    # Reset environment
    obs, info = env.reset()
    
    start_time = time.time()
    step_count = 0
    episode_reward = 0
    
    try:
        while True:
            elapsed = time.time() - start_time
            
            # Check duration
            if duration is not None and elapsed >= duration:
                print(f"\n[Done] Reached duration limit ({duration}s)")
                break
            
            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action = policy(obs_tensor)
                if hasattr(action, 'numpy'):
                    action = action.numpy()
                action = action.squeeze()
            
            # Execute step
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            episode_reward += reward
            
            # Print status periodically
            if step_count % 100 == 0:
                print(f"\r[Step {step_count}] Time: {elapsed:.1f}s, "
                      f"Episode Reward: {episode_reward:.2f}", end="", flush=True)
            
            # Check for episode end
            if done or truncated:
                print(f"\n[Episode ended] Reward: {episode_reward:.2f}")
                obs, info = env.reset()
                episode_reward = 0
    
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    
    finally:
        elapsed = time.time() - start_time
        print(f"\n\nPolicy run completed: {step_count} steps in {elapsed:.1f}s")


def run_sb3_policy(env, model, duration: float = None, deterministic: bool = True):
    """Run a trained SB3 policy.
    
    Args:
        env: RealMetaMachine environment
        model: Loaded SB3 model (CrossQ, SAC, PPO, etc.)
        duration: Duration in seconds (None = run until interrupt)
        deterministic: Use deterministic actions (no exploration noise)
    """
    print("\n" + "=" * 60)
    print("Running SB3 Policy")
    print("=" * 60)
    print(f"  Duration: {'infinite' if duration is None else f'{duration}s'}")
    print(f"  Deterministic: {deterministic}")
    print("=" * 60)
    print("\nPress 'e' to enable motors, 'q' to quit")
    
    # Reset environment
    obs, info = env.reset()
    
    start_time = time.time()
    step_count = 0
    episode_reward = 0
    episode_count = 0
    
    try:
        while True:
            elapsed = time.time() - start_time
            
            # Check duration
            if duration is not None and elapsed >= duration:
                print(f"\n[Done] Reached duration limit ({duration}s)")
                break
            
            # Get action from SB3 model
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Execute step
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            episode_reward += reward
            
            # Print status periodically
            if step_count % 100 == 0:
                print(f"\r[Step {step_count}] Time: {elapsed:.1f}s, "
                      f"Episode Reward: {episode_reward:.2f}", end="", flush=True)
            
            # Check for episode end
            if done or truncated:
                episode_count += 1
                print(f"\n[Episode {episode_count}] Reward: {episode_reward:.2f}")
                obs, info = env.reset()
                episode_reward = 0
    
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    
    finally:
        elapsed = time.time() - start_time
        print(f"\n\nPolicy run completed: {step_count} steps, "
              f"{episode_count} episodes in {elapsed:.1f}s")


def run_sb3_policy_multi(env, runner, duration: float = None, deterministic: bool = True):
    """Run multiple SB3 policies with runtime switching.
    
    Args:
        env: RealMetaMachine environment
        runner: MultiModelRunner with loaded models
        duration: Duration in seconds (None = run until interrupt)
        deterministic: Use deterministic actions (no exploration noise)
    """
    print("\n" + "=" * 60)
    print("Running Multi-Model Policy")
    print("=" * 60)
    print(f"  Models loaded: {runner.num_models}")
    print(f"  Current model: {runner.get_status_string()}")
    print(f"  Duration: {'infinite' if duration is None else f'{duration}s'}")
    print(f"  Deterministic: {deterministic}")
    print("=" * 60)
    print("\nModel switching:")
    print("  , (comma)  - Previous model")
    print("  . (period) - Next model")
    print("  / (slash)  - Show model list")
    print("\nMotor controls: e=enable, d=disable, q=quit")
    
    # Register model runner with environment for keyboard handling
    env.model_runner = runner
    
    # Update dashboard with initial model info
    if hasattr(env, '_update_dashboard_model'):
        env._update_dashboard_model()
    
    # Log model list to dashboard
    if hasattr(env, '_dashboard_log'):
        env._dashboard_log(f"Loaded {runner.num_models} models. Use ,/./slash to switch.", "info")
        for i, (name, obs_dim) in enumerate(zip(runner.model_names, runner.obs_dims)):
            marker = "►" if i == runner.current_idx else " "
            env._dashboard_log(f"  {marker}[{i+1}] {name} (obs={obs_dim})", "info")
    
    # Reset environment
    obs, info = env.reset()
    
    start_time = time.time()
    step_count = 0
    episode_reward = 0
    episode_count = 0
    
    try:
        while True:
            elapsed = time.time() - start_time
            
            # Check duration
            if duration is not None and elapsed >= duration:
                print(f"\n[Done] Reached duration limit ({duration}s)")
                break
            
            # Get action from current model (with automatic observation adaptation)
            action, _ = runner.predict(obs, deterministic=deterministic)
            
            # Execute step
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            episode_reward += reward
            
            # Print status periodically
            if step_count % 100 == 0:
                model_status = runner.get_status_string()
                print(f"\r[Step {step_count}] Model: {model_status}, "
                      f"Reward: {episode_reward:.2f}", end="", flush=True)
            
            # Check for episode end
            if done or truncated:
                episode_count += 1
                print(f"\n[Episode {episode_count}] Model: {runner.current_name}, "
                      f"Reward: {episode_reward:.2f}")
                obs, info = env.reset()
                episode_reward = 0
    
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    
    finally:
        # Clean up
        env.model_runner = None
        elapsed = time.time() - start_time
        print(f"\n\nMulti-model run completed: {step_count} steps, "
              f"{episode_count} episodes in {elapsed:.1f}s")


def run_idle(env, duration: float = None):
    """Run in idle mode - just monitor modules without motion.
    
    Useful for testing connectivity and calibration.
    """
    print("\n" + "=" * 60)
    print("Idle Mode - Monitoring Only")
    print("=" * 60)
    print("  Motors will not move automatically")
    print("  Use this mode for connectivity testing and calibration")
    print("=" * 60)
    print("\nControls:")
    print("  e - Enable motors")
    print("  d - Disable motors")
    print("  r - Restart motors")
    print("  c - Calibrate motors")
    print("  s - Print status")
    print("  q - Quit")
    
    # Reset environment
    obs, info = env.reset()
    
    start_time = time.time()
    step_count = 0
    
    try:
        while True:
            elapsed = time.time() - start_time
            
            if duration is not None and elapsed >= duration:
                print(f"\n[Done] Reached duration limit ({duration}s)")
                break
            
            # Send zero action to maintain communication
            num_actions = env.action_space.shape[0]
            action = np.zeros(num_actions)
            
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            
            # Handle special keyboard input for status
            if hasattr(env, 'input_key') and env.input_key == 's':
                env.print_status()
                env.input_key = ""
            
            if done or truncated:
                obs, info = env.reset()
    
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    
    finally:
        elapsed = time.time() - start_time
        print(f"\nIdle mode ended: {step_count} steps in {elapsed:.1f}s")


# =============================================================================
# Viewer-Enhanced Run Functions
# =============================================================================

def run_sinusoidal_test_with_viewer(env, amplitude: float, frequency: float, 
                                     duration: float = None, viewer_config=None):
    """Run sinusoidal test motion with viewer."""
    from metamachine.utils.viewer_utils import run_with_viewer
    
    num_actions = env.action_space.shape[0]
    start_time = time.time()
    
    def action_fn(obs, step_count):
        elapsed = time.time() - start_time
        phase = 2 * np.pi * frequency * elapsed
        
        # Create action with phase offset per motor for gait-like motion
        action = np.zeros(num_actions)
        for i in range(num_actions):
            phase_offset = (2 * np.pi * i) / num_actions
            action[i] = amplitude * np.sin(phase + phase_offset)
        
        return action
    
    print(f"Sinusoidal Test: amplitude={amplitude}, frequency={frequency}")
    run_with_viewer(env, action_fn, duration, viewer_config)


def run_policy_with_viewer(env, policy, duration: float = None, viewer_config=None):
    """Run PyTorch policy with viewer."""
    import torch
    from metamachine.utils.viewer_utils import run_with_viewer
    
    def action_fn(obs, step_count):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action = policy(obs_tensor)
            if hasattr(action, 'numpy'):
                action = action.numpy()
            action = action.squeeze()
        return action
    
    print("Running trained policy with viewer")
    run_with_viewer(env, action_fn, duration, viewer_config)


def run_sb3_policy_with_viewer(env, model, duration: float = None, 
                                deterministic: bool = True, viewer_config=None):
    """Run SB3 policy with viewer."""
    from metamachine.utils.viewer_utils import run_with_viewer
    
    def action_fn(obs, step_count):
        action, _ = model.predict(obs, deterministic=deterministic)
        return action
    
    print(f"Running SB3 policy with viewer (deterministic={deterministic})")
    run_with_viewer(env, action_fn, duration, viewer_config)


def run_sb3_policy_multi_with_viewer(env, runner, duration: float = None,
                                      deterministic: bool = True, viewer_config=None):
    """Run multiple SB3 policies with viewer and runtime switching."""
    from metamachine.utils.viewer_utils import run_with_viewer
    
    # Register model runner with environment for keyboard handling
    env.model_runner = runner
    
    # Update dashboard with initial model info
    if hasattr(env, '_update_dashboard_model'):
        env._update_dashboard_model()
    
    def action_fn(obs, step_count):
        action, _ = runner.predict(obs, deterministic=deterministic)
        return action
    
    print(f"Running multi-model policy with viewer (models={runner.num_models})")
    try:
        run_with_viewer(env, action_fn, duration, viewer_config)
    finally:
        env.model_runner = None


def run_idle_with_viewer(env, duration: float = None, viewer_config=None):
    """Run in idle mode with viewer (no automatic motion)."""
    from metamachine.utils.viewer_utils import run_with_viewer
    
    num_actions = env.action_space.shape[0]
    
    def action_fn(obs, step_count):
        return np.zeros(num_actions)
    
    print("Idle mode with viewer - monitoring orientation only")
    run_with_viewer(env, action_fn, duration, viewer_config)


def _snapshot_obs(env, obs_vector=None):
    """Capture current sensor/state snapshot from env as JSON-serializable dict.
    
    Caller should call env.step(np.zeros(n)) before this to refresh state.
    """
    out = {}
    # Observable data (raw sensors)
    if hasattr(env, 'observable_data') and env.observable_data:
        for key in ["dof_pos", "dof_vel", "quat", "ang_vel_body", "projected_gravity"]:
            if key in env.observable_data and env.observable_data[key] is not None:
                arr = np.asarray(env.observable_data[key]).flatten()
                out[key] = [float(x) for x in arr]
    # Derived projected_gravity / theta from state
    if hasattr(env, 'state') and hasattr(env.state, 'derived'):
        if hasattr(env.state.derived, 'projected_gravity'):
            pg = env.state.derived.projected_gravity
            if pg is not None:
                pg_arr = np.asarray(pg).flatten()
                out["projected_gravity"] = [float(x) for x in pg_arr]
                if len(pg_arr) >= 2:
                    theta_rad = np.arctan2(float(pg_arr[1]), float(pg_arr[0]))
                    out["theta_deg"] = float(np.degrees(theta_rad))
    # Full observation vector if available
    if obs_vector is not None:
        out["obs"] = [float(x) for x in np.asarray(obs_vector).flatten()]
    elif hasattr(env, 'state') and hasattr(env.state, 'get_observation'):
        try:
            o = env.state.get_observation(insert=False)
            if o is not None:
                out["obs"] = [float(x) for x in np.asarray(o).flatten()]
        except Exception:
            pass
    return out


class PIDController:
    """Simple PID controller with output clamp, derivative limit, and rate limiting."""
    
    def __init__(self, kp: float = 0.04, ki: float = 0.005, kd: float = 0.03,
                 output_limit: float = 0.6, derivative_limit: float = 50.0,
                 rate_limit: float = 3.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.derivative_limit = derivative_limit
        self.rate_limit = rate_limit
        self.reset()
    
    def reset(self):
        self.integral = 0.0
        self.last_error = None
        self.last_time = None
        self.last_output = 0.0
    
    def update(self, error: float, current_time: float, external_derivative: float = None):
        """Update PID. error is scaled (e.g. tanh-scaled). external_derivative is d(theta)/dt in deg/s for D-term."""
        dt = 0.0
        if self.last_time is not None:
            dt = current_time - self.last_time
        if dt <= 0:
            dt = 1e-6
        
        self.integral += error * dt
        # Anti-windup: clamp integral contribution
        max_i = self.output_limit / max(self.ki, 1e-9)
        self.integral = np.clip(self.integral, -max_i, max_i)
        
        if external_derivative is not None:
            d_term = -self.kd * external_derivative
        else:
            d_term = 0.0
            if self.last_error is not None:
                d_term = self.kd * (error - self.last_error) / dt
            d_term = np.clip(d_term, -self.derivative_limit, self.derivative_limit)
        
        self.last_error = error
        self.last_time = current_time
        
        out = self.kp * error + self.ki * self.integral + d_term
        out = np.clip(out, -self.output_limit, self.output_limit)
        
        # Rate limit
        delta = out - self.last_output
        max_delta = self.rate_limit * dt
        delta = np.clip(delta, -max_delta, max_delta)
        out = self.last_output + delta
        self.last_output = out
        return out


def run_calibration(env, output_path: str = None):
    """Run calibration mode to record sensor data for later analysis.
    
    Two phases:
      Phase 1 (Sweep × 3): Slowly tilt the robot from one side → balance → other side.
                            Records all obs continuously during each sweep.
      Phase 2 (Balance-point × 3): Place robot at the balance position, press Enter
                                    to capture a snapshot. Repeat 3 times.
    
    All data is saved to a JSON file.
    
    Args:
        env: RealMetaMachine environment
        output_path: Output JSON file path (auto-generated if None)
    """
    if output_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"calibration_{ts}.json"
    
    print("\n" + "=" * 60)
    print("  CALIBRATION MODE")
    print("=" * 60)
    print()
    print("Motors are auto-disabled during calibration (no motion).")
    print()
    print("This will record sensor readings in two phases:")
    print()
    print("  Phase 1 — Full sweep (×3)")
    print("    Slowly tilt the robot: lying flat → upright → flat on other side")
    print("    Press ENTER to start each sweep, press ENTER again to stop.")
    print()
    print("  Phase 2 — Balance-point capture (×3)")
    print("    Hold the robot at the balance position and press ENTER to capture.")
    print()
    print(f"  Output file: {output_path}")
    print("=" * 60)
    
    # Calibration mode: skip "motors ON" requirement so we don't need to press 'e', and force motors disabled
    if hasattr(env, 'calibration_mode'):
        env.calibration_mode = True
    if hasattr(env, '_disable_motor'):
        env._disable_motor()
    
    # Reset environment (will proceed once modules are connected; motors stay disabled)
    obs, info = env.reset()
    
    # Ensure motors stay disabled after reset and for the rest of calibration
    if hasattr(env, '_disable_motor'):
        env._disable_motor()
    num_actions = env.action_space.shape[0]
    for _ in range(10):
        env.step(np.zeros(num_actions))
    
    calibration_data = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sweeps": [],
        "balance_points": [],
    }
    
    # ─────────────────────────────────────────────
    # Phase 1: Full sweeps
    # ─────────────────────────────────────────────
    NUM_SWEEPS = 3
    print(f"\n{'─' * 60}")
    print("  PHASE 1: Full Sweeps")
    print(f"{'─' * 60}")
    
    for sweep_idx in range(NUM_SWEEPS):
        print(f"\n  Sweep {sweep_idx + 1}/{NUM_SWEEPS}")
        print("  Position robot lying flat on one side.")
        input("  Press ENTER to START recording (then slowly tilt through balance to the other side)... ")
        
        print("  Recording... (press ENTER to STOP)")
        
        sweep_samples = []
        stop_event = threading.Event()
        start_t = time.time()
        sample_count = 0
        
        # Background thread waits for Enter key to signal stop
        def _wait_for_enter():
            sys.stdin.readline()
            stop_event.set()
        
        input_thread = threading.Thread(target=_wait_for_enter, daemon=True)
        input_thread.start()
        
        while not stop_event.is_set():
            # Step env to get fresh sensor data, then capture snapshot
            obs, _, _, _, _ = env.step(np.zeros(num_actions))
            snap = _snapshot_obs(env, obs_vector=obs)
            snap["sweep_elapsed"] = time.time() - start_t
            sweep_samples.append(snap)
            sample_count += 1
            
            # Show live theta
            theta_str = f"{snap.get('theta_deg', '?'):.1f}°" if isinstance(snap.get('theta_deg'), (int, float)) else "?"
            print(f"\r    Samples: {sample_count} | Theta: {theta_str}  ", end="", flush=True)
        
        elapsed = time.time() - start_t
        print(f"\n  ✓ Sweep {sweep_idx + 1} done: {sample_count} samples in {elapsed:.1f}s")
        
        calibration_data["sweeps"].append({
            "sweep_index": sweep_idx,
            "num_samples": sample_count,
            "duration_s": elapsed,
            "samples": sweep_samples,
        })
    
    # ─────────────────────────────────────────────
    # Phase 2: Balance-point captures
    # ─────────────────────────────────────────────
    NUM_CAPTURES = 3
    SETTLE_SAMPLES = 20  # Average over this many samples for stability
    
    print(f"\n{'─' * 60}")
    print("  PHASE 2: Balance-Point Captures")
    print(f"{'─' * 60}")
    
    for cap_idx in range(NUM_CAPTURES):
        print(f"\n  Capture {cap_idx + 1}/{NUM_CAPTURES}")
        print("  Hold the robot at the exact balance position.")
        input("  Press ENTER to capture... ")
        
        print(f"    Averaging {SETTLE_SAMPLES} samples...", end="", flush=True)
        
        # Collect several samples and average for stability (step env for fresh data)
        samples = []
        for _ in range(SETTLE_SAMPLES):
            obs, _, _, _, _ = env.step(np.zeros(num_actions))
            snap = _snapshot_obs(env, obs_vector=obs)
            samples.append(snap)
        
        # Compute averaged values
        avg_snapshot = {"capture_index": cap_idx, "num_averaged": SETTLE_SAMPLES}
        
        # Average scalar/vector fields
        for key in ["dof_pos", "dof_vel", "quat", "ang_vel_body", "projected_gravity", "obs"]:
            vals = [s[key] for s in samples if key in s]
            if vals:
                avg_snapshot[key] = np.mean(vals, axis=0).tolist()
                avg_snapshot[f"{key}_std"] = np.std(vals, axis=0).tolist()
        
        # Average theta
        thetas = [s["theta_deg"] for s in samples if "theta_deg" in s]
        if thetas:
            avg_snapshot["theta_deg"] = float(np.mean(thetas))
            avg_snapshot["theta_deg_std"] = float(np.std(thetas))
            print(f"  θ = {avg_snapshot['theta_deg']:.2f}° ± {avg_snapshot['theta_deg_std']:.2f}°")
        else:
            print("  (no theta available)")
        
        # Also store all raw samples
        avg_snapshot["raw_samples"] = samples
        
        calibration_data["balance_points"].append(avg_snapshot)
    
    # ─────────────────────────────────────────────
    # Summary & Save
    # ─────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  CALIBRATION SUMMARY")
    print(f"{'─' * 60}")
    
    total_sweep_samples = sum(s["num_samples"] for s in calibration_data["sweeps"])
    print(f"  Sweeps: {NUM_SWEEPS} ({total_sweep_samples} total samples)")
    
    bp_thetas = [bp["theta_deg"] for bp in calibration_data["balance_points"] if "theta_deg" in bp]
    if bp_thetas:
        mean_theta = np.mean(bp_thetas)
        std_theta = np.std(bp_thetas)
        print(f"  Balance points: {NUM_CAPTURES}")
        print(f"  Mean balance θ: {mean_theta:.2f}° ± {std_theta:.2f}°")
        calibration_data["balance_theta_mean"] = float(mean_theta)
        calibration_data["balance_theta_std"] = float(std_theta)
    
    # Save to JSON (convert any numpy types for serialization)
    def _json_default(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    
    with open(output_path, 'w') as f:
        json.dump(calibration_data, f, indent=2, default=_json_default)
    
    print(f"\n  Saved to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print("=" * 60)
    print("\nCalibration complete! Use the balance_theta_mean as --target-theta.")
    
    # Leave calibration mode so normal enable/disable works again
    if hasattr(env, 'calibration_mode'):
        env.calibration_mode = False


def run_balance(env, target_theta: float = -117.57, kp: float = 0.035,
                ki: float = 0.003, kd: float = 0.06, error_scale: float = 15.0,
                near_balance_scale: float = 5.0, near_balance_min_gain: float = 0.25,
                rate_limit: float = 2.0, output_limit: float = 0.5,
                pos_kp: float = 0.3, pos_kd: float = 0.15,
                pos_max_theta: float = 4.0, position_hold: bool = True,
                duration: float = None):
    """Run balance mode with cascade PID control.
    
    Inner loop: PID on theta (tilt angle) to keep the robot balanced.
    Outer loop: estimates linear position from wheel encoders and adjusts
    the target theta to drive the robot back to its starting position.
    
    Outer loop output:
      theta_offset = pos_kp * position_error + pos_kd * velocity
      effective_target = base_target_theta + clamp(theta_offset, ±pos_max_theta)
    
    Position is estimated from wheel encoders (dof_pos). Since the two wheels
    are mounted in opposite directions, forward displacement is:
      position = (delta_dof_pos[0] - delta_dof_pos[1]) / 2
    
    Tuning for oscillation (do gradually, one change at a time):
      1. Increase Kd (--kd): more damping, reduces swing. Try 0.08, 0.10, 0.12.
      2. Decrease Kp (--kp): less aggressive correction. Try 0.03, 0.025.
      3. Decrease Ki (--ki) or set 0: integral can cause overshoot. Try 0.002 or 0.
      4. Lower --rate-limit: smoother output, less jerk. Try 1.5, 1.0.
      5. Lower --output-limit: cap max motor command. Try 0.4.
      6. Increase --near-balance-scale: softer near center. Try 8, 10.
    
    Tuning position hold:
      - If robot drifts but doesn't return: increase --pos-kp (try 0.5, 0.8).
      - If outer loop causes new oscillation: decrease --pos-kp, increase --pos-kd.
      - --pos-max-theta caps the maximum theta offset (default 4°).
      - Use --no-position-hold to disable and run inner loop only.
    """
    print("\n" + "=" * 60)
    print("Balance Mode - Cascade PID Control")
    print("=" * 60)
    print(f"  Target theta: {target_theta}°")
    print(f"  Inner PID: Kp={kp}, Ki={ki}, Kd={kd}")
    print(f"  Error scale: {error_scale}° | Near-balance: scale={near_balance_scale}° min_gain={near_balance_min_gain}")
    print(f"  Rate limit: {rate_limit}/s  Output limit: ±{output_limit}")
    if position_hold:
        print(f"  Outer position loop: pos_Kp={pos_kp}, pos_Kd={pos_kd}, max_offset=±{pos_max_theta}°")
    else:
        print(f"  Outer position loop: DISABLED")
    print(f"  Duration: {'infinite' if duration is None else f'{duration}s'}")
    print("=" * 60)
    print("\nControls: e=enable, d=disable, r=restart, c=calibrate, q=quit")
    
    # Inner PID (balance)
    pid = PIDController(kp=kp, ki=ki, kd=kd, output_limit=output_limit,
                        derivative_limit=50.0, rate_limit=rate_limit)
    
    # Reset environment
    obs, info = env.reset()
    pid.reset()
    
    # Outer loop state: integrate dof_vel to estimate displacement
    integrated_pos = 0.0
    last_outer_time = None
    theta_offset = 0.0
    
    start_time = time.time()
    step_count = 0
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            if duration is not None and elapsed >= duration:
                print(f"\n[Done] Reached duration limit ({duration}s)")
                break
            
            # --- Read sensor data ---
            current_theta = None
            
            if hasattr(env, 'state') and hasattr(env.state, 'derived'):
                if hasattr(env.state.derived, 'projected_gravity'):
                    pg = env.state.derived.projected_gravity
                    if pg is not None:
                        pg_arr = np.asarray(pg).flatten()
                        if len(pg_arr) >= 2:
                            theta_rad = np.arctan2(pg_arr[1], pg_arr[0])
                            current_theta = np.degrees(theta_rad)
            
            if current_theta is None and hasattr(env, 'observable_data') and env.observable_data:
                if 'quat' in env.observable_data:
                    quat = env.observable_data['quat']
                    if quat is not None:
                        x, y, z, w = quat
                        gx = 2 * (x * z - w * y)
                        gy = 2 * (y * z + w * x)
                        gz = w * w - x * x - y * y + z * z
                        pg_arr = np.array([-gx, -gy, -gz])
                        theta_rad = np.arctan2(pg_arr[1], pg_arr[0])
                        current_theta = np.degrees(theta_rad)
            
            if current_theta is None:
                if step_count % 50 == 0:
                    print(f"\r[Step {step_count}] Waiting for IMU data...", end="", flush=True)
                num_actions = env.action_space.shape[0]
                obs, reward, done, truncated, info = env.step(np.zeros(num_actions))
                step_count += 1
                if done or truncated:
                    obs, info = env.reset()
                    pid.reset()
                    init_dof_pos = None
                continue
            
            # --- Outer loop: position hold by integrating dof_vel ---
            theta_offset = 0.0
            wheel_vel = 0.0
            
            if position_hold and hasattr(env, 'observable_data') and env.observable_data:
                dof_vel = env.observable_data.get('dof_vel')
                
                if dof_vel is not None:
                    dof_vel = np.asarray(dof_vel).flatten()
                    # Net wheel velocity (both wheels mounted opposite)
                    wheel_vel = (dof_vel[0] - dof_vel[1]) / 2.0
                    
                    # Integrate velocity to get displacement estimate
                    if last_outer_time is not None:
                        dt = current_time - last_outer_time
                        if dt > 0:
                            integrated_pos += wheel_vel * dt
                    last_outer_time = current_time
                    
                    # PD on integrated position + velocity damping
                    raw_offset = pos_kp * integrated_pos + pos_kd * wheel_vel
                    theta_offset = np.clip(raw_offset, -pos_max_theta, pos_max_theta)
            
            effective_target = target_theta + theta_offset
            
            # --- Gyro for inner D-term ---
            gyro_theta_rate = 0.0
            if hasattr(env, 'observable_data') and env.observable_data:
                ang_vel = env.observable_data.get('ang_vel_body')
                if ang_vel is not None:
                    ang_vel = np.asarray(ang_vel).flatten()
                    gyro_theta_rate = np.degrees(ang_vel[0])
            
            # --- Inner loop: balance PID ---
            raw_error = effective_target - current_theta
            while raw_error > 180:
                raw_error -= 360
            while raw_error < -180:
                raw_error += 360
            
            error = error_scale * np.tanh(raw_error / error_scale)
            
            abs_err = abs(raw_error)
            ramp = 1.0 - (1.0 - near_balance_min_gain) * np.exp(-abs_err / max(1e-6, near_balance_scale))
            error = error * ramp
            
            control_output = pid.update(error, current_time, external_derivative=gyro_theta_rate)
            
            # --- Apply to motors ---
            num_actions = env.action_space.shape[0]
            action = np.zeros(num_actions)
            action[0] = control_output
            action[1] = -control_output

            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            
            if step_count % 50 == 0:
                pos_str = f"Pos: {integrated_pos:+.2f}rad v:{wheel_vel:+.2f} θoff: {theta_offset:+.1f}°" if position_hold else ""
                print(f"\r[{step_count}] {elapsed:.0f}s | "
                      f"θ:{current_theta:.1f}° tgt:{effective_target:.1f}° "
                      f"err:{raw_error:+.1f}° ctrl:{control_output:+.3f} "
                      f"gyro:{gyro_theta_rate:+.1f}°/s {pos_str}",
                      end="", flush=True)
            
            if done or truncated:
                print(f"\n[Episode ended] Done={done}, Truncated={truncated}")
                obs, info = env.reset()
                pid.reset()
                integrated_pos = 0.0
                last_outer_time = None
    
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    
    finally:
        num_actions = env.action_space.shape[0]
        env.step(np.zeros(num_actions))
        
        elapsed = time.time() - start_time
        print(f"\n\nBalance mode ended: {step_count} steps in {elapsed:.1f}s")



# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse arguments
    args = parse_args()
    
    print("=" * 60)
    print("Real Robot Control - RealMetaMachine")
    print("=" * 60)
    
    # Check if loading multiple policies
    if args.log_dirs:
        # Multi-policy mode - use utility module
        print(f"\nMulti-policy mode: Loading from {len(args.log_dirs)} directories")
        try:
            from metamachine.utils.policy_runner import load_policies
            
            # Load all policies
            runner, first_cfg = load_policies(
                args.log_dirs,
                policy_names=args.model_names,
                checkpoint=args.checkpoint
            )
            
            if runner.num_policies == 0:
                print("Error: No policies loaded successfully. Exiting.")
                return
            
            # Build cfg_real overrides from CLI arguments
            cfg_real_overrides = {}
            if args.module_ids is not None:
                cfg_real_overrides["module_ids"] = args.module_ids
            if args.sensor_module_ids is not None:
                cfg_real_overrides["sensor_module_ids"] = args.sensor_module_ids
            if args.no_dashboard:
                cfg_real_overrides["enable_dashboard"] = False
            
            # Create real robot environment from first config
            if first_cfg is not None:
                # Apply real robot overrides
                if cfg_real_overrides:
                    if not hasattr(first_cfg, 'real') or first_cfg.real is None:
                        first_cfg.real = {}
                    for key, value in cfg_real_overrides.items():
                        setattr(first_cfg.real, key, value)
                
                # Force real mode
                first_cfg.environment.mode = "real"
                
                env = create_environment(first_cfg)
            else:
                print("Error: No config available. Exiting.")
                return
            
            try:
                if args.viewer:
                    # Use viewer-enhanced version
                    run_sb3_policy_multi_with_viewer(
                        env, runner, 
                        duration=args.duration,
                        viewer_config=args.viewer_config
                    )
                else:
                    # Standard version without viewer
                    run_sb3_policy_multi(env, runner, duration=args.duration)
            finally:
                print("\nCleaning up...")
                env.close()
            return
            
        except ImportError as e:
            print(f"Error: Could not import policy_runner utilities: {e}")
            return
    
    # Check if loading from single training log directory
    if args.log_dir:
        # Use load_from_checkpoint utility for seamless loading
        print(f"\nLoading from training log: {args.log_dir}")
        try:
            from metamachine.utils.sb3_utils import load_from_checkpoint
            
            # Build cfg_real overrides from CLI arguments
            cfg_real_overrides = {}
            if args.module_ids is not None:
                cfg_real_overrides["module_ids"] = args.module_ids
            if args.sensor_module_ids is not None:
                cfg_real_overrides["sensor_module_ids"] = args.sensor_module_ids
            if args.no_dashboard:
                cfg_real_overrides["enable_dashboard"] = False
            
            env, model, cfg = load_from_checkpoint(
                args.log_dir,
                checkpoint=args.checkpoint,
                real_robot=True,
                cfg_real=cfg_real_overrides if cfg_real_overrides else None
            )
            
            # Override dashboard setting if requested (redundant but safe)
            if args.no_dashboard:
                if hasattr(cfg, 'real') and cfg.real:
                    cfg.real.enable_dashboard = False
            
            try:
                if args.viewer:
                    # Use viewer-enhanced version
                    if model is not None:
                        run_sb3_policy_with_viewer(
                            env, model, 
                            duration=args.duration,
                            viewer_config=args.viewer_config
                        )
                    else:
                        print("No model found in log directory. Running idle mode with viewer.")
                        run_idle_with_viewer(env, duration=args.duration, viewer_config=args.viewer_config)
                else:
                    # Standard version without viewer
                    if model is not None:
                        run_sb3_policy(env, model, duration=args.duration)
                    else:
                        print("No model found in log directory. Running idle mode.")
                        run_idle(env, duration=args.duration)
            finally:
                print("\nCleaning up...")
                env.close()
            return
            
        except ImportError as e:
            print(f"Error: Could not import sb3_utils: {e}")
            print("Falling back to manual config loading...")
    
    # Load configuration from file
    print(f"\nLoading config: {args.config}")
    cfg = load_config(args.config)
    
    # Override dashboard setting if requested
    if args.no_dashboard:
        if "real" not in cfg:
            cfg.real = {}
        cfg.real.enable_dashboard = False
    
    # Create environment
    print("\nCreating RealMetaMachine environment...")
    env = create_environment(cfg)
    
    try:
        if args.calibration:
            run_calibration(env, output_path=args.calibration_output)
            return
        if args.balance:
            run_balance(
                env,
                target_theta=args.target_theta,
                kp=args.kp,
                ki=args.ki,
                kd=args.kd,
                error_scale=args.error_scale,
                near_balance_scale=args.near_balance_scale,
                near_balance_min_gain=args.near_balance_min_gain,
                rate_limit=args.rate_limit,
                output_limit=args.output_limit,
                pos_kp=args.pos_kp,
                pos_kd=args.pos_kd,
                pos_max_theta=args.pos_max_theta,
                position_hold=not args.no_position_hold,
                duration=args.duration,
            )
            return
        if args.test_motion:
            # Run sinusoidal test motion
            if args.viewer:
                run_sinusoidal_test_with_viewer(
                    env,
                    amplitude=args.amplitude,
                    frequency=args.frequency,
                    duration=args.duration,
                    viewer_config=args.viewer_config
                )
            else:
                run_sinusoidal_test(
                    env,
                    amplitude=args.amplitude,
                    frequency=args.frequency,
                    duration=args.duration
                )
        
        elif args.policy:
            # Load and run policy
            policy = load_policy(args.policy, env)
            if policy is not None:
                if args.viewer:
                    run_policy_with_viewer(env, policy, duration=args.duration, viewer_config=args.viewer_config)
                else:
                    run_policy(env, policy, duration=args.duration)
            else:
                print("Failed to load policy. Running idle mode instead.")
                if args.viewer:
                    run_idle_with_viewer(env, duration=args.duration, viewer_config=args.viewer_config)
                else:
                    run_idle(env, duration=args.duration)
        
        else:
            # Run idle mode (monitoring only)
            if args.viewer:
                run_idle_with_viewer(env, duration=args.duration, viewer_config=args.viewer_config)
            else:
                run_idle(env, duration=args.duration)
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        env.close()


if __name__ == "__main__":
    main()
