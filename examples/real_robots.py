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

    # Manual positioning mode (all kp/kd = 0)
    python examples/real_robots.py --manual-position
    python examples/real_robots.py --manual-position --viewer \
        --config metamachine/environments/configs/default_configs/real_one_module.yaml \
        --module-ids 23
    
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
import os
import signal
import sys
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
    "default_configs" / "real_three_modules.yaml"
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

    # Manual positioning mode (all kp/kd = 0)
    python real_robots.py --manual-position
    python real_robots.py --manual-position --viewer
    
    # Run with trained policy
    python real_robots.py --policy logs/experiment/policy.pt
    
    # Load from training log with custom module IDs
    python real_robots.py --log-dir logs/20251228_223556l_lego_tripod --module-ids 5 21 16
    
    # Load from training log with viewer
    python real_robots.py --log-dir logs/experiment --module-ids 5 21 16 --viewer
    
    # Load from training log with sensor modules
    python real_robots.py --log-dir logs/experiment --module-ids 5 21 16 --sensor-module-ids 100
    
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

    parser.add_argument(
        "--manual-position",
        action="store_true",
        help="Manual positioning mode: set all kp/kd to 0 and run idle monitoring"
    )
    
    return parser.parse_args()


def build_cfg_real_overrides(args) -> dict:
    """Build real-robot config overrides from CLI arguments."""
    cfg_real_overrides = {}
    if args.module_ids is not None:
        cfg_real_overrides["module_ids"] = args.module_ids
    if args.sensor_module_ids is not None:
        cfg_real_overrides["sensor_module_ids"] = args.sensor_module_ids
    if args.no_dashboard:
        cfg_real_overrides["enable_dashboard"] = False
    return cfg_real_overrides


def apply_cfg_real_overrides(cfg, cfg_real_overrides: dict) -> None:
    """Apply real-robot overrides to a loaded config.

    Keeps dependent fields in sync so RealMetaMachine validation passes:
    - control.num_actions matches len(real.module_ids)
    - real.sources.main_imu / goal_distance stay within valid module IDs
    """
    if not cfg_real_overrides:
        return

    if not hasattr(cfg, "real") or cfg.real is None:
        cfg.real = {}

    for key, value in cfg_real_overrides.items():
        setattr(cfg.real, key, value)

    active_ids = list(cfg.real.get("module_ids", []))
    sensor_ids = list(cfg.real.get("sensor_module_ids", []))

    if active_ids:
        # Keep action dimensionality consistent with active modules.
        cfg.control.num_actions = len(active_ids)

        # Keep default position vector shape aligned with num_actions.
        default_dof_pos = cfg.control.get("default_dof_pos", None)
        if isinstance(default_dof_pos, (list, tuple)):
            default_dof_pos = list(default_dof_pos)
            if len(default_dof_pos) != len(active_ids):
                if len(default_dof_pos) == 0:
                    cfg.control.default_dof_pos = [0.0] * len(active_ids)
                elif len(default_dof_pos) > len(active_ids):
                    cfg.control.default_dof_pos = default_dof_pos[:len(active_ids)]
                else:
                    fill_value = float(default_dof_pos[-1])
                    cfg.control.default_dof_pos = default_dof_pos + [
                        fill_value
                    ] * (len(active_ids) - len(default_dof_pos))

        all_ids = set(active_ids) | set(sensor_ids)
        if "sources" not in cfg.real or cfg.real.sources is None:
            cfg.real.sources = {}

        main_imu_id = cfg.real.sources.get("main_imu", active_ids[0])
        if main_imu_id not in all_ids:
            cfg.real.sources.main_imu = active_ids[0]
            print(
                f"[Info] real.sources.main_imu={main_imu_id} is invalid after "
                f"CLI overrides. Using {active_ids[0]}."
            )

        goal_distance_id = cfg.real.sources.get("goal_distance", None)
        if goal_distance_id is not None and goal_distance_id not in all_ids:
            cfg.real.sources.goal_distance = active_ids[0]
            print(
                f"[Info] real.sources.goal_distance={goal_distance_id} is invalid "
                f"after CLI overrides. Using {active_ids[0]}."
            )


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


def enable_manual_position_mode(env) -> None:
    """Zero all PD gains so joints can be moved by hand."""
    num_actions = len(getattr(env, "expected_module_ids", []))
    if num_actions <= 0 and hasattr(env, "action_space"):
        num_actions = int(env.action_space.shape[0])

    zeros = np.zeros(num_actions, dtype=np.float32)

    # Runtime gains used for regular step() command streaming.
    env.kps = zeros.copy()
    env.kds = zeros.copy()

    # Default gains used by enable/special command packets.
    env.kp_default = 0.0
    env.kd_default = 0.0

    # Keep config aligned for introspection/debugging.
    if hasattr(env, "cfg") and hasattr(env.cfg, "control"):
        env.cfg.control.kp = 0.0
        env.cfg.control.kd = 0.0

    print("\n[Mode] Manual positioning enabled")
    print("  All kp/kd gains set to 0.0")
    print("  Use 'e' to enable and move joints by hand")


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
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse arguments
    args = parse_args()
    cfg_real_overrides = build_cfg_real_overrides(args)
    
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

            # Create real robot environment from first config
            if first_cfg is not None:
                # Apply real robot overrides
                apply_cfg_real_overrides(first_cfg, cfg_real_overrides)
                
                # Force real mode
                first_cfg.environment.mode = "real"
                
                env = create_environment(first_cfg)
                if args.manual_position:
                    enable_manual_position_mode(env)
            else:
                print("Error: No config available. Exiting.")
                return
            
            try:
                if args.manual_position:
                    if args.viewer:
                        run_idle_with_viewer(
                            env, duration=args.duration, viewer_config=args.viewer_config
                        )
                    else:
                        run_idle(env, duration=args.duration)
                elif args.viewer:
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

            env, model, cfg = load_from_checkpoint(
                args.log_dir,
                checkpoint=args.checkpoint,
                real_robot=True,
                cfg_real=cfg_real_overrides if cfg_real_overrides else None
            )
            if args.manual_position:
                enable_manual_position_mode(env)
            
            # Override dashboard setting if requested (redundant but safe)
            if args.no_dashboard:
                if hasattr(cfg, 'real') and cfg.real:
                    cfg.real.enable_dashboard = False
            
            try:
                if args.manual_position:
                    if args.viewer:
                        run_idle_with_viewer(env, duration=args.duration, viewer_config=args.viewer_config)
                    else:
                        run_idle(env, duration=args.duration)
                elif args.viewer:
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

    # Apply CLI real-robot overrides (module IDs, sensor IDs, dashboard).
    apply_cfg_real_overrides(cfg, cfg_real_overrides)
    
    # Create environment
    print("\nCreating RealMetaMachine environment...")
    env = create_environment(cfg)
    if args.manual_position:
        enable_manual_position_mode(env)
    
    try:
        if args.manual_position:
            # Manual positioning always runs idle loop.
            if args.viewer:
                run_idle_with_viewer(env, duration=args.duration, viewer_config=args.viewer_config)
            else:
                run_idle(env, duration=args.duration)
        elif args.test_motion:
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
