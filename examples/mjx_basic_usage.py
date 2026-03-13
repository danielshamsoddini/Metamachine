#!/usr/bin/env python3
"""
MJX Basic Usage Example

This script demonstrates how to use the MJX-based MetaMachine environment
for GPU-accelerated parallel simulation.

Key Features Demonstrated:
1. Single environment simulation with video recording or real-time viewer
2. Batched parallel simulation with jax.vmap
3. JIT compilation for speed
4. Unified rendering API (render_mode="mp4" or "viewer")

Usage:
    # Single environment (default)
    python examples/mjx_basic_usage.py
    
    # Batched simulation (1024 parallel envs)
    python examples/mjx_basic_usage.py --batch_size 1024
    
    # With video rendering (mp4)
    python examples/mjx_basic_usage.py --render
    
    # With real-time viewer
    python examples/mjx_basic_usage.py --viewer

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>
"""

# Set MuJoCo rendering backend BEFORE any imports (required for headless EGL rendering)
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import time

import jax
import jax.numpy as jp
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="MJX Basic Usage Example")
    parser.add_argument(
        "--config",
        type=str,
        default="basic_quadruped",
        help="Config name from registry",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of parallel environments (1 = single env)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=200,
        help="Number of steps to run",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable video recording (mp4 mode)",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Enable real-time visualization (viewer mode)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path (default: auto in log_dir)",
    )
    args = parser.parse_args()
    
    # Print JAX device info
    from metamachine.utils.mjx_utils import print_mjx_info
    print("=" * 60)
    print("MJX MetaMachine Example")
    print("=" * 60)
    print_mjx_info()
    
    # Import and create environment
    from metamachine.environments.env_mjx import MJXMetaMachine
    from metamachine.environments.configs.config_registry import ConfigRegistry
    
    print(f"\nLoading config: {args.config}")
    cfg = ConfigRegistry.create_from_name(args.config)
    
    # Configure rendering mode
    if args.viewer:
        cfg.simulation.render_mode = "viewer"
    elif args.render:
        cfg.simulation.render_mode = "mp4"
    else:
        cfg.simulation.render_mode = "none"
    
    print("Creating MJX environment...")
    env = MJXMetaMachine(cfg)
    
    print(f"Action size: {env.action_size}")
    print(f"Observation size: {env.observation_size}")
    print(f"Control dt: {env.dt}s")
    print(f"Simulation dt: {env.sim_dt}s")
    print(f"Substeps: {env.n_substeps}")
    if env.log_dir:
        print(f"Log directory: {env.log_dir}")
    
    # Run simulation
    if args.batch_size == 1:
        run_single_env(env, args.num_steps, args.render or args.viewer, args.output)
    else:
        if args.viewer:
            print("Warning: Viewer mode only works with batch_size=1, ignoring --viewer")
        run_batched_env(env, args.batch_size, args.num_steps)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


def run_single_env(env, num_steps: int, render: bool, output_path: str = None):
    """Run a single environment with optional video recording or viewer."""
    from metamachine.utils.mjx_utils import create_single_env_fns, warmup_jit
    
    print("\n" + "=" * 60)
    print("Running Single Environment")
    print("=" * 60)
    
    # Determine render mode
    render_mode = env.render_mode
    is_viewer = render_mode == "viewer"
    is_mp4 = render_mode == "mp4"
    
    # Get control dt for real-time playback
    dt = env.dt
    
    if is_viewer:
        print(f"Viewer mode: Real-time visualization at {1/dt:.1f}Hz (dt={dt:.4f}s)")
    elif is_mp4:
        print("MP4 mode: Recording video (fast simulation)")
    
    # Create JIT-compiled functions
    jit_reset, jit_step = create_single_env_fns(env)
    
    # Warmup JIT
    print("Warming up JIT compilation...")
    warmup_jit(env, jit_reset, jit_step, batch_size=1)
    
    # Run simulation
    print(f"Running {num_steps} steps...")
    
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    
    # Initialize rendering based on mode
    if is_mp4:
        env.start_video_recording()
        env.capture_frame(state)
    elif is_viewer:
        env.sync_viewer(state)
    
    total_reward = 0.0
    start_time = time.time()
    
    for step in range(num_steps):
        # Record step start time for real-time playback
        step_start_time = time.time()
        
        # Simple sinusoidal action
        action = jp.zeros(env.action_size)
        action = action.at[0].set(jp.sin(step * 0.1))
        
        state = jit_step(state, action)
        total_reward += float(state.reward)
        
        # Update rendering based on mode
        if is_mp4:
            env.capture_frame(state)
        elif is_viewer:
            env.sync_viewer(state)
            # Sleep to maintain real-time frequency
            elapsed = time.time() - step_start_time
            sleep_time = max(0, dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        if state.done:
            print(f"Episode terminated at step {step}")
            break
    
    elapsed = time.time() - start_time
    fps = num_steps / elapsed
    
    print(f"Completed in {elapsed:.3f}s ({fps:.1f} steps/sec)")
    print(f"Total reward: {total_reward:.3f}")
    
    # Save video if recording mp4
    if is_mp4:
        if output_path:
            # Use specified output path
            import os
            video_dir = os.path.dirname(output_path) or "."
            os.makedirs(video_dir, exist_ok=True)
            suffix = "_" + os.path.basename(output_path).replace(".mp4", "")
            env.stop_video_recording(suffix=suffix)
        else:
            env.stop_video_recording()
    
    # Close viewer if used
    if is_viewer:
        env.close()


def run_batched_env(env, batch_size: int, num_steps: int):
    """Run batched parallel simulation."""
    from metamachine.utils.mjx_utils import create_batched_env_fns, warmup_jit
    
    print("\n" + "=" * 60)
    print(f"Running Batched Environment ({batch_size} parallel envs)")
    print("=" * 60)
    
    # Create JIT-compiled batched functions
    jit_batch_reset, jit_batch_step = create_batched_env_fns(env)
    
    # Warmup
    print("Warming up JIT compilation...")
    warmup_jit(env, jit_batch_reset, jit_batch_step, batch_size=batch_size)
    
    # Run simulation
    print(f"Running {num_steps} steps x {batch_size} envs...")
    
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, batch_size)
    states = jit_batch_reset(rngs)
    total_rewards = jp.zeros(batch_size)
    
    start_time = time.time()
    
    for step in range(num_steps):
        # Same action for all envs
        actions = jp.zeros((batch_size, env.action_size))
        actions = actions.at[:, 0].set(jp.sin(step * 0.1))
        
        states = jit_batch_step(states, actions)
        total_rewards = total_rewards + states.reward
    
    # Block until complete
    total_rewards = total_rewards.block_until_ready()
    
    elapsed = time.time() - start_time
    total_steps = num_steps * batch_size
    fps = total_steps / elapsed
    
    print(f"Completed in {elapsed:.3f}s ({fps:.1f} steps/sec)")
    print(f"Mean reward: {float(total_rewards.mean()):.3f}")
    print(f"Min reward: {float(total_rewards.min()):.3f}")
    print(f"Max reward: {float(total_rewards.max()):.3f}")


if __name__ == "__main__":
    main()
