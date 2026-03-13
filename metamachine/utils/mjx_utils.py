"""
MJX Utilities for GPU-accelerated simulation.

This module provides reusable utilities for MJX-based environments:
- Batched simulation runners
- JIT-compiled rollout functions
- Video rendering utilities for MJX trajectories

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>

Licensed under the Apache License, Version 2.0 (the "License");
"""

from typing import Any, Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np

try:
    from mujoco import mjx
    MJX_AVAILABLE = True
except ImportError:
    MJX_AVAILABLE = False
    mjx = None


def create_batched_env_fns(env) -> Tuple[Callable, Callable]:
    """Create JIT-compiled batched reset and step functions.
    
    Args:
        env: MJX environment with reset(rng) and step(state, action) methods
        
    Returns:
        Tuple of (jit_batch_reset, jit_batch_step)
        
    Example:
        >>> jit_batch_reset, jit_batch_step = create_batched_env_fns(env)
        >>> rngs = jax.random.split(jax.random.PRNGKey(0), 1024)
        >>> states = jit_batch_reset(rngs)
        >>> actions = jp.zeros((1024, env.action_size))
        >>> states = jit_batch_step(states, actions)
    """
    batch_reset = jax.vmap(env.reset)
    batch_step = jax.vmap(env.step)
    
    jit_batch_reset = jax.jit(batch_reset)
    jit_batch_step = jax.jit(batch_step)
    
    return jit_batch_reset, jit_batch_step


def create_single_env_fns(env) -> Tuple[Callable, Callable]:
    """Create JIT-compiled single-environment reset and step functions.
    
    Args:
        env: MJX environment with reset(rng) and step(state, action) methods
        
    Returns:
        Tuple of (jit_reset, jit_step)
    """
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    return jit_reset, jit_step


def warmup_jit(
    env,
    jit_reset: Callable,
    jit_step: Callable,
    rng: Optional[jax.Array] = None,
    batch_size: int = 1,
) -> None:
    """Warmup JIT compilation by running one reset and step.
    
    Args:
        env: MJX environment
        jit_reset: JIT-compiled reset function
        jit_step: JIT-compiled step function
        rng: Random key (default: PRNGKey(42))
        batch_size: Number of environments (1 for single, >1 for batched)
    """
    if rng is None:
        rng = jax.random.PRNGKey(42)
    
    if batch_size > 1:
        rngs = jax.random.split(rng, batch_size)
        states = jit_reset(rngs)
        actions = jp.zeros((batch_size, env.action_size))
        _ = jit_step(states, actions)
    else:
        state = jit_reset(rng)
        action = jp.zeros(env.action_size)
        _ = jit_step(state, action)


def run_batched_rollout(
    env,
    policy_fn: Callable,
    num_steps: int,
    batch_size: int,
    rng: Optional[jax.Array] = None,
    warmup: bool = True,
) -> Tuple[Any, jax.Array, float]:
    """Run a batched rollout with a policy function.
    
    Args:
        env: MJX environment
        policy_fn: Function (state, rng) -> action
        num_steps: Number of steps to run
        batch_size: Number of parallel environments
        rng: Random key (default: PRNGKey(0))
        warmup: Whether to warmup JIT compilation first
        
    Returns:
        Tuple of (final_states, total_rewards, elapsed_time)
    """
    import time
    
    if rng is None:
        rng = jax.random.PRNGKey(0)
    
    jit_batch_reset, jit_batch_step = create_batched_env_fns(env)
    
    if warmup:
        warmup_jit(env, jit_batch_reset, jit_batch_step, 
                   jax.random.PRNGKey(42), batch_size)
    
    # Run rollout
    rng, reset_rng = jax.random.split(rng)
    rngs = jax.random.split(reset_rng, batch_size)
    states = jit_batch_reset(rngs)
    total_rewards = jp.zeros(batch_size)
    
    start_time = time.time()
    
    for step in range(num_steps):
        rng, action_rng = jax.random.split(rng)
        actions = policy_fn(states, action_rng)
        states = jit_batch_step(states, actions)
        total_rewards = total_rewards + states.reward
    
    # Block until complete
    total_rewards = total_rewards.block_until_ready()
    elapsed = time.time() - start_time
    
    return states, total_rewards, elapsed


def run_single_rollout(
    env,
    policy_fn: Callable,
    num_steps: int,
    rng: Optional[jax.Array] = None,
    warmup: bool = True,
    collect_trajectory: bool = False,
) -> Tuple[Any, float, float, Optional[List]]:
    """Run a single-environment rollout with a policy function.
    
    Args:
        env: MJX environment
        policy_fn: Function (state, rng) -> action
        num_steps: Number of steps to run
        rng: Random key (default: PRNGKey(0))
        warmup: Whether to warmup JIT compilation first
        collect_trajectory: Whether to collect trajectory for rendering
        
    Returns:
        Tuple of (final_state, total_reward, elapsed_time, trajectory)
    """
    import time
    
    if rng is None:
        rng = jax.random.PRNGKey(0)
    
    jit_reset, jit_step = create_single_env_fns(env)
    
    if warmup:
        warmup_jit(env, jit_reset, jit_step, jax.random.PRNGKey(42), 1)
    
    # Run rollout
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)
    total_reward = 0.0
    trajectory = [state] if collect_trajectory else None
    
    start_time = time.time()
    
    for step in range(num_steps):
        rng, action_rng = jax.random.split(rng)
        action = policy_fn(state, action_rng)
        state = jit_step(state, action)
        total_reward += float(state.reward)
        
        if collect_trajectory:
            trajectory.append(state)
        
        if state.done:
            break
    
    elapsed = time.time() - start_time
    
    return state, total_reward, elapsed, trajectory


def render_mjx_trajectory(
    mj_model: mujoco.MjModel,
    trajectory: List,
    height: int = 480,
    width: int = 640,
    camera: Optional[Union[str, int]] = None,
) -> List[np.ndarray]:
    """Render MJX trajectory to list of images.
    
    Converts MJX data back to MuJoCo data for CPU rendering.
    
    Args:
        mj_model: MuJoCo model
        trajectory: List of MJXState objects with data attribute
        height: Image height
        width: Image width
        camera: Camera name or ID (-1 for default)
        
    Returns:
        List of rendered images as numpy arrays
    """
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    
    if camera is None:
        camera_id = -1
    elif isinstance(camera, str):
        camera_id = mj_model.camera(camera).id
    else:
        camera_id = camera
    
    images = []
    mj_data = mujoco.MjData(mj_model)
    
    for state in trajectory:
        # Copy state from MJX data to MuJoCo data
        mj_data.qpos[:] = np.array(state.data.qpos)
        mj_data.qvel[:] = np.array(state.data.qvel)
        mujoco.mj_forward(mj_model, mj_data)
        
        renderer.update_scene(mj_data, camera=camera_id)
        pixels = renderer.render()
        images.append(pixels.copy())
    
    renderer.close()
    return images


def save_video(
    images: List[np.ndarray],
    output_path: str,
    fps: int = 20,
) -> bool:
    """Save list of images as MP4 video.
    
    Args:
        images: List of images (H, W, 3) as numpy arrays
        output_path: Output file path
        fps: Frames per second
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        
        clip = ImageSequenceClip(images, fps=fps)
        clip.write_videofile(
            output_path,
            codec="libx264",
            fps=fps,
            audio=False,
            logger=None,
        )
        clip.close()
        return True
    except ImportError:
        print("moviepy not available, saving frames as numpy array...")
        np.save(output_path.replace(".mp4", ".npy"), np.array(images))
        return False
    except Exception as e:
        print(f"Error saving video: {e}")
        return False


def get_mjx_data_as_mujoco(
    mj_model: mujoco.MjModel,
    mjx_data: Any,
    mj_data: Optional[mujoco.MjData] = None,
) -> mujoco.MjData:
    """Convert MJX data to MuJoCo data for rendering.
    
    This is the recommended way to render MJX simulations - convert
    the MJX data back to CPU MuJoCo data for visualization.
    
    Args:
        mj_model: MuJoCo model
        mjx_data: MJX Data object (or MJXState.data)
        mj_data: Optional pre-allocated MjData (reused for efficiency)
        
    Returns:
        MuJoCo MjData with state from mjx_data
    """
    if mj_data is None:
        mj_data = mujoco.MjData(mj_model)
    
    # Use mjx.get_data if available (more complete transfer)
    if MJX_AVAILABLE and hasattr(mjx, 'get_data'):
        mj_data = mjx.get_data(mj_model, mjx_data)
    else:
        # Manual copy of essential fields
        mj_data.qpos[:] = np.array(mjx_data.qpos)
        mj_data.qvel[:] = np.array(mjx_data.qvel)
        mujoco.mj_forward(mj_model, mj_data)
    
    return mj_data


def print_mjx_info():
    """Print information about MJX availability and JAX backend."""
    print(f"MJX available: {MJX_AVAILABLE}")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")
    
    if jax.default_backend() == 'cpu':
        print("\nNote: Running on CPU. For GPU acceleration, install CUDA-enabled JAX:")
        print("  pip install --upgrade 'jax[cuda12_pip]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")


# Zero-action policy for testing
def zero_policy(state, rng):
    """Simple zero-action policy for testing."""
    action_size = state.info["last_act"].shape[-1]
    return jp.zeros(action_size)


def random_policy(state, rng, scale: float = 1.0):
    """Random action policy for testing."""
    action_size = state.info["last_act"].shape[-1]
    return jax.random.uniform(rng, shape=(action_size,), minval=-scale, maxval=scale)

