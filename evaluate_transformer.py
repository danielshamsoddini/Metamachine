#!/usr/bin/env python3
"""
Evaluate a trained transformer policy by loading a checkpoint and running a robot rollout with video.

Usage:
    python evaluate_transformer.py --checkpoint ./logs/transformer_multi_expert/final_model.pt
    python evaluate_transformer.py --checkpoint ./logs/transformer_multi_expert/final_model.pt --n-steps 1000 --seed 123
"""

import argparse
import json
import os
import time
import numpy as np


def load_action_norm(checkpoint_path):
    """Load action normalization stats from checkpoint or sidecar JSON file.
    
    Returns:
        (action_mean, action_std) as numpy arrays, or (None, None) if not found.
    """
    import torch
    
    # First try loading from the checkpoint itself
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "action_norm" in checkpoint:
            norm = checkpoint["action_norm"]
            print(f"  Loaded action normalization from checkpoint")
            return np.array(norm["action_mean"]), np.array(norm["action_std"])
    except Exception:
        pass
    
    # Fallback: try sidecar JSON file next to the checkpoint
    log_dir = os.path.dirname(checkpoint_path) or "."
    json_path = os.path.join(log_dir, "action_norm.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            norm = json.load(f)
        print(f"  Loaded action normalization from {json_path}")
        return np.array(norm["action_mean"]), np.array(norm["action_std"])
    
    return None, None


def evaluate_robot_rollout(trainer, n_steps=1000, seed=42,
                           action_mean=None, action_std=None,
                           config_path=None, pose_path=None):
    """
    Run robot rollout with the trained policy and record video.
    
    Args:
        action_mean: If provided, denormalize predicted actions: action = pred * std + mean
        action_std: If provided, denormalize predicted actions: action = pred * std + mean
        config_path: Path to environment config YAML (uses modular_quadruped default if None)
        pose_path: Path to optimized_pose.yaml to override default_dof_pos, init_pos, init_quat
    """
    from metamachine.environments.configs.config_registry import ConfigRegistry
    from metamachine.environments.env_sim import MetaMachine
    from omegaconf import OmegaConf

    # Get inference policy
    policy = trainer.get_inference()
    policy.reset()

    if action_mean is not None:
        print(f"  Action denormalization enabled:")
        print(f"    mean = {action_mean}")
        print(f"    std  = {action_std}")

    # Load environment config
    if config_path is not None:
        print(f"  Loading config from: {config_path}")
        cfg = OmegaConf.load(config_path)
    else:
        cfg = ConfigRegistry.create_from_name("modular_quadruped")
    
    # Override with optimized pose if provided
    if pose_path is not None:
        print(f"  Loading optimized pose from: {pose_path}")
        pose = OmegaConf.load(pose_path)
        
        # Start the robot at the optimized joint positions, but set
        # default_dof_pos to zero so env.step() doesn't add an offset.
        # This way the transformer's absolute joint targets pass through directly.
        cfg.initialization.init_joint_pos = pose.default_dof_pos
        cfg.control.default_dof_pos = [0.0] * len(pose.default_dof_pos)
        cfg.initialization.init_pos = pose.init_pos
        cfg.initialization.init_quat = pose.init_quat
        
        # Disable pose optimization since we already have the optimized pose
        cfg.pose_optimization.enabled = False
        
        # Load orientation vectors needed by ballance_auto termination and rewards
        if hasattr(pose, 'projected_upward'):
            cfg.observation.projected_upward_vec = list(pose.projected_upward)
        if hasattr(pose, 'projected_forward'):
            cfg.observation.projected_forward_vec = list(pose.projected_forward)
        if hasattr(pose, 'forward_vec'):
            cfg.observation.forward_vec = list(pose.forward_vec)
        
        print(f"    init_joint_pos = {list(pose.default_dof_pos)} (from optimized pose)")
        print(f"    default_dof_pos = {list(cfg.control.default_dof_pos)} (zeroed)")
        print(f"    init_pos = {list(pose.init_pos)}")
        print(f"    init_quat = {list(pose.init_quat)}")
        print(f"    projected_upward = {list(pose.get('projected_upward', []))}")
        print(f"    pose_optimization: disabled (using provided pose)")
    elif config_path is None:
        # Default v1 pose when no config or pose is specified
        cfg.control.default_dof_pos = [0, 0, 0, 0, 0]
        if 'initialization' not in cfg:
            cfg['initialization'] = OmegaConf.create()
        cfg.initialization.init_joint_pos = [0, -1., 1., 1., -1.]

    # Ensure modular observations are enabled for transformer
    if not OmegaConf.is_missing(cfg, "observation") and hasattr(cfg.observation, 'modular'):
        cfg.observation.modular.enabled = True
    
    cfg.control.symmetric_limit = 100  # Don't clip transformer outputs
    
    # Restore default termination conditions
    from omegaconf import OmegaConf as _OC
    _OC.update(cfg, "task.termination_conditions.termination_strategy", "ballance_auto", force_add=True)
    
    cfg.simulation.render = True
    cfg.simulation.render_mode = "mp4"
    cfg.simulation.video_record_interval = 1

    env = MetaMachine(cfg)
    obs, _ = env.reset(seed=seed)

    env_log_dir = getattr(env, '_log_dir', './logs')
    print(f"  Video will be saved to: {env_log_dir}")

    obs_info = env.state.get_modular_observation_info()
    print(f"  Observation structure: {obs_info['num_modules']} modules × "
          f"{obs_info['per_module_obs_size']} + {obs_info['global_obs_size']} global "
          f"= {obs_info['total_obs_size']} total")

    # Run rollout
    total_reward = 0.0
    positions = []
    all_actions = []

    for step in range(n_steps):
        t0 = time.time()

        state = env.state.flat_obs_to_dict(obs)
        action = policy.step(state)
        
        # Denormalize actions if normalization stats are provided
        if action_mean is not None and action_std is not None:
            action = action * action_std + action_mean
        
        all_actions.append(action.copy())

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        if hasattr(env, 'state'):
            positions.append(env.state.raw.pos_world[:2].copy())

        elapsed = time.time() - t0
        sleep_time = max(0, cfg.control.dt * 0.5 - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

        if done or truncated:
            print(f"  Episode ended at step {step}")
            break

    # Calculate distance traveled
    distance = 0.0
    if len(positions) > 1:
        positions_arr = np.array(positions)
        distance = np.linalg.norm(positions_arr[-1] - positions_arr[0])

    env.close()

    # Rename video
    try:
        import glob
        video_files = glob.glob(os.path.join(env_log_dir, "episode_*.mp4"))
        if video_files:
            latest_video = max(video_files, key=os.path.getmtime)
            new_name = os.path.join(env_log_dir, "transformer_eval.mp4")
            os.rename(latest_video, new_name)
            print(f"  Video saved: {new_name}")
    except Exception as e:
        print(f"  Warning: Could not rename video: {e}")

    # Print action statistics
    all_actions = np.array(all_actions)
    print(f"\n  Action statistics:")
    print(f"    Shape: {all_actions.shape}")
    for j in range(all_actions.shape[1]):
        print(f"    Joint {j}: mean={all_actions[:, j].mean():.3f}, "
              f"std={all_actions[:, j].std():.3f}, "
              f"min={all_actions[:, j].min():.3f}, "
              f"max={all_actions[:, j].max():.3f}")

    print(f"\n  Results:")
    print(f"    Total reward: {total_reward:.2f}")
    print(f"    Distance traveled: {distance:.3f}m")
    print(f"    Steps completed: {step + 1}")
    print(f"    Video dir: {env_log_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained transformer policy")
    parser.add_argument("--checkpoint", type=str,
                        default="./logs/transformer_multi_expert/final_model.pt",
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--n-steps", type=int, default=1000,
                        help="Number of rollout steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to environment config YAML (e.g. logs/20260127_102303m/config.yaml)")
    parser.add_argument("--pose", type=str, default=None,
                        help="Path to optimized_pose.yaml to set default_dof_pos and init pose")
    args = parser.parse_args()

    from capyformer import HFActionChunkingTrainer

    print("=" * 60)
    print("Transformer Policy Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config or '(default modular_quadruped)'}")
    print(f"Pose: {args.pose or '(default v1 pose)'}")
    print(f"Steps: {args.n_steps}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    # Load model from checkpoint (no dataset needed)
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: Running on CPU - inference will be very slow!")
    trainer = HFActionChunkingTrainer.from_checkpoint(args.checkpoint, device=device)
    
    # Load action normalization stats (if available)
    action_mean, action_std = load_action_norm(args.checkpoint)

    print(f"\nRunning rollout...")
    evaluate_robot_rollout(trainer, n_steps=args.n_steps, seed=args.seed,
                           action_mean=action_mean, action_std=action_std,
                           config_path=args.config, pose_path=args.pose)

    print("\nDone!")


if __name__ == "__main__":
    main()
