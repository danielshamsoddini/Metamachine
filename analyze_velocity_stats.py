#!/usr/bin/env python3
"""
Analyze velocity statistics from expert datasets to determine forward/backward motion bias.
"""

import pickle
import numpy as np
from pathlib import Path

def load_and_analyze_velocities(filepath, dataset_name):
    """Load pickle file and extract velocity statistics."""
    print(f"\n{'='*60}")
    print(f"Analyzing {dataset_name}")
    print(f"{'='*60}")
    
    print(f"Loading {filepath}...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Data type: {type(data)}")
    
    # Try to extract accurate_vel_world
    vel_world = None
    
    # Check if data is a dict
    if isinstance(data, dict):
        print(f"Keys in data: {list(data.keys())}")
        
        # Look for accurate_vel_world at top level
        if 'accurate_vel_world' in data:
            vel_world = data['accurate_vel_world']
            print("Found 'accurate_vel_world' key at top level")
        elif 'vel_world' in data:
            vel_world = data['vel_world']
            print("Found 'vel_world' key at top level")
        elif 'velocity' in data:
            vel_world = data['velocity']
            print("Found 'velocity' key at top level")
        elif 'trajectories' in data:
            # Handle trajectories structure
            trajectories = data['trajectories']
            print(f"Found 'trajectories' key with {len(trajectories)} trajectories")
            
            if len(trajectories) > 0:
                print(f"First trajectory type: {type(trajectories[0])}")
                if isinstance(trajectories[0], dict):
                    print(f"First trajectory keys: {list(trajectories[0].keys())}")
                    
                    # Try to collect velocities from all trajectories
                    if 'accurate_vel_world' in trajectories[0]:
                        vel_list = [traj['accurate_vel_world'] for traj in trajectories if 'accurate_vel_world' in traj]
                        if vel_list:
                            vel_world = np.concatenate(vel_list, axis=0)
                            print(f"Collected 'accurate_vel_world' from {len(vel_list)} trajectories")
                    elif 'vel_world' in trajectories[0]:
                        vel_list = [traj['vel_world'] for traj in trajectories if 'vel_world' in traj]
                        if vel_list:
                            vel_world = np.concatenate(vel_list, axis=0)
                            print(f"Collected 'vel_world' from {len(vel_list)} trajectories")
                    elif 'velocity' in trajectories[0]:
                        vel_list = [traj['velocity'] for traj in trajectories if 'velocity' in traj]
                        if vel_list:
                            vel_world = np.concatenate(vel_list, axis=0)
                            print(f"Collected 'velocity' from {len(vel_list)} trajectories")
                    elif 'obs' in trajectories[0]:
                        # Check if obs contains velocity
                        obs = trajectories[0]['obs']
                        if isinstance(obs, dict) and 'accurate_vel_world' in obs:
                            vel_list = [traj['obs']['accurate_vel_world'] for traj in trajectories if 'obs' in traj and 'accurate_vel_world' in traj['obs']]
                            if vel_list:
                                vel_world = np.concatenate(vel_list, axis=0)
                                print(f"Collected 'accurate_vel_world' from obs in {len(vel_list)} trajectories")
    
    # Check if data is a list (e.g., list of trajectories)
    elif isinstance(data, list):
        print(f"Data is a list with {len(data)} elements")
        if len(data) > 0:
            print(f"First element type: {type(data[0])}")
            if isinstance(data[0], dict):
                print(f"First element keys: {list(data[0].keys())}")
                # Try to collect velocities from all trajectories
                if 'accurate_vel_world' in data[0]:
                    vel_list = [traj['accurate_vel_world'] for traj in data if 'accurate_vel_world' in traj]
                    if vel_list:
                        vel_world = np.concatenate(vel_list, axis=0)
                        print("Collected 'accurate_vel_world' from all trajectories")
    
    # Check if data is already a numpy array
    elif isinstance(data, np.ndarray):
        print("Data is a numpy array")
        vel_world = data
    
    if vel_world is None:
        print("ERROR: Could not find velocity data. Attempting to calculate from position...")
        # Try to find position data and calculate velocity
        if isinstance(data, dict):
            if 'trajectories' in data:
                trajectories = data['trajectories']
                pos_list = []
                for traj in trajectories:
                    if isinstance(traj, dict):
                        if 'pos_world' in traj:
                            pos_list.append(traj['pos_world'])
                        elif 'position' in traj:
                            pos_list.append(traj['position'])
                        elif 'obs' in traj and isinstance(traj['obs'], dict):
                            if 'pos_world' in traj['obs']:
                                pos_list.append(traj['obs']['pos_world'])
                            elif 'position' in traj['obs']:
                                pos_list.append(traj['obs']['position'])
                
                if pos_list:
                    # Concatenate all positions and calculate velocity
                    all_pos = np.concatenate(pos_list, axis=0)
                    vel_world = np.diff(all_pos, axis=0)
                    print(f"Calculated velocity from position data in {len(pos_list)} trajectories")
            elif 'pos_world' in data:
                pos = data['pos_world']
                vel_world = np.diff(pos, axis=0)
                print("Calculated velocity from 'pos_world'")
            elif 'position' in data:
                pos = data['position']
                vel_world = np.diff(pos, axis=0)
                print("Calculated velocity from 'position'")
        
        if vel_world is None:
            print("ERROR: Cannot calculate velocity - no position data found")
            return None
    
    # Convert to numpy array if not already
    vel_world = np.array(vel_world)
    print(f"Velocity shape: {vel_world.shape}")
    print(f"Velocity dtype: {vel_world.dtype}")
    
    # Handle edge cases
    if vel_world.size == 0:
        print("ERROR: Velocity array is empty")
        return None
    
    # Extract X and Y velocities (assuming first two dimensions are X and Y)
    if len(vel_world.shape) == 1:
        # If 1D, might be flattened or single dimension
        print("Warning: Velocity is 1D, assuming X velocity")
        vel_x = vel_world
        vel_y = None
    elif len(vel_world.shape) == 2:
        # Shape: (time_steps, dims) or (dims, time_steps)
        if vel_world.shape[0] > vel_world.shape[1]:
            # Likely (time_steps, dims)
            vel_x = vel_world[:, 0]
            vel_y = vel_world[:, 1] if vel_world.shape[1] > 1 else None
        else:
            # Likely (dims, time_steps)
            vel_x = vel_world[0, :]
            vel_y = vel_world[1, :] if vel_world.shape[0] > 1 else None
    elif len(vel_world.shape) == 3:
        # Shape: (batch, time_steps, dims) - flatten batch dimension
        vel_x = vel_world[:, :, 0].flatten()
        vel_y = vel_world[:, :, 1].flatten() if vel_world.shape[2] > 1 else None
    else:
        print(f"Warning: Unexpected velocity shape {vel_world.shape}, flattening")
        vel_flat = vel_world.flatten()
        # Try to split in half for X and Y
        mid = len(vel_flat) // 2
        vel_x = vel_flat[:mid]
        vel_y = vel_flat[mid:] if len(vel_flat) > mid else None
    
    # Calculate statistics
    print(f"\nX Velocity Statistics:")
    print(f"  Mean: {np.mean(vel_x):.6f}")
    print(f"  Std:  {np.std(vel_x):.6f}")
    print(f"  Min:  {np.min(vel_x):.6f}")
    print(f"  Max:  {np.max(vel_x):.6f}")
    print(f"  Samples: {len(vel_x)}")
    
    if vel_y is not None and len(vel_y) > 0:
        print(f"\nY Velocity Statistics:")
        print(f"  Mean: {np.mean(vel_y):.6f}")
        print(f"  Std:  {np.std(vel_y):.6f}")
        print(f"  Min:  {np.min(vel_y):.6f}")
        print(f"  Max:  {np.max(vel_y):.6f}")
        print(f"  Samples: {len(vel_y)}")
    
    # Interpretation
    mean_x = np.mean(vel_x)
    print(f"\nInterpretation:")
    if mean_x < -0.01:
        print(f"  X velocity mean is NEGATIVE ({mean_x:.6f}) - Expert was moving BACKWARD")
    elif mean_x > 0.01:
        print(f"  X velocity mean is POSITIVE ({mean_x:.6f}) - Expert was moving FORWARD")
    else:
        print(f"  X velocity mean is NEAR ZERO ({mean_x:.6f}) - No clear forward/backward bias")
    
    return {
        'vel_x': vel_x,
        'vel_y': vel_y,
        'mean_x': np.mean(vel_x),
        'std_x': np.std(vel_x),
        'mean_y': np.mean(vel_y) if vel_y is not None and len(vel_y) > 0 else None,
        'std_y': np.std(vel_y) if vel_y is not None and len(vel_y) > 0 else None,
    }

def main():
    workspace = Path("/home/eric/mujoco/Metamachine")
    
    v1_file = workspace / "v1_expert_data.pkl"
    batch_file = workspace / "batch_rollouts.pkl"
    
    print("Velocity Statistics Analysis")
    print("="*60)
    
    # Analyze v1_expert_data
    v1_stats = load_and_analyze_velocities(v1_file, "v1_expert_data.pkl")
    
    # Analyze batch_rollouts
    batch_stats = load_and_analyze_velocities(batch_file, "batch_rollouts.pkl")
    
    # Comparison
    if v1_stats and batch_stats:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"\nX Velocity Mean:")
        print(f"  v1_expert_data:    {v1_stats['mean_x']:+.6f} ± {v1_stats['std_x']:.6f}")
        print(f"  batch_rollouts:    {batch_stats['mean_x']:+.6f} ± {batch_stats['std_x']:.6f}")
        print(f"  Difference:        {batch_stats['mean_x'] - v1_stats['mean_x']:+.6f}")
        
        if v1_stats['mean_y'] is not None and batch_stats['mean_y'] is not None:
            print(f"\nY Velocity Mean:")
            print(f"  v1_expert_data:    {v1_stats['mean_y']:+.6f} ± {v1_stats['std_y']:.6f}")
            print(f"  batch_rollouts:    {batch_stats['mean_y']:+.6f} ± {batch_stats['std_y']:.6f}")
            print(f"  Difference:        {batch_stats['mean_y'] - v1_stats['mean_y']:+.6f}")

if __name__ == "__main__":
    main()
