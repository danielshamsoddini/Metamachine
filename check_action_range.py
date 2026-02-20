import pickle
import numpy as np
import os

files = ["v1_expert_data.pkl", "v2_expert_data.pkl", "batch_rollouts.pkl"]

for path in files:
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        all_actions = []
        for traj in data["trajectories"]:
            all_actions.append(traj["actions"])
        
        all_actions = np.concatenate(all_actions, axis=0)
        print(f"\nAction statistics for {path}:")
        print(f"  Shape: {all_actions.shape}")
        print(f"  Min: {np.min(all_actions, axis=0)}")
        print(f"  Max: {np.max(all_actions, axis=0)}")
        print(f"  Mean: {np.mean(all_actions, axis=0)}")
        print(f"  Global Min: {np.min(all_actions):.4f}")
        print(f"  Global Max: {np.max(all_actions):.4f}")
    else:
        print(f"\nFile {path} not found.")
