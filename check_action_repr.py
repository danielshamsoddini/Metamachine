import pickle
import numpy as np

# Check the action representations
for path in ["v1_expert_data.pkl", "v2_expert_data.pkl"]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    traj = data["trajectories"][0]
    print(f"\n=== {path} ===")
    print(f"  Keys per trajectory: {list(traj.keys())}")
    print(f"  Actions shape: {traj['actions'].shape}")
    print(f"  First 3 actions:")
    for i in range(3):
        print(f"    t={i}: {traj['actions'][i]}")
    print(f"  Action mean: {np.mean(traj['actions'], axis=0)}")
    print(f"  Action min: {np.min(traj['actions'], axis=0)}")
    print(f"  Action max: {np.max(traj['actions'], axis=0)}")
    
    # Check if actions are absolute positions or deltas
    diffs = np.diff(traj['actions'], axis=0)
    print(f"  Action diffs (first 3):")
    for i in range(3):
        print(f"    dt={i}: {diffs[i]}")
    print(f"  Mean abs diff: {np.mean(np.abs(diffs), axis=0)}")
