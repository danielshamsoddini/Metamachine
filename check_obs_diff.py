import pickle
import numpy as np

# The key question: does the observation contain enough signal to distinguish?
# dof_pos is at index 6 in each module's 8-dim observation

# V1: dof_pos means ~ [0.09, -0.95, 1.22, 0.79, -0.93]
# V2: dof_pos means ~ [-2.58, 1.38, -3.37, -2.64, 1.23]

# These are totally different! The model should be able to distinguish.

# But let's check: what does the model see at t=0 when we initialize with v1's pose?
# init_joint_pos = [0, -1, 1, 1, -1] for v1 test

# The v1 training data dof_pos at t=0 is around those values
# The v2 training data dof_pos at t=0 is around [-3.1, 1.6, -3.4, -2.9, 1.2]

# So the observations ARE distinguishable. Let me check the gravity/gyro components 
# to see if the morphology difference is also visible.

for path in ["v1_expert_data.pkl", "v2_expert_data.pkl"]:
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Aggregate stats across multiple trajectories
    all_obs = {f"module{i}": [] for i in range(5)}
    all_actions = []
    
    for traj in data["trajectories"][:100]:  # first 100 trajectories
        for i in range(5):
            all_obs[f"module{i}"].append(traj[f"module{i}"])
        all_actions.append(traj["actions"])
    
    all_actions = np.concatenate(all_actions, axis=0)
    
    print(f"\n{'='*60}")
    print(f"=== {path} (100 trajectories) ===")
    print(f"{'='*60}")
    
    print(f"\nPer-module observation stats:")
    print(f"  {'Module':<10} {'proj_grav (3d)':<40} {'gyro (3d)':<40} {'dof_pos':<15} {'dof_vel':<15}")
    for i in range(5):
        obs = np.concatenate(all_obs[f"module{i}"], axis=0)
        pg = obs[:, :3].mean(axis=0)
        gy = obs[:, 3:6].mean(axis=0)
        dp = obs[:, 6].mean()
        dv = obs[:, 7].mean()
        print(f"  module{i:<4} [{pg[0]:6.3f},{pg[1]:6.3f},{pg[2]:6.3f}]"
              f"{'':>15} [{gy[0]:6.3f},{gy[1]:6.3f},{gy[2]:6.3f}]"
              f"{'':>15} {dp:8.3f}{'':>7} {dv:8.3f}")
    
    print(f"\n  Action stats: mean={all_actions.mean(axis=0)}")
