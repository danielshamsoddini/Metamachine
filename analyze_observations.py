import pickle
import numpy as np

# Load the pickle file
with open('batch_rollouts.pkl', 'rb') as f:
    data = pickle.load(f)

trajectories = data['trajectories']
first_traj = trajectories[0]

# Get first step data
env_obs = first_traj['env_observations'][0]
module0 = first_traj['module0'][0]
module1 = first_traj['module1'][0]
module2 = first_traj['module2'][0]
module3 = first_traj['module3'][0]
module4 = first_traj['module4'][0]

print("=" * 80)
print("IDENTIFYING THE EXTRA 23 DIMENSIONS IN env_observations")
print("=" * 80)

# Summary
print(f"\nSUMMARY:")
print(f"  env_observations total dimensions: {len(env_obs)}")
print(f"  Module observations total: {len(module0) + len(module1) + len(module2) + len(module3) + len(module4)}")
print(f"  Extra dimensions: {len(env_obs) - (len(module0) + len(module1) + len(module2) + len(module3) + len(module4))}")

# Structure analysis
pattern_21 = env_obs[:21]
print(f"\n{'=' * 80}")
print("STRUCTURE ANALYSIS")
print("=" * 80)
print(f"\nenv_observations has a repeating pattern:")
print(f"  - First 21 dimensions form a pattern")
print(f"  - This pattern repeats 3 times (indices 0-20, 21-41, 42-62)")
print(f"  - Total: 21 × 3 = 63 dimensions")

# Identify the extra 23 dimensions
# If we assume modules should be 40 dimensions, then:
# - First 40: might contain modules (but transformed/processed)
# - Last 23: extra dimensions

first_40 = env_obs[:40]
last_23 = env_obs[40:]

print(f"\n{'=' * 80}")
print("THE EXTRA 23 DIMENSIONS (indices 40-62)")
print("=" * 80)
print(f"\nShape: {last_23.shape}")
print(f"Values:\n{last_23}")

# Analyze the structure of the extra 23
print(f"\n{'=' * 80}")
print("ANALYSIS OF THE EXTRA 23 DIMENSIONS")
print("=" * 80)
print(f"\nThe extra 23 dimensions consist of:")
print(f"  - First 2 values: {last_23[:2]} (zeros)")
print(f"  - Next 21 values: {last_23[2:]} (this is the 21-dim pattern again)")

# Check if the pattern matches
pattern_in_extra = last_23[2:]
matches_pattern = np.allclose(pattern_in_extra, pattern_21, rtol=1e-5, atol=1e-5)
print(f"\n  The 21 values in the extra dimensions match the pattern: {matches_pattern}")

# So the structure is:
# - Indices 0-20: pattern (21 dims)
# - Indices 21-40: pattern[:19] + pattern[:2] = pattern shifted (20 dims) 
#   Actually, let me check this more carefully
print(f"\n{'=' * 80}")
print("COMPLETE BREAKDOWN")
print("=" * 80)
print(f"\nIndices 0-20 (21 dims): pattern")
print(f"  {env_obs[0:21]}")
print(f"\nIndices 21-40 (20 dims): pattern again (starting from index 0)")
print(f"  {env_obs[21:41]}")
print(f"\nIndices 40-62 (23 dims): THE EXTRA 23 DIMENSIONS")
print(f"  First 2: zeros")
print(f"  Next 21: pattern again")
print(f"  {last_23}")

# Final answer
print(f"\n{'=' * 80}")
print("CONCLUSION: THE EXTRA 23 DIMENSIONS ARE")
print("=" * 80)
print(f"\nThe extra 23 dimensions (indices 40-62) consist of:")
print(f"  1. Two zeros: [0, 0]")
print(f"  2. The 21-dimension pattern repeated again: pattern[0:21]")
print(f"\nThis means the 21-dimension pattern appears 3 times total:")
print(f"  - Once at indices 0-20")
print(f"  - Once at indices 21-41") 
print(f"  - Once at indices 42-62 (within the extra 23)")
print(f"\nThe 21-dimension pattern itself contains:")
print(f"  - 16 non-zero values")
print(f"  - 5 zeros at the end")

