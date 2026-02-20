import torch
import numpy as np

checkpoint = torch.load("./logs/transformer_multi_expert/final_model.pt", map_location="cpu")

model_config = checkpoint["model_config"]
state_mean = model_config.get("state_mean")
state_std = model_config.get("state_std")

print("=== Normalization stats stored in checkpoint ===\n")
if state_mean is not None:
    for key in sorted(state_mean.keys()):
        m = state_mean[key].numpy() if hasattr(state_mean[key], 'numpy') else np.array(state_mean[key])
        s = state_std[key].numpy() if hasattr(state_std[key], 'numpy') else np.array(state_std[key])
        print(f"  {key}:")
        print(f"    mean = {m}")
        print(f"    std  = {s}")
        print()

# Now show what v1 and v2 observations look like AFTER this normalization
import pickle

for path in ["v1_expert_data.pkl", "v2_expert_data.pkl"]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    traj = data["trajectories"][0]
    print(f"\n=== {path} normalized with combined stats ===")
    for mod in range(5):
        key = f"module{mod}"
        raw = traj[key][0]  # first timestep
        m = state_mean[key].numpy() if hasattr(state_mean[key], 'numpy') else np.array(state_mean[key])
        s = state_std[key].numpy() if hasattr(state_std[key], 'numpy') else np.array(state_std[key])
        normed = (raw - m) / s
        print(f"  {key} raw:    {raw}")
        print(f"  {key} normed: {normed}")
        print()
