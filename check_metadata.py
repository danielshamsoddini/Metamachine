import pickle
import os

path = "v2_expert_data.pkl"
if os.path.exists(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    if "_metadata" in data:
        print(f"Metadata: {data['_metadata']}")
else:
    print(f"File {path} not found.")
