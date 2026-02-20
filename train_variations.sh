#!/bin/bash

# Script to train three different robot morphologies consecutively

# Define the configurations to train
# v2: Re-run after observation alignment
CONFIGS=("new_five_modules_v12" "new_five_modules_v13" "new_five_modules_v14" "new_five_modules_v15" "new_five_modules_v16" "new_five_modules_v17" "new_five_modules_v18" "new_five_modules_v19" "new_five_modules_v20" "new_five_modules_v21")

# Total timesteps per training run (increased to 1M for athletic experts)
TIMESTEPS=500000

# Path to the python executable (using the one provided in previous logs)
PYTHON_EXEC="/home/eric/miniconda3/envs/mujoco/bin/python"

# # Enable JAX Persistent Compilation Cache
# export JAX_COMPILATION_CACHE_DIR="/home/eric/mujoco/Metamachine/.jax_cache"
# export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
# mkdir -p $JAX_COMPILATION_CACHE_DIR

echo "=========================================================="
echo "Starting Batch Training for ${#CONFIGS[@]} Variations"
echo "=========================================================="

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "----------------------------------------------------------"
    echo "Current Configuration: $cfg"
    echo "Time: $(date)"
    echo "----------------------------------------------------------"
    
    # Run the training command
    $PYTHON_EXEC examples/train_rl_policy_sb3.py --config "$cfg" --timesteps $TIMESTEPS
    
    # Check if training succeeded
    if [ $? -eq 0 ]; then
        echo "Successfully completed training for $cfg"
    else
        echo "Error: Training failed for $cfg"
        # Optional: exit if one fails
        # exit 1
    fi
done

echo ""
echo "=========================================================="
echo "Batch Training Complete!"
echo "=========================================================="

