#!/bin/bash

# Script to collect expert data from v2 and v13 for Transformer training

PYTHON_EXEC="/home/eric/miniconda3/envs/mujoco/bin/python"
SCRIPT="examples/batch_record_rollout.py"

echo "=========================================================="
echo "Starting Expert Data Collection"
echo "=========================================================="

# We'll use sed to temporarily change the LOG_DIR and OUTPUT in the python script
# or we can pass them as environment variables if the script supports it.
# Since the script uses global variables, I'll update the script for each run.

# Function to run collection
run_collection() {
    local log_dir=$1
    local output_file=$2
    local num_episodes=$3
    local config_name=$4
    
    # Update the script's global variables
    sed -i "s|^LOG_DIR = .*|LOG_DIR = \"$log_dir\"|" $SCRIPT
    sed -i "s|^OUTPUT = .*|OUTPUT = \"$output_file\"|" $SCRIPT
    sed -i "s|^NUM_EPISODES = .*|NUM_EPISODES = $num_episodes|" $SCRIPT
    sed -i "s|^CONFIG = .*|CONFIG = None|" $SCRIPT
    sed -i "s|^RECORDING_CONFIG = .*|RECORDING_CONFIG = None|" $SCRIPT
    sed -i "s|^USE_OPTIMIZED_POSE = .*|USE_OPTIMIZED_POSE = True|" $SCRIPT
    
    $PYTHON_EXEC $SCRIPT
}

# Run v1 (logs/20260203_172228m)
echo ""
echo "----------------------------------------------------------"
echo "Collecting data from v1 (logs/20260203_172228m)"
echo "----------------------------------------------------------"
run_collection "logs/20260203_172228m" "v1_expert_data.pkl" 8000 "modular_quadruped"

echo ""
echo "=========================================================="
echo "Data Collection Complete!"
echo "Check v1_expert_data.pkl"
echo "=========================================================="
