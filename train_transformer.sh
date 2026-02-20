#!/bin/bash

# Script to train Transformer using expert data from v1 + v2
# Clean training from scratch with action normalization

PYTHON_EXEC="/home/eric/miniconda3/envs/mujoco/bin/python"
TRAIN_SCRIPT="CapyFormer/examples/train_transformer_debug.py"

# Paths to data files - using both v1 and v2 expert data
V1_DATA="v1_expert_data.pkl"
V2_DATA="v2_expert_data.pkl"

LOG_DIR="./logs/transformer_clean_test"

echo "=========================================================="
echo "Starting Clean Transformer Training (1000 epochs)"
echo "=========================================================="

# Check if files exist
FILES_TO_USE=""
for f in $V1_DATA $V2_DATA; do
    if [ -f "$f" ]; then
        echo "Found data file: $f"
        FILES_TO_USE="$FILES_TO_USE $f"
    else
        echo "Warning: Data file not found: $f"
    fi
done

if [ -z "$FILES_TO_USE" ]; then
    echo "Error: No data files found. Please collect data first."
    exit 1
fi

# Clean start: remove old log.csv to avoid appending to stale data
if [ -f "$LOG_DIR/log.csv" ]; then
    echo "Removing old log.csv to ensure clean training log"
    rm "$LOG_DIR/log.csv"
fi

# Train from scratch with action normalization (1000 epochs)
# No --resume-checkpoint: fully fresh weights
echo ""
echo "Running training script with files: $FILES_TO_USE"
echo "Action normalization: ENABLED"
echo "Epochs: 1000"
echo "Log dir: $LOG_DIR"
$PYTHON_EXEC $TRAIN_SCRIPT \
    --rollout-path $FILES_TO_USE \
    --n-epochs 1000 \
    --batch-size 32 \
    --log-dir $LOG_DIR \
    --normalize-actions

echo ""
echo "=========================================================="
echo "Training complete!"
echo "Logs and models saved to $LOG_DIR"
echo "=========================================================="
