#!/bin/bash

# Fall Detection Training Script for BlockGCN

# Config path
CONFIG="config/fall-detection/default.yaml"

# Training command
python main.py \
    --config $CONFIG \
    --phase train \
    --save-score True \
    --device 0 \
    --log-interval 10

echo ""
echo "=================================="
echo "Training completed!"
echo "=================================="
echo "Results saved in: work_dir/fall_detection/joint/"
echo ""
