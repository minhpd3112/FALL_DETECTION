#!/bin/bash

# Fall Detection Training Script for ProtoGCN
# Based on tools/dist_train.sh

cd /home/aidev/workspace/reid/bkai/Minh/FALL_DETECTION/ProtoGCN

# Activate ProtoGCN environment
source ~/miniconda3/bin/activate protogcn

# Config file
CONFIG="configs/fall_detection/j.py"

# Training command (1 GPU)
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=29500 \
    tools/train.py \
    $CONFIG \
    --validate \
    --test-last \
    --test-best \
    --launcher pytorch

echo ""
echo "=================================="
echo "Training completed!"
echo "=================================="
echo "Results saved in: work_dirs/fall_detection/j/"
echo ""
