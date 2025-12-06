#!/bin/bash

# ProtoGCN 6-Stream Ensemble Training for Fall Detection
# Train all 6 modalities: j, b, bm, jm, k, km

cd /home/aidev/workspace/reid/bkai/Minh/FALL_DETECTION/ProtoGCN

# Activate ProtoGCN environment
source ~/miniconda3/bin/activate protogcn

echo "======================================"
echo "ProtoGCN 6-Stream Ensemble Training"
echo "======================================"
echo ""

# Array of modalities
MODALITIES=("j" "b" "bm" "jm" "k" "km")
NAMES=("Joint" "Bone" "Bone-Motion" "Joint-Motion" "K-Bone" "K-Bone-Motion")

# Train each stream
for i in {0..5}; do
    mod=${MODALITIES[$i]}
    name=${NAMES[$i]}
    
    echo ""
    echo "======================================"
    echo "[$((i+1))/6] Training $name Stream ($mod)"
    echo "======================================"
    echo ""
    
    python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=$((29500+i)) \
        tools/train.py \
        configs/fall_detection/${mod}.py \
        --validate \
        --test-last \
        --test-best \
        --launcher pytorch
    
    echo ""
    echo "✅ $name stream completed!"
    echo ""
done

echo ""
echo "======================================"
echo "✅ All 6 streams trained!"
echo "======================================"
echo ""
echo "Results saved in:"
for mod in "${MODALITIES[@]}"; do
    echo "  - work_dirs/fall_detection/$mod/"
done
echo ""
echo "Next: Run ensemble script to combine predictions"
echo ""
