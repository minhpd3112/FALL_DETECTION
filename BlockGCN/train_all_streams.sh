#!/bin/bash

# Train all 4 streams for Fall Detection
# Following BlockGCN ensemble methodology

echo "======================================"
echo "4-Stream Ensemble Training"
echo "======================================"
echo ""

# Stream 1: Joint
echo "🔵 [1/4] Training Joint Stream..."
python main.py \
    --config config/fall-detection/default.yaml \
    --phase train \
    --save-score True \
    --device 0 \
    --log-interval 10

echo ""
echo "✅ Joint stream completed!"
echo ""

# Stream 2: Bone
echo "🟢 [2/4] Training Bone Stream..."
python main.py \
    --config config/fall-detection/bone.yaml \
    --phase train \
    --save-score True \
    --device 0 \
    --log-interval 10

echo ""
echo "✅ Bone stream completed!"
echo ""

# Stream 3: Velocity
echo "🟡 [3/4] Training Velocity Stream..."
python main.py \
    --config config/fall-detection/vel.yaml \
    --phase train \
    --save-score True \
    --device 0 \
    --log-interval 10

echo ""
echo "✅ Velocity stream completed!"
echo ""

# Stream 4: Bone-Velocity
echo "🔴 [4/4] Training Bone-Velocity Stream..."
python main.py \
    --config config/fall-detection/bone_vel.yaml \
    --phase train \
    --save-score True \
    --device 0 \
    --log-interval 10

echo ""
echo "✅ Bone-Velocity stream completed!"
echo ""

echo "======================================"
echo "✅ All 4 streams trained!"
echo "======================================"
echo ""
echo "Results saved in:"
echo "  - work_dir/fall_detection/joint/"
echo "  - work_dir/fall_detection/bone/"
echo "  - work_dir/fall_detection/velocity/"
echo "  - work_dir/fall_detection/bone_velocity/"
echo ""
echo "Next: Run ensemble.py to combine predictions"
echo ""
