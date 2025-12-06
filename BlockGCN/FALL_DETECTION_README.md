# BlockGCN for Fall Detection

## ✅ Setup Complete!

### Files adapted for Fall Detection:

1. **feeders/feeder_fall.py** - Custom data loader for Fall dataset
2. **graph/coco.py** - COCO 17 keypoints graph structure  
3. **config/fall-detection/train_joint.yaml** - Training configuration
4. **train_fall.sh** - Training script

---

## 📊 Dataset Info

- **Classes**: 2 (Fall, No Fall)
- **Keypoints**: 17 (COCO format)
- **Training samples**: 1,421
- **Validation samples**: 304
- **Test samples**: 305
- **Frames per sample**: 30

---

## 🚀 How to Train

### 1. Install dependencies (if not already installed)

```bash
cd /home/aidev/workspace/reid/bkai/Minh/FALL_DETECTION/BlockGCN
conda activate reid
pip install -e torchlight
pip install pyyaml
```

### 2. Train the model

```bash
./train_fall.sh
```

Or manually:

```bash
python main.py \
    --config config/fall-detection/train_joint.yaml \
    --phase train \
    --save-score True \
    --device 0 \
    --log-interval 10
```

### 3. Monitor training

Results will be saved in: `work_dir/fall_detection/joint/`

---

## ⚙️ Configuration Details

**Model Parameters:**
- num_class: 2 (Fall/No Fall)
- num_point: 17 (COCO keypoints)
- num_person: 1
- window_size: 30 frames

**Training Hyperparameters:**
- batch_size: 32
- num_epoch: 100
- base_lr: 0.01
- lr_decay_rate: 0.1
- weight_decay: 0.0004

---

## 📝 Evaluation

After training, evaluate on test set:

```bash
python main.py \
    --config config/fall-detection/train_joint.yaml \
    --phase test \
    --save-score True \
    --weights work_dir/fall_detection/joint/runs-XX-XXXX/epoch_best.pt \
    --device 0
```

Replace XX-XXXX with your run timestamp.

---

## 🔧 Troubleshooting

### If you get import errors:

```bash
cd /home/aidev/workspace/reid/bkai/Minh/FALL_DETECTION/BlockGCN
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

### If torchlight is not installed:

```bash
pip install -e torchlight/
```

---

## 📈 Expected Performance

With 2,030 samples and proper training:
- Expected accuracy: 80-95% (depending on data quality and training)
- Training time: ~10-30 minutes (depends on GPU)

---

## 🎯 Next Steps After Training

1. Check accuracy in `work_dir/fall_detection/joint/`
2. If accuracy is low:
   - Try data augmentation
   - Adjust learning rate
   - Train for more epochs
3. If accuracy is good:
   - Test on real videos
   - Deploy for camera system

---

Good luck with training! 🚀
