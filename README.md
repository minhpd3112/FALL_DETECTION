# Fall Detection - BlockGCN & ProtoGCN

Skeleton-based Fall Detection using state-of-the-art Graph Convolutional Networks.

## 🎯 Project Overview

This project implements fall detection using two cutting-edge GCN models:
- **BlockGCN** (CVPR 2024) - Efficient topology-aware action recognition
- **ProtoGCN** (CVPR 2025) - Prototypical learning for fine-grained action distinction

## 📊 Dataset

- **Classes**: 2 (Fall, No Fall)
- **Keypoints**: COCO 17 format
- **Samples**: 2,046 total (1,437 train / 304 val / 305 test)
- **Window Size**: 30 frames

## 🚀 Quick Start with Google Colab

### Prerequisites

1. VS Code with Google Colab extension installed
2. Google account
3. This repository cloned

### Training on Colab (Free GPU!)

1. **Open notebook**: `train_blockgcn_colab.ipynb` in VS Code
2. **Select Kernel**: Choose "Colab" → "T4 GPU"
3. **Run All Cells**: Training takes ~45 minutes

See [COLAB_SETUP_README.md](COLAB_SETUP_README.md) for detailed instructions.

## 📁 Project Structure

```
FALL_DETECTION/
├── BlockGCN/                   # BlockGCN model implementation
│   ├── config/fall-detection/  # Training configs
│   ├── model/                  # Model architecture
│   └── feeders/                # Data loaders
├── ProtoGCN/                   # ProtoGCN model implementation
├── processed_data/             # Preprocessed skeleton data
│   ├── X_train.npy
│   ├── y_train.npy
│   └── ...
├── train_blockgcn_colab.ipynb # Colab training notebook
└── COLAB_SETUP_README.md      # Setup guide
```

## 🔧 Local Training (if you have GPU)

### BlockGCN

```bash
cd BlockGCN
pip install -e torchlight/
bash train_all_streams.sh  # Train all 4 streams
```

### ProtoGCN

```bash
cd ProtoGCN
conda env create -f protogcn.yaml
conda activate protogcn
bash train_fall.sh
```

## 📈 Expected Results

- **Accuracy**: 80-95% on test set
- **Training Time**: 
  - BlockGCN (1 stream): ~30-45 min on T4 GPU
  - BlockGCN (4 streams): ~2-3 hours

## 📚 References

- [BlockGCN Paper (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_BlockGCN_Redefine_Topology_Awareness_for_Skeleton-Based_Action_Recognition_CVPR_2024_paper.pdf)
- [ProtoGCN Paper (CVPR 2025)](https://arxiv.org/abs/2411.18941)

## 📝 License

See individual model directories for their respective licenses.
