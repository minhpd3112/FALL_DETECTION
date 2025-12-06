# Hướng Dẫn Kết Nối VS Code với Google Colab

## 🎯 Mục Tiêu
Train BlockGCN trên GPU miễn phí của Google Colab, code từ VS Code local.

---

## ✅ Phương Pháp Khuyến Nghị: Official Google Colab Extension

### Bước 1: Cài Đặt Extension

1. Mở VS Code
2. Nhấn `Ctrl+Shift+X` (Extensions)
3. Tìm **"Google Colab"**
4. Install extension chính thức từ Google
5. Install **"Jupyter"** extension (sẽ tự động prompt)

### Bước 2: Chuẩn Bị Code

**Option A: Sử dụng GitHub (Khuyến nghị)**

```bash
# 1. Tạo repo mới trên GitHub
# 2. Push code lên
cd e:/PYTHON/FALL_DETECTION
git init
git add BlockGCN/ processed_data/
git commit -m "Initial commit for Colab"
git remote add origin https://github.com/YOUR_USERNAME/FALL_DETECTION.git
git push -u origin main
```

**Option B: Upload lên Google Drive**

```bash
# Windows (Git Bash)
cd e:/PYTHON/FALL_DETECTION
bash prepare_colab_upload.sh

# Upload 2 files .zip vào Google Drive:/FALL_DETECTION/
# - BlockGCN.zip
# - processed_data.zip
```

### Bước 3: Train với Colab

1. **Mở notebook**: `train_blockgcn_colab.ipynb` trong VS Code
2. **Select Kernel**: Click góc phải trên → Chọn "Colab"
3. **New Colab Server**: Đăng nhập Google
4. **Chọn GPU**: T4 (Free) hoặc A100 (Pro)
5. **Chạy cells**: Nhấn `Shift+Enter` để run từng cell

### Cell-by-Cell Workflow

```python
# Cell 1: Mount Drive + Check GPU ✅
# Cell 2: Upload code (chọn Git clone hoặc Drive copy) ✅
# Cell 3: Install dependencies ✅
# Cell 4: Verify data ✅
# Cell 5: TRAIN! 🚀
# Cell 7: Save results về Drive 💾
```

---

## 📊 Expected Results

### Training Time (T4 GPU - Free)
- Joint stream: **30-45 phút** (100 epochs)
- All 4 streams: **2-3 giờ**

### Accuracy
- Expected: **80-95%** on test set
- File output: `work_dir/fall_detection/joint/runs-XXXX/`

### Files to Download
```
work_dir/
├── fall_detection/
│   └── joint/
│       └── runs-20241206-XXXX/
│           ├── epoch_best.pt      # Best checkpoint
│           ├── config.yaml        # Config used
│           └── log.txt            # Training logs
```

---

## 🔧 Troubleshooting

### Q: "No GPU detected"?
**A**: Runtime → Change runtime type → Hardware accelerator: **T4 GPU**

### Q: Colab timeout sau 12h?
**A**: 
- Train ít hơn 12h (100 epochs ~45 phút OK)
- Hoặc upgrade Colab Pro ($9.99/tháng, 24h runtime)
- Save checkpoints thường xuyên về Drive

### Q: Import error "No module named 'torchlight'"?
**A**: Chạy lại Cell 3 (Install dependencies)

### Q: Data not found?
**A**: Check Cell 2, đảm bảo file paths đúng:
```python
!ls /content/FALL_DETECTION/BlockGCN
!ls /content/FALL_DETECTION/processed_data
```

---

## 💡 Tips

1. **Kiểm tra GPU usage**: 
   ```python
   !nvidia-smi
   ```

2. **Monitor training**: 
   ```python
   # Trong notebook, logs sẽ hiện real-time
   # Hoặc check file log
   !tail -f work_dir/fall_detection/joint/runs-XXXX/log.txt
   ```

3. **Resume training nếu disconnect**:
   - Modify config: thêm `resume_from: path/to/checkpoint.pt`
   - Re-run Cell 5

4. **Tối ưu batch size** nếu Out of Memory:
   - Edit `config/fall-detection/default.yaml`
   - Giảm `batch_size: 32` → `16`

---

## 📦 Files Created

| File | Purpose |
|------|---------|
| `train_blockgcn_colab.ipynb` | Main training notebook |
| `prepare_colab_upload.sh` | Zip code for upload |
| This README | Quick reference guide |

---

## 🚀 Quick Start (TL;DR)

```bash
# 1. Cài Colab extension trong VS Code
# 2. Prepare code
cd e:/PYTHON/FALL_DETECTION
bash prepare_colab_upload.sh

# 3. Upload .zip lên Drive hoặc push lên GitHub
# 4. Mở train_blockgcn_colab.ipynb
# 5. Select Kernel → Colab → T4 GPU
# 6. Run all cells!
```

**Thời gian tổng**: ~5 phút setup + 45 phút training = **~50 phút** để có model trained! 🎉

---

## 📞 Hỗ Trợ

- Official Colab Extension: https://marketplace.visualstudio.com/items?itemName=Google.colab
- Colab Docs: https://colab.research.google.com/
- BlockGCN Paper: https://openaccess.thecvf.com/content/CVPR2024/
