#!/bin/bash
# Upload preparation script for BlockGCN to Google Colab
# This creates zip files optimized for Colab upload

echo "=========================================="
echo "BlockGCN Colab Upload Preparation"
echo "=========================================="
echo ""

cd /e/PYTHON/FALL_DETECTION

# Create upload directory
mkdir -p colab_upload
cd colab_upload

echo "📦 Step 1: Zipping BlockGCN code..."
# Zip BlockGCN excluding unnecessary files
zip -r BlockGCN.zip ../BlockGCN/ \
    -x "*/work_dir/*" \
    -x "*/.git/*" \
    -x "*/__pycache__/*" \
    -x "*.pyc" \
    -x "*/.gitignore"

echo "✅ BlockGCN.zip created ($(du -h BlockGCN.zip | cut -f1))"
echo ""

echo "📦 Step 2: Zipping processed data..."
# Zip processed data
zip -r processed_data.zip ../processed_data/

echo "✅ processed_data.zip created ($(du -h processed_data.zip | cut -f1))"
echo ""

echo "=========================================="
echo "✅ Upload preparation complete!"
echo "=========================================="
echo ""
echo "Files created in: e:/PYTHON/FALL_DETECTION/colab_upload/"
ls -lh

echo ""
echo "📤 Next steps:"
echo "1. Upload BlockGCN.zip to Google Drive/FALL_DETECTION/"
echo "2. Upload processed_data.zip to Google Drive/FALL_DETECTION/"
echo "3. Open train_blockgcn_colab.ipynb in VS Code"
echo "4. Select Colab kernel and run cells"
echo ""
