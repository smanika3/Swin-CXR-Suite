#!/bin/bash
# Environment setup for OpenMMLab MMDetection with Swin Transformer backbone

set -e

echo "=== Setting up MMDetection Environment ==="

# 1. Install PyTorch matching your CUDA version
# conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# 2. Install OpenMMLab Package Manager
pip install -U openmim

# 3. Install MMEngine, MMCV, and MMDetection via MIM
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"

# 4. Additional dependencies
pip install timm tqdm scikit-learn

echo "=== Environment Ready ==="
