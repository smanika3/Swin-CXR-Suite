# TBX11K-Swin-MMDetection: Tuberculosis Lesion Detection using Swin Transformer & MMDetection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![MMDetection](https://img.shields.io/badge/MMDetection-3.0+-blue.svg)](https://github.com/open-mmlab/mmdetection)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../../LICENSE)

An open-source deep learning framework for **automated pulmonary tuberculosis (TB) detection and bounding box localization** on chest radiographs (CXRs), built on top of the OpenMMLab **MMDetection** suite with a **Swin Transformer** hierarchical backbone.

---

## Overview

Tuberculosis remains one of the world's deadliest infectious diseases. Automated chest radiograph triage can substantially accelerate clinical diagnosis. This project leverages the **TBX11K** benchmark to detect and distinguish between:
1. **Active Tuberculosis (`ActiveTuberculosis`)**
2. **Obsolete / Latent Pulmonary Tuberculosis (`ObsoletePulmonaryTuberculosis`)**

### Key Features
* **Hierarchical Vision Transformer**: Employs a pre-trained **Swin-Base** backbone (`swin_base_patch4_window7_224_22k`) featuring shifted-window multi-head self-attention.
* **Feature Pyramid Network (FPN)**: Combines multi-resolution feature maps across stages `[128, 256, 512, 1024]` projected to 256-channel representations.
* **COCO Data Integration**: Complete end-to-end tooling to translate raw TBX11K annotations into standard COCO format.
* **HPC & SLURM Ready**: Fully compatible with cluster deployment, mixed-precision acceleration, and OpenMMLab's `mim` tooling.

---

## Model Architecture

```
Input CXR (1333 x 800)
         │
         ▼
┌──────────────────────────────────────────────┐
│  Swin-B Transformer Backbone                 │
│  - Patch Partition & Linear Embedding        │
│  - Stage 1: Embed Dim 128, Depth 2           │
│  - Stage 2: Embed Dim 256, Depth 2           │
│  - Stage 3: Embed Dim 512, Depth 18          │
│  - Stage 4: Embed Dim 1024, Depth 2          │
└──────────────────────────────────────────────┘
         │  Multi-Scale Feature Maps (C2, C3, C4, C5)
         ▼
┌──────────────────────────────────────────────┐
│  Feature Pyramid Network (FPN)               │
│  - Lateral 1x1 convolutions                  │
│  - Top-down pathway + 3x3 anti-aliasing      │
│  - Outputs: P2, P3, P4, P5, P6 (256-d)       │
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  Region Proposal Network (RPN) & RoI Head    │
│  - Multi-scale anchor matching               │
│  - RoIAlign pooling                          │
│  - Bounding Box Classification & Regression  │
└──────────────────────────────────────────────┘
```

---

## Repository Structure

```
tbx11k-swin-mmdetection/
├── configs/                          # MMDetection configuration files
│   ├── tbx11k_swin_frcnn.py          # Primary experiment config (Swin-B + FPN)
│   ├── coco_detection.py             # Base dataset loader configs
│   ├── schedule_1x.py                # Optimizer & LR scheduling
│   └── default_runtime.py            # Logging & runtime hooks
├── tools/
│   └── convert_coco.py               # Robust TBX11K to COCO JSON converter
├── scripts/
│   ├── setup_env.sh                  # Automated environment setup
│   └── train_slurm.sh                # SLURM GPU job submission template
├── requirements.txt                  # Python dependencies
├── .gitignore
└── README.md
```

---

## Installation & Setup

### 1. Prerequisites
Ensure you have Python >= 3.8 and PyTorch >= 2.0 with CUDA support installed:
```bash
conda create -n swin_mmdet python=3.9 -y
conda activate swin_mmdet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install OpenMMLab Packages
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"
pip install -r requirements.txt
```

---

## Dataset Preparation

1. Download the **TBX11K** dataset from the official repository or Kaggle.
2. Structure your raw dataset as follows:
   ```
   tbx11k/
   ├── train/
   │   ├── img/   # CXR image files (.png)
   │   └── ann/   # Annotation JSON files (.json)
   ├── val/
   │   ├── img/
   │   └── ann/
   └── test/
       ├── img/
       └── ann/
   ```
3. Convert raw annotations to COCO format using the included utility:
   ```bash
   python tools/convert_coco.py \
       --dataset-root /path/to/tbx11k \
       --output-dir /path/to/tbx11k/annotations
   ```
   This generates `train.json`, `val.json`, and `test.json` under `annotations/`.

---

## Training

### Single GPU Training
```bash
mim train mmdet configs/tbx11k_swin_frcnn.py \
    --work-dir ./work_dirs/tbx_swin_exp1
```

### Multi-GPU Distributed Training
```bash
mim train mmdet configs/tbx11k_swin_frcnn.py \
    --launcher pytorch \
    --gpus 4 \
    --work-dir ./work_dirs/tbx_swin_distributed
```

### Resume Training from Checkpoint
```bash
mim train mmdet configs/tbx11k_swin_frcnn.py \
    --resume-from ./work_dirs/tbx_swin_exp1/epoch_10.pth
```

### SLURM HPC Training
Submit the batch script directly to a cluster scheduler:
```bash
sbatch scripts/train_slurm.sh
```

---

## Evaluation & Testing

Evaluate model performance on the test set using standard COCO bounding box metrics (mAP, AP50, AP75):
```bash
mim test mmdet configs/tbx11k_swin_frcnn.py \
    --checkpoint ./work_dirs/tbx_swin_exp1/best_coco_bbox_mAP_epoch_*.pth \
    --eval bbox
```

---

## Citation & References

```bibtex
@article{liu2021swin,
  title={Swin transformer: Hierarchical vision transformer using shifted windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={ICCV},
  year={2021}
}

@article{chen2019mmdetection,
  title={MMDetection: Open mmlab detection toolbox and benchmark},
  author={Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wanyi and Liu, Ziwei and Xu, Jiarui and others},
  journal={arXiv preprint arXiv:1906.07155},
  year={2019}
}
```
