# TBX11K-Swin-Faster-RCNN: Native PyTorch Tuberculosis Lesion Detection using Swin Transformer FPN

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Vision Transformer](https://img.shields.io/badge/Backbone-Swin--Base-blue.svg)](https://github.com/microsoft/Swin-Transformer)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

A standalone, pure **PyTorch and Torchvision** implementation of Faster R-CNN with a hierarchical **Swin Transformer Feature Pyramid Network (FPN)** for automated pulmonary tuberculosis detection and bounding box localization on the **TBX11K** benchmark.

---

## Why Native PyTorch?

Unlike heavy frameworks (such as MMDetection or Detectron2) which impose complex abstractions and fragile compilation chains, this repository provides:
* **Zero Heavy Dependencies**: Built purely on core `torch`, `torchvision`, and `timm`.
* **Complete Transparency**: Direct access to multi-scale RoI pooling, anchor generation, classification, and regression loss routines.
* **Clinical Evaluation Suite**: In addition to standard COCO mAP, it includes native **Free-Response ROC (FROC)** calculation tailored for radiological CAD systems.

---

## Results

Faster R-CNN, Swin-B + FPN, 512×512, AdamW, 16 epochs, custom stratified split (train 4,622 · val 1,800 · test 1,977).

| Epoch | AP50 agnostic | AP50 Active | AP50 Latent | AP75 agnostic | FROC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 6 (best AP50) | **0.6622** | **0.6241** | 0.0763 | 0.1059 | 0.4991 |
| 11 (best FROC) | 0.6449 | 0.6114 | **0.1410** | 0.1239 | **0.5930** |
| 15 (best AP75) | 0.6332 | 0.6016 | 0.1249 | **0.1905** | 0.5817 |

Active TB is detected well; Latent TB (73 training images) stays weak.

<p align="center">
  <img src="results/figures/exp2_froc_curve_epoch11.jpg" width="48%" alt="FROC curve at epoch 11" />
  <img src="results/figures/exp2_ap50_vs_epoch.jpg" width="48%" alt="AP50 per epoch" />
</p>

Loss, AP75, FROC-per-epoch, leaderboard reference and slides: [`results/README.md`](results/README.md).

---

## Architecture Overview

```
Input CXR (800 x 800)
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  Swin Transformer Backbone (timm)                      │
│  - Stage 0: 128 channels                               │
│  - Stage 1: 256 channels                               │
│  - Stage 2: 512 channels                               │
│  - Stage 3: 1024 channels                              │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  FPN Lateral Projections (Conv2d 1x1 -> 256-d)         │
│  - Multi-scale Feature Maps: ['0', '1', '2', '3']      │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  Torchvision Faster R-CNN Heads                        │
│  - AnchorGenerator: Sizes (32, 64, 128, 256)           │
│  - MultiScaleRoIAlign: Output size 7x7                 │
│  - Two-Class Classification: Active TB vs. Obsolete TB │
│  - Smooth L1 Bounding Box Regression                   │
└────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
tbx11k-swin-faster-rcnn/
├── models/
│   ├── __init__.py
│   └── swin_fpn.py            # SwinBackboneFPN and Faster R-CNN factory
├── datasets/
│   ├── __init__.py
│   └── tbx11k.py              # PyTorch Dataset & custom collate function
├── tools/
│   └── fix_coco_json.py       # Validates and patches COCO JSON metadata
├── scripts/
│   └── train_slurm.sh         # Cluster batch submission script
├── train.py                   # CLI training entrypoint
├── evaluate.py                # COCO mAP and FROC evaluation script
├── visualize.py               # Ground Truth vs. Prediction overlay tool
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/tbx11k-swin-faster-rcnn.git
cd tbx11k-swin-faster-rcnn

conda create -n swin_frcnn python=3.9 -y
conda activate swin_frcnn

pip install -r requirements.txt
```

---

## Dataset Format

This repository expects the **TBX11K** dataset in standard COCO format:
```
data_dir/
├── train/
│   └── img/
├── val/
│   └── img/
└── annotations/
    ├── train.json
    └── val.json
```

If your annotations are missing the COCO `"info"` dictionary, run:
```bash
python tools/fix_coco_json.py /path/to/annotations/*.json
```

---

## Usage Guide

### 1. Training
Train the model with configurable hyperparameters:
```bash
python train.py \
    --data-dir /path/to/tbx11k \
    --checkpoint-dir ./checkpoints \
    --epochs 30 \
    --batch-size 8 \
    --lr 0.005 \
    --img-size 800
```
To resume training from the latest saved checkpoint, simply append `--resume`.

### 2. Evaluation (mAP & FROC)
Evaluate a trained model checkpoint on the validation split:
```bash
python evaluate.py \
    --data-dir /path/to/tbx11k \
    --checkpoint ./checkpoints/checkpoint_epoch_30.pth \
    --split val
```
This outputs:
* **Standard COCO metrics**: `AP@[.5:.95]`, `AP50`, `AP75`
* **Clinical FROC metrics**: Average Sensitivity across `[0.5, 1.0, 2.0, 4.0, 8.0]` False Positives per Image (FPPI).

### 3. Visual Debugging & Qualitative Inspection
Generate comparison plots of Ground Truth (green) vs Predictions (red dashed with confidence score):
```bash
python visualize.py \
    --data-dir /path/to/tbx11k \
    --checkpoint ./checkpoints/checkpoint_epoch_30.pth \
    --num-samples 10 \
    --score-thresh 0.3 \
    --output-dir ./visualizations
```

---

## Clinical Relevance & Metrics

In radiological lesion detection, precision-recall alone can be misleading due to varying disease prevalence and multiple lesions per image. This project implements:
* **Free-Response ROC (FROC)**: Plots lesion detection sensitivity against the average number of false-positive detections per patient image.
* **Sensitivity @ Target FPPI**: Evaluates clinically acceptable operating points (e.g., 1 to 2 false alerts per radiograph).

---

## Citation

```bibtex
@article{liu2021swin,
  title={Swin transformer: Hierarchical vision transformer using shifted windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={ICCV},
  year={2021}
}
```
