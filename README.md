# Swin-CXR-Suite: Deep Vision Transformers for Chest Radiograph Analysis

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Vision Transformer](https://img.shields.io/badge/Architecture-Swin--Transformer-blue.svg)](https://github.com/microsoft/Swin-Transformer)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A unified deep learning research toolkit applying **Swin Transformer** architectures across the full spectrum of **Chest Radiograph (CXR) Computer Vision tasks**:
1. **Multi-Label Thoracic Disease Classification** ([CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/))
2. **Tuberculosis Lesion Detection & Localization** ([TBX11K](https://www.kaggle.com/datasets/usmanshams/tbx-11))
3. **Anatomical Semantic Segmentation of Lungs & Heart** ([CheXmask](https://physionet.org/content/chexmask-cxr-segmentation/0.6.3/))

Developed and scaled on Arizona State University's **Sol Supercomputer** GPU cluster.

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Input ["Input Chest Radiograph"]
        IMG[Chest X-Ray Image]
    end

    subgraph Backbone ["Hierarchical Vision Transformer"]
        SWIN[Swin Transformer Backbone<br>Shifted-Window Self-Attention]
    end

    subgraph Task1 ["Module 01: Classification"]
        C1[Multi-Label Head] --> C2[Uncertainty Policies]
        C2 --> C3[Direct AUROC Maximization<br>LibAUC]
        C2 --> C4[Teacher-Student<br>Self-Training]
    end

    subgraph Task2 ["Module 02 & 04: Detection"]
        D1[Multi-Scale FPN Projections] --> D2[Anchor Generator]
        D2 --> D3[Faster R-CNN Detection Head]
        D3 --> D4[Active vs. Latent TB Bounding Boxes]
    end

    subgraph Task3 ["Module 03: Segmentation"]
        S1[Swin-Unet-V2 Decoder] --> S2[Patch Expanding & Skip Layers]
        S2 --> S3[4-Class Pixel Mask<br>Background, R-Lung, L-Lung, Heart]
    end

    IMG --> SWIN
    SWIN --> Task1
    SWIN --> Task2
    SWIN --> Task3
```

---

## Research Modules & Navigation

The repository is modularized into dedicated sub-projects, each with its own self-contained workflow and documentation:

| Directory | Task | Target Classes | Architecture | Highlights |
| :--- | :--- | :--- | :--- | :--- |
| **[`01_classification/`](./01_classification)** | Multi-Label Thoracic Pathology | 14 CheXpert Labels (5 Leaderboard) | Swin-Base (`224x224`) | U-Ones/U-Zeros policies, LibAUC surrogate AUROC loss, pseudo-label distillation |
| **[`02_detection/`](./02_detection)** | Tuberculosis Bounding Box Detection | Active TB vs. Obsolete TB | Swin-Base + Custom FPN + Faster R-CNN | Pure PyTorch / Torchvision implementation, standard COCO mAP & clinical FROC metrics |
| **[`03_segmentation/`](./03_segmentation)** | Anatomical Multi-Organ Segmentation | Right Lung, Left Lung, Heart | Swin-Unet-V2 | Multi-process RLE mask decoder, per-organ Dice tracking, failure case auditing |
| **[`04_localization_tabular/`](./04_localization_tabular)** | TB Lesion Localization (Tabular) | Active TB, Latent TB, Normal | Swin-FPN Faster R-CNN (`512x512`) | Stratified 4-category splitting (`No_TB`, `Active`, `Latent`, `Mixed`), class-agnostic AP |
| **[`05_detection_mmdet/`](./05_detection_mmdet)** | Detection Benchmarking (MMDetection) | Active TB, Obsolete TB | Swin-B Mask/Faster R-CNN | OpenMMLab MMDetection configurations and benchmark scripts |

---

## Experimental Highlights & Visualizations

### 1. Multi-Organ Segmentation Convergence (CheXmask)
<p align="center">
  <img src="assets/dice_vs_epoch.png" width="45%" alt="Validation Dice vs Epoch" />
  <img src="assets/loss_vs_epoch.png" width="45%" alt="Loss vs Epoch" />
</p>

### 2. Qualitative Mask Segmentation Outputs
<p align="center">
  <img src="assets/sample_predictions_best_model.png" width="92%" alt="Segmentation Visualizations" />
</p>

### 3. Classification AUROC Training Traces (CheXpert)
<p align="center">
  <img src="assets/auc_traces_chexpert.png" width="85%" alt="AUROC Traces across CheXpert Epochs" />
</p>

---

## Benchmark Performance Summary

Best checkpoint of each experiment. Full results (every curve, per-epoch table, slide and qualitative case) are in **[`results/`](./results/README.md)**, with every table also as CSV in [`results/tables/`](./results/tables).

| Experiment | Module | Model | Best result |
| :--- | :--- | :--- | :--- |
| CheXpert classification | [`01_classification`](./01_classification) | Swin-B (ImageNet-1K), U-Ignore | Mean 5-class test AUC **0.9004** (epoch 2) |
| TBX11K TB detection | [`04_localization_tabular`](./04_localization_tabular) | Faster R-CNN, Swin-B + FPN | FROC score **0.5930** (epoch 11); AP50 agnostic **0.6622** (epoch 6) |
| CheXmask segmentation | [`03_segmentation`](./03_segmentation) | Swin-Unet-V2 | Mean test Dice **0.9619** (epoch 11) |

### Classification (CheXpert test set, epoch 2)
| Atelectasis | Cardiomegaly | Consolidation | Edema | Pleural Effusion | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.8606 | 0.8697 | 0.8987 | 0.9165 | 0.9566 | **0.9004** |

### Lesion Detection (TBX11K, custom stratified test split)
| Epoch | AP50 agnostic | AP50 Active | AP50 Latent | AP75 agnostic | AP75 Active | AP75 Latent | FROC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 6 (best AP50) | **0.6622** | **0.6241** | 0.0763 | 0.1059 | 0.0902 | 0.0025 | 0.4991 |
| 11 (best FROC) | 0.6449 | 0.6114 | **0.1410** | 0.1239 | 0.1095 | 0.0095 | **0.5930** |
| 15 (best AP75) | 0.6332 | 0.6016 | 0.1249 | **0.1905** | **0.1754** | **0.0367** | 0.5817 |

### Semantic Segmentation (CheXmask test set, epoch 11)
| Background | Right Lung | Left Lung | Heart | Mean Dice |
| :--- | :--- | :--- | :--- | :--- |
| 0.9854 | 0.9616 | 0.9512 | 0.9494 | **0.9619** |

---

## Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/swin-cxr-suite.git
cd swin-cxr-suite
```

### 2. Create Environment
```bash
conda create -n swin_cxr python=3.9 -y
conda activate swin_cxr

# Install PyTorch matching your CUDA runtime (example CUDA 11.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install suite requirements:
pip install -r requirements.txt
```

---

## Quick Execution Guide

Each module can be executed independently from root or from within its subdirectory:

```bash
# 1. Run CheXpert multi-label classification training
python 01_classification/train_baseline.py --data-dir /path/to/chexpert

# 2. Run TBX11K native Faster R-CNN detection training
python 02_detection/train.py --data-dir /path/to/tbx11k

# 3. Run CheXmask Swin-Unet segmentation training
python 03_segmentation/train.py \
    --img-dir /path/to/chexpert \
    --mask-dir /path/to/chexmask-masks \
    --train-list /path/to/train_files.txt \
    --val-list /path/to/val_files.txt

# 4. Run TBX11K stratified tabular localization training
python 04_localization_tabular/train.py --data-dir /path/to/tbx11k-simplified
```

---

## Repository Structure

```text
swin-cxr-suite/
├── README.md                      # Master repository documentation
├── requirements.txt               # Unified project dependencies
├── .gitignore                     # Git tracking exclusions
│
├── 01_classification/             # CheXpert multi-label Vision Transformer
│   ├── self_training/             # Teacher-Student pseudo-labeling distillation
│   ├── train_baseline.py          # Supervised Swin-B training
│   ├── train_uncertainty.py       # Clinical uncertainty policies
│   ├── finetune_libauc.py         # Direct AUROC surrogate optimization
│   └── evaluate.py                # Multi-label AUROC evaluator
│
├── 02_detection/                  # TBX11K native Faster R-CNN detection
│   ├── models/                    # SwinBackboneFPN implementation
│   ├── datasets/                  # COCO dataset loader & collate
│   ├── tools/                     # Raw TBX11K to COCO converter
│   ├── train.py                   # PyTorch training loop
│   ├── evaluate.py                # COCO mAP and FROC evaluation
│   └── visualize.py               # Ground Truth vs Prediction visualizer
│
├── 03_segmentation/               # CheXmask multi-organ semantic segmentation
│   ├── models/                    # Swin-Unet-V2 architecture definitions
│   ├── data/                      # High-throughput RLE mask decoding tool
│   ├── analysis/                  # Failure mode and worst-case auditor
│   ├── train.py                   # Dice & CrossEntropy segmentation loop
│   └── evaluate.py                # Organ-wise Dice score evaluation
│
├── 04_localization_tabular/       # TBX11K stratified tabular localization
│   ├── models/                    # SwinBackboneWithFPN & Faster R-CNN
│   ├── data/                      # 4-category stratified split generator
│   ├── train.py                   # Training pipeline
│   ├── evaluate.py                # Class-specific and class-agnostic AP
│   └── visualize.py               # Side-by-side visualizer
│
├── 05_detection_mmdet/            # MMDetection TBX11K benchmark configs
│   ├── configs/                   # MMDetection Swin-B configs
│   └── tools/                     # Conversion and training helpers
│
└── assets/                        # Research figures, training curves & outputs
```

---

## Citation & References

```bibtex
@article{liu2021swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={ICCV},
  year={2021}
}

@article{cao2022swin,
  title={Swin-Unet: Unet-Like Pure Transformer for Medical Image Segmentation},
  author={Cao, Hu and Wang, Yueyue and Chen, Joy and Jiang, Dongsheng and Zhang, Xiaopeng and Tian, Qi and Wang, Manning},
  journal={ECCV},
  year={2022}
}

@inproceedings{yuan2021robust,
  title={Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification},
  author={Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao},
  booktitle={ACM SIGKDD},
  year={2021}
}
```
