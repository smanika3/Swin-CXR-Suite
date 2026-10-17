# CheXmask-Swin-Unet: Anatomical Semantic Segmentation of Lungs & Heart

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Vision Transformer](https://img.shields.io/badge/Architecture-Swin--Unet--V2-blue.svg)](https://github.com/HuCaoFighting/Swin-Unet)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

An end-to-end deep learning framework for **pixel-level multi-organ anatomical semantic segmentation** on chest radiographs (CXRs), trained on the **CheXmask** database using **Swin-Unet-V2**.

---

## Overview

Accurate boundary segmentation of cardiothoracic structures is fundamental for automated cardiomegaly assessment (cardiothoracic ratio), lung disease quantification, and spatial conditioning.

### Target Anatomical Classes
* **Class 0**: Background
* **Class 1**: Right Lung
* **Class 2**: Left Lung
* **Class 3**: Heart

---

## Quantitative Results & Training Dynamics

Best checkpoint: epoch 11 of 11, CheXmask test set (55,978 images).

| Anatomical Structure | Test Dice Coefficient |
| :--- | :--- |
| **Background** | 0.9854 |
| **Right Lung** | 0.9616 |
| **Left Lung** | 0.9512 |
| **Heart** | 0.9494 |
| **Mean Dice** | **0.9619** |

Per-epoch Dice and loss tables, best/worst test cases and the setup are in [`results/README.md`](results/README.md).

<p align="center">
  <img src="assets/dice_vs_epoch.png" width="45%" alt="Test Dice vs Epoch" />
  <img src="assets/loss_vs_epoch.png" width="45%" alt="Loss vs Epoch" />
</p>

### Qualitative Predictions

<p align="center">
  <img src="assets/sample_predictions_best_model.png" width="90%" alt="Sample Predictions on CheXmask" />
</p>

---

## Model Architecture

```
Input CXR (224 x 224 x 3)
           │
           ▼
┌──────────────────────────────────────────────┐
│  Swin-Unet Encoder (Swin Transformer Blocks) │
│  - Patch Partition: 4x4 patches              │
│  - Stage 1: Linear Embedding -> Swin Blocks  │
│  - Stage 2: Patch Merging (H/8, W/8, 2C)     │
│  - Stage 3: Patch Merging (H/16, W/16, 4C)   │
│  - Stage 4: Patch Merging (H/32, W/32, 8C)   │
└──────────────────────────────────────────────┘
           │
      Skip │ Connections
           ▼
┌──────────────────────────────────────────────┐
│  Swin-Unet Decoder (Patch Expanding)         │
│  - Patch Expanding: 2x upsampling            │
│  - Skip concatenation with encoder features  │
│    (concat + linear projection)              │
│  - Swin blocks (shifted-window self-attn)    │
└──────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│  Segmentation Head (Conv2d 1x1 -> 4 Classes) │
│  [Background, Right Lung, Left Lung, Heart]  │
└──────────────────────────────────────────────┘
```

---

## Repository Structure

```
03_segmentation/
├── models/
│   ├── __init__.py
│   ├── swin_unet_v2.py          # Swin-Unet-V2 architecture (SwinTransformerSys), used by all scripts
│   └── vision_transformer.py    # Unused reference wrapper from upstream Swin-Unet (needs scipy + a config object)
├── data/
│   ├── __init__.py
│   ├── prep_mask.py             # CheXmask RLE -> PNG masks (RCA >= 0.7 filter) + image lists
│   ├── split_train_test.py      # Image-level 70/30 train/test split of train_files.txt
│   └── dataset.py               # PyTorch CheXmask dataset loader
├── analysis/
│   ├── __init__.py
│   └── min_max_analysis.py      # Plots the 3 best and 3 worst test predictions
├── original_run/                # Exact scripts of the reported run (hardcoded ASU Sol paths)
│   ├── train_swin_seg.py / .sh  # Training
│   └── test.py / test.sh        # Scores every checkpoint on the test set: Dice/loss plots, sample predictions
├── assets/                      # Evaluation curves and sample predictions
├── results/                     # Results extracted from the final presentation
├── scripts/
│   └── train_slurm.sh           # SLURM training job (ASU Sol)
├── train.py                     # Training (AMP, CrossEntropy loss, Dice metric on validation)
├── evaluate.py                  # Test set Dice evaluation
├── requirements.txt
└── README.md
```

---

## Installation & Setup

```bash
git clone https://github.com/smanika3/Swin-CXR-Suite.git
cd Swin-CXR-Suite/03_segmentation

conda create -n chexmask_env python=3.9 -y
conda activate chexmask_env

pip install -r requirements.txt
```

All commands below are run from `03_segmentation/`.

---

## Data

1. **CheXpert** (Stanford AIMI): download the full-resolution CheXpert-v1.0 images. You need the `train/` and `valid/` folders
   (`train/patient00001/study1/view1_frontal.jpg`, ...).
2. **CheXmask** (PhysioNet, credentialed access required):
   <https://physionet.org/content/chexmask-cxr-segmentation/>.
   The reported run used **v1.0.0**, file `Preprocessed/CheXpert.csv`
   (`chexmask-cxr-segmentation-data/1.0.0/Preprocessed/CheXpert.csv`).
   Columns used: `Path, Height, Width, Right Lung, Left Lung, Heart, Dice RCA (Mean)`.

The commands below use these placeholders:

```bash
CHEXPERT=/path/to/CheXpert-v1.0          # contains train/ and valid/
CHEXMASK_CSV=/path/to/chexmask-cxr-segmentation-data/1.0.0/Preprocessed/CheXpert.csv
MASKS=/path/to/chexmask-masks            # output folder
```

---

## Pipeline

Setup of the reported run:

* Masks with `Dice RCA (Mean)` < 0.7 are dropped (CheXmask's quality estimate).
* Train/test: **image-level** random 70/30 split of the CheXpert *train* images
  (`train_test_split(test_size=0.3, random_state=42)`). This is **not** patient-independent:
  images of the same patient can be in both train and test.
* Validation: the CheXpert `valid/` images that have a CheXmask mask (199 images).
* AdamW (lr 1e-4, weight decay 0.05), CrossEntropy loss, AMP, batch size 96, 11 epochs, 224×224 input.
* Reported **Mean Dice = mean over all 4 classes including background** (0.9619, test set, epoch 11).
  The scripts also print the 3-organ mean (right lung, left lung, heart).

### 1. Decode CheXmask masks

Writes `$MASKS/train/`, `$MASKS/valid/`, `$MASKS/train_files.txt` and `$MASKS/valid_files.txt`.
The script prints how many rows the RCA filter dropped and how many rows had no matching image.

```bash
python data/prep_mask.py \
    --csv-path $CHEXMASK_CSV \
    --train-img-dir $CHEXPERT/train \
    --val-img-dir $CHEXPERT/valid \
    --output-dir $MASKS \
    --min-rca 0.7 \
    --num-workers 16
```

### 2. Split train into train/test (70/30)

```bash
python data/split_train_test.py \
    --input $MASKS/train_files.txt \
    --train-out $MASKS/train_split.txt \
    --test-out $MASKS/test_files.txt
```

### 3. Train

```bash
python train.py \
    --img-dir $CHEXPERT/train \
    --mask-dir $MASKS/train \
    --train-list $MASKS/train_split.txt \
    --val-img-dir $CHEXPERT/valid \
    --val-mask-dir $MASKS/valid \
    --val-list $MASKS/valid_files.txt \
    --checkpoint-dir ./checkpoints \
    --epochs 11 \
    --batch-size 96 \
    --lr 1e-4
```

Each epoch writes `checkpoints/checkpoint_epoch_<N>.pth`; `checkpoints/best_model.pth` holds the epoch with the best
validation 3-organ mean Dice. Add `--resume` to continue from the latest checkpoint (optimizer, AMP scaler and the best
validation Dice are restored). On ASU Sol: `mkdir -p logs && sbatch scripts/train_slurm.sh`.

### 4. Evaluate on the test split

The test images live in CheXpert `train/` (they come from the split in step 2).
The reported numbers are from the epoch-11 checkpoint:

```bash
python evaluate.py \
    --img-dir $CHEXPERT/train \
    --mask-dir $MASKS/train \
    --test-list $MASKS/test_files.txt \
    --checkpoint ./checkpoints/checkpoint_epoch_11.pth \
    --batch-size 96
```

Dice is pooled over the pixels of each batch and averaged weighted by batch size (as in the original test script),
so use batch size 96 to reproduce the reported numbers.

### 5. Best / worst cases

Ranks test images by per-image 3-organ mean Dice and plots the 3 best and 3 worst
(red = right lung, green = left lung, blue = heart) to `./analysis_results/best_worst_predictions.png`.
To diagnose model limitations (e.g., severe pleural effusion obscuring lung bases, cardiomegaly boundary bleeding):

```bash
python analysis/min_max_analysis.py \
    --img-dir $CHEXPERT/train \
    --mask-dir $MASKS/train \
    --test-list $MASKS/test_files.txt \
    --checkpoint ./checkpoints/checkpoint_epoch_11.pth \
    --output-dir ./analysis_results
```

---

## Citation & Acknowledgments

```bibtex
@article{cao2022swin,
  title={Swin-unet: Unet-like pure transformer for medical image segmentation},
  author={Cao, Hu and Wang, Yueyue and Chen, Joy and Jiang, Dongsheng and Zhang, Xiaopeng and Tian, Qi and Wang, Manning},
  journal={ECCV},
  year={2022}
}

@article{chexmask2023,
  title={CheXmask: Anatomical Segmentation for CheXpert Chest Radiographs},
  journal={PhysioNet},
  year={2023}
}
```
