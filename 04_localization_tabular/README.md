# TBX11K-TB-Localization: Tuberculosis Lesion Detection with Swin Transformer FPN & Stratified Splitting

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Vision Transformer](https://img.shields.io/badge/Backbone-Swin--Base-blue.svg)](https://github.com/microsoft/Swin-Transformer)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

An accessible, self-contained deep learning pipeline for **pulmonary tuberculosis bounding box localization and lesion subtyping** on the **TBX11K** benchmark, featuring **stratified 4-category dataset splitting** and **class-agnostic lesion detection evaluation**.

---

## Key Highlights

1. **Simplified Tabular Workflow**: Directly processes tabular CSV annotations (`data.csv`) with dictionary-encoded bounding boxes, removing the requirement to convert to complex COCO/Pascal VOC formats.
2. **Clinical Stratification Scheme**: Solves sample imbalance and lesion co-occurrence by partitioning radiographs across 4 diagnostic disease categories:
   * **`No_TB`**: Healthy / non-tuberculous control radiographs.
   * **`Active_TB`**: Manifests exclusively active tuberculosis lesions.
   * **`Latent_TB`**: Manifests exclusively obsolete / latent tuberculosis calcifications.
   * **`Mixed_TB`**: Complex presentations containing *both* active and obsolete lesions in the same patient.
3. **Comprehensive Metric Suite**: Computes standard AP50 and AP75, alongside **class-agnostic TB detection** (measuring the model's ability to localize any TB abnormality regardless of exact sub-classification).

---

## Model Architecture

```
Input CXR (512 x 512)
          │
          ▼
┌────────────────────────────────────────────────────────┐
│  Swin Transformer Base (timm)                          │
│  - Stage 1 (H/8, W/8): 256 channels                    │
│  - Stage 2 (H/16, W/16): 512 channels                  │
│  - Stage 3 (H/32, W/32): 1024 channels                 │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│  Feature Pyramid Network (Torchvision FPN)             │
│  - Lateral Conv2d (256, 512, 1024 -> 256)              │
│  - Top-Down pathway with LastLevelMaxPool              │
│  - 4 Output scales: ['0', '1', '2', '3']               │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│  Faster R-CNN Detection Head                           │
│  - Anchor sizes: (32, 64, 128, 256)                    │
│  - 3-Class Classifier: Background, Active TB, Latent TB│
│  - Smooth L1 Box Regressor                             │
└────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
tbx11k-tb-localization/
├── models/
│   ├── __init__.py
│   └── swin_fpn.py           # SwinBackboneWithFPN and Faster R-CNN builder
├── data/
│   ├── __init__.py
│   ├── dataset.py            # Tabular TBX11K Dataset & collation
│   ├── split_dataset.py      # Stratified 4-category split generator
│   └── analyze_dataset.py    # Split and bounding-box distribution auditor
├── scripts/
│   └── train_slurm.sh        # HPC batch submission script
├── train.py                  # Training pipeline with checkpoint resumption
├── evaluate.py               # AP50, AP75, class-specific & class-agnostic eval
├── visualize.py              # Side-by-side Ground Truth vs Prediction visualizer
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/tbx11k-tb-localization.git
cd tbx11k-tb-localization

conda create -n tb_localization python=3.9 -y
conda activate tb_localization

pip install -r requirements.txt
```

---

## Data Preparation & Stratification

Ensure your dataset directory is structured as:
```
data_dir/
├── data.csv
└── images/
    ├── img_001.png
    └── img_002.png
```

### 1. Generate Stratified Splits
Generate balanced `train.txt`, `val.txt`, and `test.txt` split files:
```bash
python data/split_dataset.py \
    --csv-path /path/to/data.csv \
    --img-dir /path/to/images \
    --output-dir /path/to/data_dir \
    --test-size 600
```

### 2. Verify Split Distributions
```bash
python data/analyze_dataset.py \
    --csv-path /path/to/data.csv \
    --splits-dir /path/to/data_dir
```

---

## Training

Train the Faster R-CNN detector:
```bash
python train.py \
    --data-dir /path/to/data_dir \
    --checkpoint-dir ./checkpoints \
    --epochs 30 \
    --batch-size 8 \
    --lr 0.005 \
    --img-size 512
```

To resume from the latest saved checkpoint, append `--resume`.

---

## Evaluation & Metrics

Evaluate the test set across multiple IoU thresholds:
```bash
python evaluate.py \
    --data-dir /path/to/data_dir \
    --checkpoint ./checkpoints/checkpoint_epoch_30.pth \
    --split test
```

### Output Metric Matrix
* **Active TB AP@0.50 & AP@0.75**: Performance on cavitary, infiltrate, and active tuberculous consolidation.
* **Latent TB AP@0.50 & AP@0.75**: Performance on calcified nodules, fibrotic bands, and apical scarring.
* **Class-Agnostic AP**: Overall lesion localization ability independent of subtype diagnosis.

---

## Side-by-Side Qualitative Visualizer

Generate visual comparisons highlighting Ground Truth vs Predicted bounding boxes:
```bash
python visualize.py \
    --data-dir /path/to/data_dir \
    --checkpoint ./checkpoints/checkpoint_epoch_30.pth \
    --num-samples 10 \
    --score-thresh 0.3 \
    --output-dir ./visualizations
```

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
