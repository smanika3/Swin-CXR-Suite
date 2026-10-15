# Module 01: CheXpert Multi-Label Thoracic Pathology Classification & Self-Training

Multi-label classification of 14 chest radiographic observations on the **Stanford CheXpert** benchmark using the **Swin Transformer Base** architecture, clinical uncertainty policies, LibAUC direct AUROC maximization, and a Teacher-Student pseudo-labeling pipeline.

---

## Key Methodologies

1. **Vision Transformer Backbone**: Pre-trained `swin_base_patch4_window7_224` adapting shifted-window attention to capture diffuse, multi-scale thoracic abnormalities.
2. **Clinical Uncertainty Handling**: Maps ambiguous CheXpert label entries (`-1`) into deterministic binary targets per competition guidelines:
   * **Cardiomegaly**: U-Ones (`1.0`)
   * **Edema & Consolidation**: U-Ignore (`0.0`)
   * **Atelectasis**: U-Ones (`1.0`)
   * **Pleural Effusion**: U-Zeros (`0.0`)
3. **Direct AUROC Optimization (LibAUC)**: Fine-tunes with `MultiLabelAUCMLoss` to directly optimize the Area Under the ROC Curve rather than cross-entropy loss.
4. **Self-Training & Pseudo-Labeling**: Teacher model infers soft probabilities on unlabeled CXRs to train student networks.

---

## Directory Layout

```
01_classification/
├── self_training/
│   ├── teacher_train.py           # Trains teacher network with uncertainty policies
│   ├── teacher_predict.py         # Generates soft pseudo-labels on training split
│   ├── student_train.py           # Distills pseudo-labels into student model
│   └── student_train_thresh.py    # Distills confident pseudo-labels above threshold
├── train_baseline.py              # Supervised baseline Swin-Base training
├── train_uncertainty.py           # Uncertainty-guided training
├── finetune_libauc.py             # Stage-2 direct AUROC optimization
├── evaluate.py                    # Multi-label AUROC calculation per class
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Baseline Training
```bash
python train_baseline.py \
    --data-dir /path/to/chexpert \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-4
```

### 2. Uncertainty Policy Training
```bash
python train_uncertainty.py \
    --data-dir /path/to/chexpert \
    --epochs 30 \
    --batch-size 32
```

### 3. LibAUC AUROC Maximization
```bash
python finetune_libauc.py \
    --data-dir /path/to/chexpert \
    --checkpoint checkpoints_uncertainty/checkpoint_epoch_10.pth \
    --epochs 10 \
    --lr 1e-5
```

### 4. Evaluation
```bash
python evaluate.py \
    --data-dir /path/to/chexpert \
    --checkpoint checkpoints/checkpoint_epoch_30.pth
```
