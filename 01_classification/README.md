# Module 01: CheXpert Multi-Label Thoracic Pathology Classification & Self-Training

Multi-label classification of 14 chest radiographic observations on the **Stanford CheXpert** benchmark with a **Swin Transformer Base** backbone. This module holds several training methods that were tried during the project. They are kept as separate scripts rather than merged, so pick the one that matches what you want to run (see [Choosing a training method](#choosing-a-training-method)).

---

## Results

Reported run: [`exp1_policy_training/`](exp1_policy_training). Swin-B (`swin_base_patch4_window7_224`, timm pretrained weights), 14 outputs, masked BCE + AdamW (lr 1e-4, weight decay 0.1), batch 96, 18 epochs. Uncertain (`-1`) labels: Cardiomegaly → 1 (U-Ones), Pleural Effusion → 0 (U-Zeros), every other class masked out of the loss (U-Ignore). Best checkpoint: epoch 2, mean 5-class test AUC **0.9004** (CheXpert paper U-Ignore: 0.8892).

| Atelectasis | Cardiomegaly | Consolidation | Edema | Pleural Effusion | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.8606 | 0.8697 | 0.8987 | 0.9165 | 0.9566 | **0.9004** |

Every epoch's checkpoint was scored on the 668-image test set by `exp1_policy_training/multi_test.py`, and the best epoch was picked from those test scores (there is no separate validation-based selection).

<p align="center">
  <img src="results/figures/exp1_auroc_per_epoch.jpg" width="48%" alt="Per-class AUROC per epoch" />
  <img src="results/figures/exp1_loss_curve.jpg" width="48%" alt="Training and validation loss" />
</p>

Per-epoch tables, paper/leaderboard comparison and slides: [`results/README.md`](results/README.md).

---

## Choosing a training method

| Method | Script(s) | Uncertain-label (`-1`) handling | Config | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Per-class policy (reported result)** | `exp1_policy_training/v2_train_swin.py` → `multi_test.py` | Cardiomegaly U-Ones, Pleural Effusion U-Zeros, all others masked | AdamW lr 1e-4, wd 0.1, batch 96, flip/rotate/affine/colour-jitter augmentation, Dropout(0.5) after the head | Produced the 0.9004 result. Paths hardcoded at the top of each file |
| Plain baseline | `train_baseline.py` | none: `-1` is passed to the loss as a target | AdamW lr 1e-4, wd 0.05, batch 32, no augmentation | CLI flags. Not used for the reported result |
| Policy mapping (CLI) | `train_uncertainty.py` | Cardiomegaly & Atelectasis U-Ones, Pleural Effusion U-Zeros, Edema & Consolidation `-1 → 0`; other classes keep `-1` | AdamW lr 1e-4, wd 0.05, batch 32 | CLI flags. Its "U-Ignore" maps `-1 → 0` (i.e. U-Zeros), not masking |
| Teacher–student self-training | `self_training/teacher_train.py` → `teacher_predict.py` → `student_train.py` or `student_train_thresh.py` | Teacher: Cardiomegaly, Edema, Atelectasis U-Ones; Consolidation and others masked; Pleural Effusion `U-MultiClass` (passed through, masked). Student trains on teacher pseudo-labels | Same as reported run (wd 0.1, batch 96, augmentation, Dropout head) | Paths hardcoded at the top of each file; see the comments there for how the files connect |
| LibAUC AUROC fine-tuning | `finetune_libauc.py` | from the loaded checkpoint | PESG + `MultiLabelAUCMLoss` | **Does not run as written**: its `MultiLabelAUCMLoss(num_classes=…)` / `PESG(model, …)` calls match no libauc release (1.2 lacks the loss; ≥1.3 uses `num_labels` and `PESG(params, …)`) |

`evaluate.py` scores one checkpoint on `valid.csv` (234 images). It builds the model without the Dropout wrapper, so it loads checkpoints from `train_baseline.py` / `train_uncertainty.py` only; use `exp1_policy_training/multi_test.py` for the reported-run and self-training checkpoints.

---

## Data layout and paths

CheXpert CSVs list images as `CheXpert-v1.0/train/patientXXXXX/studyY/viewZ.jpg` (or `.../valid/...`). Every script strips that `CheXpert-v1.0/train/` / `CheXpert-v1.0/valid/` prefix and joins the rest onto the image folder. If your download uses a different prefix (for example the Kaggle **CheXpert-v1.0-small** copy uses `CheXpert-v1.0-small/train/`), change the prefix on the line marked `# PATH NOTE` in the script you run.

Scripts without CLI flags (`exp1_policy_training/`, `self_training/`) have their paths in a block marked `# EDIT BEFORE RUNNING` at the top of the file. Set those to your own folders; the defaults are the author's ASU Sol cluster paths. The `.sh` files are SLURM job scripts for ASU Sol (module names, QoS, conda env) and need adapting for other clusters.

All views (frontal and lateral) in the CSVs are used; no script filters by view.

---

## Directory Layout

```
01_classification/
├── exp1_policy_training/          # Reported run (exact code that produced the results)
│   ├── v2_train_swin.py           # Per-class policy training, masked BCE
│   ├── multi_test.py              # Scores every checkpoint on the test set, writes per-epoch AUROC CSV + plots
│   ├── v2_train_swin.sh           # SLURM job (ASU Sol)
│   └── multi-test.sh              # SLURM job (ASU Sol)
├── self_training/
│   ├── teacher_train.py           # Trains the teacher with uncertainty policies
│   ├── teacher_predict.py         # Writes soft / thresholded pseudo-labels for the training split
│   ├── student_train.py           # Trains a student on the pseudo-labels
│   └── student_train_thresh.py    # Student on confidence-thresholded pseudo-labels
├── train_baseline.py              # Plain supervised baseline (CLI)
├── train_uncertainty.py           # Per-class policy mapping (CLI)
├── finetune_libauc.py             # LibAUC AUROC fine-tuning (does not run as written, see above)
├── evaluate.py                    # Per-class AUROC of one checkpoint on valid.csv
├── assets/                        # Plots from earlier exploratory runs (not the reported result)
├── results/                       # Reported-run figures, slides, CSV tables
├── requirements.txt
└── README.md
```

---

## Quick Start

### Reported run (per-class policy)
Edit the paths at the top of both files, then:
```bash
cd exp1_policy_training
python v2_train_swin.py        # writes checkpoint_epoch_*.pth; the reported run stopped after 18 epochs
python multi_test.py           # per-epoch test AUROC CSV + plots; pick the best epoch from it
```

### Baseline / policy mapping (CLI)
```bash
python train_baseline.py --data-dir /path/to/chexpert --epochs 30 --batch-size 32 --lr 1e-4
python train_uncertainty.py --data-dir /path/to/chexpert --epochs 30 --batch-size 32
python evaluate.py --data-dir /path/to/chexpert --checkpoint checkpoints/checkpoint_epoch_30.pth
```

### Teacher–student self-training
Edit the paths at the top of each file, then run in order:
```bash
cd self_training
python teacher_train.py
python teacher_predict.py      # writes train_pseudolabels.csv
python student_train.py        # or student_train_thresh.py (see the comment in that file about its input CSV)
```
