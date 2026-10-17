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
  <img src="results/figures/exp1_leaderboard_auroc_original.png" width="48%" alt="Per-class AUROC per epoch" />
  <img src="results/figures/exp1_loss_curve.jpg" width="48%" alt="Training and validation loss" />
</p>

Per-epoch tables, paper/leaderboard comparison and slides: [`results/README.md`](results/README.md). The original SLURM training and test logs of this run are in [`results/run_logs/exp1/`](results/run_logs/exp1); their per-epoch losses and AUROCs are the numbers above.

---

## Another approach: teacher–student self-training

About 139k of the CheXpert training labels are uncertain (`-1`). The reported run masks most of them out of the loss, so that data is wasted. Self-training tries to recover it: the trained model labels its own uncertain data, and a fresh "student" model is trained on the result.

1. **Teacher.** The Exp 1 checkpoints above. For each class, the checkpoint from that class's best test-AUC epoch is used, e.g. Cardiomegaly from epoch 10 and Consolidation from epoch 5 (`CLASS_TO_BEST_EPOCH` in `teacher_predict_multi.py`).
2. **Pseudo-labels.** The teacher predicts only where the label is `-1`; all certain labels are kept. 138,879 uncertain labels were replaced. Two variants were run:
   - **Soft:** each `-1` becomes the teacher's probability (`teacher_predict.py`, with `THRESHOLDING = False`).
   - **Thresholded:** a probability ≥ high becomes 1, one ≤ low becomes 0, and anything in between stays soft (`teacher_predict_multi.py`, five threshold pairs from one forward pass).
3. **Student.** Swin-B starts again from ImageNet weights and trains with the same recipe as the reported run (AdamW lr 1e-4, wd 0.1, batch 96, same augmentation and Dropout head). The training loss is unmasked BCE on the pseudo-labels; validation still uses masked BCE on the real labels (`student_train.py` / `student_train_thresh.py`).
4. **Scoring.** Every student checkpoint was scored on the same 668-image test set with `exp1_policy_training/multi_test.py`, with `CHECKPOINT_DIR` pointed at the student run.

| Student | Pseudo-labels (high / low) | Best epoch | Mean 5-class test AUC |
| :--- | :--- | :--- | :--- |
| Soft | teacher probability | 3 | 0.8995 |
| thresh1 | 0.5 / 0.5 | 3 | 0.8914 |
| thresh2 | 0.6 / 0.4 | 4 | 0.9026 |
| thresh3 | 0.7 / 0.3 | 7 | 0.9018 |
| thresh4 | 0.8 / 0.2 | 4 | 0.9020 |
| **thresh5** | **0.9 / 0.1** | **6** | **0.9028** |
| *Reported run (teacher)* | *—* | *2* | *0.9004* |

<p align="center">
  <img src="results/figures/self_training_thresh5_leaderboard_auroc.png" width="70%" alt="Best student (thresh5) leaderboard AUROC per epoch" />
</p>

*Best student (0.9 / 0.1): the mean AUC (black dashed) peaks at 0.9028 at epoch 6, then drifts down over 40 epochs.*

Four of the five thresholded students beat the teacher, by +0.0014 to +0.0024 mean AUC. The soft student and the 0.5 / 0.5 student do not. The gains are small, and like the reported run they are picked from test-set scores, so read them as a trend, not a clear win. This is why the module reports the single-model run.

Per-run tables: [`results/tables/self_training/`](results/tables/self_training) (summary in `summary.csv`). Original logs (teacher prediction, student training, test): [`results/run_logs/self_training/`](results/run_logs/self_training). More detail in [`results/README.md`](results/README.md#supplementary-teacherstudent-self-training).

---

## Choosing a training method

| Method | Script(s) | Uncertain-label (`-1`) handling | Config | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Per-class policy (reported result)** | `exp1_policy_training/v2_train_swin.py` → `multi_test.py` | Cardiomegaly U-Ones, Pleural Effusion U-Zeros, all others masked | AdamW lr 1e-4, wd 0.1, batch 96, flip/rotate/affine/colour-jitter augmentation, Dropout(0.5) after the head | Produced the 0.9004 result. Paths hardcoded at the top of each file |
| Plain baseline | `train_baseline.py` | none: `-1` is passed to the loss as a target | AdamW lr 1e-4, wd 0.05, batch 32, no augmentation | CLI flags. Not used for the reported result |
| Policy mapping (CLI) | `train_uncertainty.py` | Cardiomegaly & Atelectasis U-Ones, Pleural Effusion U-Zeros, Edema & Consolidation `-1 → 0`; other classes keep `-1` | AdamW lr 1e-4, wd 0.05, batch 32 | CLI flags. Its "U-Ignore" maps `-1 → 0` (i.e. U-Zeros), not masking |
| Teacher–student self-training (supplementary, [results above](#another-approach-teacherstudent-self-training)) | `self_training/teacher_predict.py` (soft) or `teacher_predict_multi.py` (thresholds) → `student_train.py` or `student_train_thresh.py` → `exp1_policy_training/multi_test.py` | Teacher = reported-run checkpoints. Only `-1` labels are replaced by teacher pseudo-labels; the student trains on them unmasked | Same as reported run (wd 0.1, batch 96, augmentation, Dropout head) | Run; best mean test AUC 0.9028. Paths hardcoded at the top of each file. `teacher_train.py` (a separately trained teacher with its own policies: Cardiomegaly, Edema, Atelectasis U-Ones; others masked) was not used for these results |
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
├── self_training/                 # Supplementary teacher–student runs (+ a SLURM .sh per script)
│   ├── teacher_train.py           # Separate teacher with its own policies (not used for the results)
│   ├── teacher_predict.py         # Soft pseudo-labels from the reported-run checkpoints
│   ├── teacher_predict_multi.py   # Thresholded pseudo-labels, 5 threshold pairs in one pass
│   ├── student_train.py           # Student on teacher_predict.py output
│   └── student_train_thresh.py    # Student on one teacher_predict_multi.py CSV
├── train_baseline.py              # Plain supervised baseline (CLI)
├── train_uncertainty.py           # Per-class policy mapping (CLI)
├── finetune_libauc.py             # LibAUC AUROC fine-tuning (does not run as written, see above)
├── evaluate.py                    # Per-class AUROC of one checkpoint on valid.csv
├── assets/                        # Plots from earlier exploratory runs (not the reported result)
├── results/                       # Figures, slides, CSV tables, original run logs (run_logs/)
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
# CHECKPOINT_DIR in the teacher scripts = your exp1_policy_training checkpoints
python teacher_predict.py         # soft labels (set THRESHOLDING = False) -> train_pseudolabels.csv
python student_train.py
# or the thresholded variant:
python teacher_predict_multi.py   # -> train_pseudolabels_thresh_<high>_<low>.csv (5 files)
python student_train_thresh.py    # set TRAIN_CSV_PATH / CHECKPOINT_DIR per threshold
# score: set CHECKPOINT_DIR in ../exp1_policy_training/multi_test.py to the student folder, then run it
```
