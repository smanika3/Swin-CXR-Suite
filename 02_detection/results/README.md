# Results: Swin + Faster R-CNN on TBX11K

Results extracted from the *Final Presentation* (Sai Manikandan's slides only). Graphs, tables and prediction images come straight from the presentation; every table is also a CSV in [`tables/`](tables/). Statements marked *Observation* are my own reading of a chart or table, not something the slide says.

**Metric reminders:** AUC and Dice: higher is better; loss: lower is better. AP50 / AP75 = average precision at IoU 0.5 / 0.75. FROC score = average sensitivity at up to 1 false positive per image (FPPI).

Folder layout: `slides/` full slides (PNG) · `figures/` original graphs/images (JPEG) · `tables/` CSV.


*Tuberculosis lesion detection — slides 55–64*

## Setup (slides 56–57)

| Item | Value |
|---|---|
| Task | Object detection (localisation) of tuberculosis lesions in chest X-rays; images fall in 4 categories: No_TB, Active_TB, Latent_TB, Mixed_TB |
| Split | Stratified split; minimum quotas for rare classes in the test set (at least 20 Latent_TB and 5 Mixed_TB). Train 4,622 · Val 1,800 · Test 1,977. Val is the official TBX11K validation set; train and test are split from the official training set, which is what the category table below counts |
| Model | Faster R-CNN, Swin Base backbone (ImageNet pretrained, features from stages 1–3), FPN with 256 output channels |
| RPN | Anchor sizes 32/64/128/256 px; aspect ratios 0.5/1.0/2.0 |
| Heads / input | 3 classes (background, Active TB, Latent TB); images resized to 512×512 |
| Training | AdamW · learning rate 1e-4 · batch size 8 · multi-task loss (RPN classification + regression, ROI classification + regression) · 30 epochs listed on slide 57 (result slides report 16) |
| Thresholds | Training proposal 0.05 · inference confidence 0.5 · IoU 0.5 |
| Best result | FROC score **0.5930** at epoch 11 |

Dataset split by category (slide 56):

| Category | Total | Train | Test |
|---|---|---|---|
| No_TB | 6000 | 4200 (70.0%) | 1800 (30.0%) |
| Active_TB | 473 | 332 (70.2%) | 141 (29.8%) |
| Latent_TB | 103 | 73 (70.9%) | 30 (29.1%) |
| Mixed_TB | 23 | 17 (73.9%) | 6 (26.1%) |

CSV: [`exp2_tbx11k_dataset_split.csv`](tables/exp2_tbx11k_dataset_split.csv)

Slide images: [task & dataset](slides/slide056_exp2_setup_task_dataset.png) · [model & training](slides/slide057_exp2_setup_model_training.png)

**Summary.** The detector localises **Active TB** well (AP50 about 0.60–0.62 at its best epochs) but struggles with **Latent TB**
(AP50 only 0.08–0.14 at the three standout epochs, AP75 near zero), the rare class. Metrics peak at different epochs: AP50 at epoch 6,
FROC at epoch 11, AP75 at epoch 15. Epoch 11 is the headline because it has the best FROC score. Validation loss rises after epoch 14, a sign of overfitting.

## 2.1 Training and validation loss (slide 58)

![Exp 2 loss curve](slides/slide058_exp2_loss_curve.png)

*Original figure: [`exp2_loss_curve.jpg`](figures/exp2_loss_curve.jpg)*

- Training loss falls from 0.0315 to about 0.008–0.009, with a visible drop at epoch 11.
- Validation loss is noisy and lowest at epoch 11 (0.0183), then jumps to 0.0287 (epoch 15) and 0.028 (epoch 16).
- *Observation:* training loss keeps improving while validation loss worsens at the end, i.e. overfitting after epoch 14.

## 2.2 Loss table (slide 59)

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 0.0315 | 0.0211 |
| 2 | 0.0208 | 0.0247 |
| 3 | 0.0207 | 0.0213 |
| 4 | 0.0172 | 0.0252 |
| 5 | 0.0156 | 0.0192 |
| 6 | 0.0145 | 0.0191 |
| 7 | 0.0156 | 0.0228 |
| 8 | 0.0141 | 0.0226 |
| 9 | 0.0148 | 0.0216 |
| 10 | 0.0143 | 0.0225 |
| **11** | **0.01** | **0.0183** |
| 12 | 0.01 | 0.019 |
| 13 | 0.0097 | 0.0196 |
| 14 | 0.0079 | 0.0195 |
| 15 | 0.0082 | 0.0287 |
| 16 | 0.0093 | 0.028 |

Epoch 11 (bold; highlighted yellow on the slide) has the lowest validation loss. CSV: [`exp2_tbx11k_loss.csv`](tables/exp2_tbx11k_loss.csv) · slide: [PNG](slides/slide059_exp2_loss_table.png)

## 2.3 FROC score vs epoch (slide 60)

![Exp 2 FROC vs epoch](slides/slide060_exp2_froc_vs_epoch.png)

*Original figure: [`exp2_froc_vs_epoch.jpg`](figures/exp2_froc_vs_epoch.jpg) · full-resolution PNG from the test job: [`exp2_froc_vs_epoch_original.png`](figures/exp2_froc_vs_epoch_original.png)*

- FROC rises from 0.24 (epoch 1) to about 0.55 by epoch 5, then oscillates between roughly 0.42 and 0.59.
- Best: **0.5930 at epoch 11**; second best 0.5817 at epoch 15.
- The large epoch-to-epoch swings mean no single epoch should be read as a stable level.

## 2.4 FROC curve at the best epoch (slide 61)

![Exp 2 FROC curve epoch 11](slides/slide061_exp2_froc_curve_epoch11.png)

*Original figure: [`exp2_froc_curve_epoch11.jpg`](figures/exp2_froc_curve_epoch11.jpg) · full-resolution PNG from the test job, with the score in its title: [`exp2_froc_curve_epoch11_original.png`](figures/exp2_froc_curve_epoch11_original.png)*

- Sensitivity climbs steeply at low false-positive rates: about 0.6 at roughly 0.07 FPPI and about 0.75 at roughly 0.3 FPPI (read from the chart).
- The plotted curve ends at about 0.79 sensitivity near 0.53 FPPI; the x-axis extends to 2.0 but no points are plotted beyond that.
- FROC score = average sensitivity up to FPPI 1.0 = **0.5930**.

## 2.5 AP50 vs epoch (slide 62)

![Exp 2 AP50](slides/slide062_exp2_ap50_vs_epoch.png)

*Original figure: [`exp2_ap50_vs_epoch.jpg`](figures/exp2_ap50_vs_epoch.jpg) · full-resolution PNG from the test job: [`exp2_ap50_vs_epoch_original.png`](figures/exp2_ap50_vs_epoch_original.png)*

- Category-agnostic and Active TB track each other and peak at epoch 6 (0.6622 and 0.6241), with sharp dips at epochs 8 and 10.
- Latent TB stays far lower (0.004–0.18 across epochs; 0.0763 at epoch 6, 0.1410 at epoch 11), trending up late in training.
- *Observation:* the Active-vs-Latent gap is the main weakness. Latent TB has only 73 training images vs 332 for Active TB (slide 56).

## 2.6 AP75 vs epoch (slide 63)

![Exp 2 AP75](slides/slide063_exp2_ap75_vs_epoch.png)

*Original figure: [`exp2_ap75_vs_epoch.jpg`](figures/exp2_ap75_vs_epoch.jpg) · full-resolution PNG from the test job: [`exp2_ap75_vs_epoch_original.png`](figures/exp2_ap75_vs_epoch_original.png)*

- AP75 is much lower than AP50 (best about 0.19 vs 0.66): objects are found but boxes are often not tightly placed.
- Category-agnostic AP75 peaks at epoch 15 (0.1905); Active TB peaks around epoch 12 (about 0.19) and is 0.1754 at epoch 15.
- Latent TB AP75 is close to zero throughout (max about 0.047 at epoch 5).

## 2.7 Metric comparison at the best epochs (slide 64)

![Exp 2 metric comparison](slides/slide064_exp2_metric_comparison.png)

| Epoch | AP50 TB-agnostic | AP50 Active TB | AP50 Latent TB | AP75 TB-agnostic | AP75 Active TB | AP75 Latent TB | FROC score | Best AP50 | Best AP75 | Best FROC |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 0.6622 | 0.6241 | 0.0763 | 0.1059 | 0.0902 | 0.0025 | 0.4991 | yes |  |  |
| **11** | **0.6449** | **0.6114** | **0.1410** | **0.1239** | **0.1095** | **0.0095** | **0.5930** |  |  | **yes** |
| 15 | 0.6332 | 0.6016 | 0.1249 | 0.1905 | 0.1754 | 0.0367 | 0.5817 |  | yes |  |

CSV: [`exp2_tbx11k_metrics_best_epochs.csv`](tables/exp2_tbx11k_metrics_best_epochs.csv)

Leaderboard reference row shown on the slide (TBX11K Tuberculosis Detection, values in %; [image](figures/exp2_leaderboard_reference.jpg)):

| Reference (leaderboard row on slide 64) | Cat-agnostic AP50 | Cat-agnostic AP75 | Cat-agnostic AP | Active AP50 | Active AP75 | Active AP | Latent AP50 | Latent AP75 | Latent AP |
|---|---|---|---|---|---|---|---|---|---|
| alitariq (rank 1, 16 entries, 11/09/21) | 61.74 | 19.64 | 27.32 | 54.76 | 19.49 | 25.69 | 11.77 | 0.67 | 3.52 |

CSV: [`exp2_tbx11k_leaderboard_reference.csv`](tables/exp2_tbx11k_leaderboard_reference.csv)

- **Epoch 6** is best for AP50 (agnostic 0.6622, Active 0.6241, Latent 0.0763; FROC 0.4991).
- **Epoch 11** is best for FROC: 0.5930.
- **Epoch 15** is best for AP75 (agnostic 0.1905, Active 0.1754, Latent 0.0367; FROC 0.5817).
- *Observation:* in percent, the epoch-6 AP50 values (66.22 agnostic, 62.41 Active) are above the leaderboard row (61.74, 54.76), epoch-11 Latent AP50 (14.10) is above 11.77, and epoch-15 agnostic AP75 (19.05) is just below 19.64. This is indicative only: the slide uses a custom train/test split, not the official benchmark split.

## 2.8 Run logs and every epoch's metrics

The original scripts of this run are in [`../original_run/`](../original_run), and their SLURM logs are in [`run_logs/`](run_logs):

| Log | What it shows |
|---|---|
| `new-split.txt` | Output of the split: per-category train/test counts and totals 4,622 / 1,800 / 1,977 |
| `swin_train_39951205.out` | Training job (Nov 20, 2025): train and validation loss per epoch, which is table 2.2 |
| `swin_test_39960340.out` | Test job: AP50 / AP75 / FROC of all 16 checkpoints on the 1,977 test images, and the best-epoch summary |
| `min_max_predict_39962531.out` | Best and worst test images per category at epoch 11 |

All 16 epochs from the test job ([`exp2_tbx11k_metrics_all_epochs.csv`](tables/exp2_tbx11k_metrics_all_epochs.csv); the bold rows are table 2.7):

| Epoch | AP50 agnostic | AP50 Active | AP50 Latent | AP75 agnostic | AP75 Active | AP75 Latent | FROC |
|---|---|---|---|---|---|---|---|
| 1 | 0.3478 | 0.3077 | 0.0042 | 0.0569 | 0.0232 | 0.0000 | 0.2394 |
| 2 | 0.4809 | 0.4543 | 0.0250 | 0.0730 | 0.0730 | 0.0012 | 0.3328 |
| 3 | 0.5532 | 0.5165 | 0.0121 | 0.1635 | 0.1546 | 0.0010 | 0.4204 |
| 4 | 0.5785 | 0.5503 | 0.0307 | 0.1518 | 0.1396 | 0.0007 | 0.4920 |
| 5 | 0.6299 | 0.5969 | 0.0874 | 0.1148 | 0.1083 | 0.0476 | 0.5542 |
| **6** | **0.6622** | **0.6241** | **0.0763** | **0.1059** | **0.0902** | **0.0025** | **0.4991** |
| 7 | 0.6007 | 0.5857 | 0.1618 | 0.1530 | 0.1544 | 0.0077 | 0.4626 |
| 8 | 0.4560 | 0.4559 | 0.0402 | 0.0718 | 0.0716 | 0.0040 | 0.4277 |
| 9 | 0.5769 | 0.5684 | 0.0752 | 0.1371 | 0.1173 | 0.0061 | 0.5011 |
| 10 | 0.4634 | 0.4343 | 0.0545 | 0.0735 | 0.0693 | 0.0046 | 0.4168 |
| **11** | **0.6449** | **0.6114** | **0.1410** | **0.1239** | **0.1095** | **0.0095** | **0.5930** |
| 12 | 0.5941 | 0.5757 | 0.0965 | 0.1826 | 0.1874 | 0.0056 | 0.4990 |
| 13 | 0.5771 | 0.5712 | 0.1109 | 0.1444 | 0.1398 | 0.0045 | 0.5212 |
| 14 | 0.5828 | 0.5361 | 0.1200 | 0.1610 | 0.1573 | 0.0173 | 0.4345 |
| **15** | **0.6332** | **0.6016** | **0.1249** | **0.1905** | **0.1754** | **0.0367** | **0.5817** |
| 16 | 0.5496 | 0.5590 | 0.1811 | 0.1352 | 0.1440 | 0.0156 | 0.4530 |

Best and worst test images per category at epoch 11 ([`exp2_tbx11k_min_max_by_category_epoch11.csv`](tables/exp2_tbx11k_min_max_by_category_epoch11.csv)). Each image is scored by F1 at confidence 0.5 and IoU 0.5:
- **No_TB:** the best images have no false boxes (F1 1.0). The worst have a single false box each.
- **Active_TB:** the 3 best images find every lesion with no false boxes (F1 1.0). The worst image boxes both lungs but too small to reach IoU 0.5 (see 2.10); the next two miss their lesion.
- **Latent_TB / Mixed_TB:** even the best images reach only F1 0.33–0.67, typically finding 1 of 2 lesions. This matches the low Latent TB AP above.

## 2.9 FROC curves of every epoch

![FROC curves of all 16 epochs](figures/exp2_froc_curves_all_epochs.png)

*[`exp2_froc_curves_all_epochs.png`](figures/exp2_froc_curves_all_epochs.png): one FROC curve per checkpoint, all on one chart. Drawn by the test job; not in the slides.*
- Epoch 1 is clearly the weakest: sensitivity stalls near 0.43 until about 0.6 FPPI.
- From epoch 2 on, most curves climb steeply, to roughly 0.5–0.65 sensitivity by 0.1 FPPI.
- The best curves, epoch 11 among them, reach about 0.75–0.8 sensitivity by 0.3–0.5 FPPI.
- Curves end at different FPPI. The rightmost point of each curve is its lowest score threshold (0.05). The better epochs produce few low-confidence false boxes, so their curves stop early; epoch 11 stops at about 0.53 FPPI.
- *Note:* the 10-colour cycle repeats, so epochs 1 and 11, 2 and 12, and so on share colours.

## 2.10 Prediction examples (epoch 11)

Ground truth (left, solid boxes) and model predictions with confidence > 0.5 (right, dashed boxes) on test images. Red = Active TB, blue = Latent TB. These are picked from the per-category best and worst lists made by [`../original_run/min_max_predict.py`](../original_run/min_max_predict.py) (log: `run_logs/min_max_predict_39962531.out`). All 24 best/worst images are listed in [`tables/exp2_tbx11k_min_max_by_category_epoch11.csv`](tables/exp2_tbx11k_min_max_by_category_epoch11.csv).

**What the model gets right**

| Image | What it shows |
|---|---|
| ![](predictions/best_Active_TB_1_img1970_f11.000_e11.png) | **Active TB, single lesion:** found with confidence 0.98 (F1 1.0). The predicted box covers the upper part of the annotated region, which is enough overlap to count. |
| ![](predictions/best_Active_TB_2_img1845_f11.000_e11.png) | **Active TB, two lesions:** both found (0.95 and 0.68), boxes closely match the annotations (F1 1.0). |
| ![](predictions/best_No_TB_1_img1976_f11.000_e11.png) | **No TB:** image labelled No_TB, no boxes predicted, which is the correct output. |
| ![](predictions/best_Latent_TB_1_img1245_f10.667_e11.png) | **Latent TB, best case:** one of two latent lesions found with the correct class (0.51); the other is missed (F1 0.67). Even the best latent case is only half detected. |
| ![](predictions/best_Mixed_TB_1_img587_f10.667_e11.png) | **Mixed TB, best case:** the active lesion is found (0.68); the latent lesion in the other lung is missed (F1 0.67). |

**Where it fails**

| Image | What it shows |
|---|---|
| ![](predictions/worst_Active_TB_1_img468_f10.000_e11.png) | **Active TB, box-size mismatch:** the model fires on both lungs with high confidence (0.99, 0.95). The annotated boxes cover each whole lung, but the predictions cover only the upper half, so IoU is below 0.5 and both count as misses (F1 0). The disease is found; the boxes are too small. |
| ![](predictions/worst_Latent_TB_2_img83_f10.000_e11.png) | **Latent TB, wrong class:** both latent lesions are boxed in the right places, but labelled Active TB (0.88, 0.92). A category-agnostic metric counts these as hits, and a per-class one does not. Mistakes like this lower Latent TB AP (0.1410 at epoch 11) without lowering the agnostic AP. |
| ![](predictions/worst_No_TB_1_img93_f10.000_e11.png) | **No TB, false alarm:** a confident Active TB box (0.99) over one lung of an image labelled No_TB. |

*Observation:* the failures are mostly not "missed disease". They are boxes that are too small for large annotations, or latent lesions labelled as active. The latter matches the Latent TB weakness seen in the metrics (73 training images vs 332 for Active TB).

---

*Source: "Final Presentation" (Arizona Board of Regents, 2025). Tables were transcribed from the slide text and cross-checked against the rendered slides; the images are the slides themselves or the figures embedded in them.*
