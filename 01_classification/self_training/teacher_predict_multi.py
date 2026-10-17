"""
teacher_predict_multi.py (OPTIMIZED + MULTI-THRESHOLD)
Ensemble per-class teachers, generate soft-labels for uncertain -1 labels in train.csv.
CheXpert-style self-training approach.

MULTI-THRESHOLD EXPERIMENT:
Generates multiple pseudo-labeled datasets with different threshold combinations:
- 0.5 & 0.5
- 0.6 & 0.4
- 0.7 & 0.3
- 0.8 & 0.2
- 0.9 & 0.1

OPTIMIZATIONS:
- Group classes by checkpoint to avoid duplicate model loading
- Only process images with -1 (uncertain) labels for each class
- Run predictions ONCE, apply multiple thresholds
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import datetime
from collections import defaultdict

print("[DEBUG] All imports successful")

# ================= CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}, DEVICE: {DEVICE}")

BATCH_SIZE = 96
NUM_CLASSES = 14
# EDIT BEFORE RUNNING: author's ASU Sol paths. CHECKPOINT_DIR = teacher checkpoints to load (the author used the
# Exp 1 run, exp1_policy_training/). Writes one train_pseudolabels_thresh_<high>_<low>.csv per THRESHOLD_CONFIGS entry
# into CSV_BASE_PATH; point student_train_thresh.py's TRAIN_CSV_PATH at the one you want.
CHECKPOINT_DIR = "/scratch/smanika3/feedback/v1/checkpoints"
UNCOMPRESSED_DIR = "/scratch/smanika3/chexpert/full_uncompressed"
TRAIN_IMAGE_BASE = os.path.join(UNCOMPRESSED_DIR, "train")
CSV_BASE_PATH = os.path.join(UNCOMPRESSED_DIR, 'CheXpert-v1.0 batch 1 (validate & csv)')
TRAIN_CSV_PATH = os.path.join(CSV_BASE_PATH, "train.csv")

CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

# 12 evaluation classes (excluding No Finding and Support Devices)
EVAL_CLASSES = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture"
]
eval_indices = [CHEXPERT_LABELS.index(cls) for cls in EVAL_CLASSES]

# ================= PER-CLASS BEST EPOCH CONFIG =================
CLASS_TO_BEST_EPOCH = {
    "Enlarged Cardiomediastinum": 1,
    "Cardiomegaly": 10,
    "Lung Opacity": 8,
    "Lung Lesion": 10,
    "Edema": 10,
    "Consolidation": 5,
    "Pneumonia": 2,
    "Atelectasis": 11,
    "Pneumothorax": 12,
    "Pleural Effusion": 10,
    "Pleural Other": 2,
    "Fracture": 13
}

print("[DEBUG] Per-class best epochs:")
for cls, epoch in CLASS_TO_BEST_EPOCH.items():
    print(f"  {cls}: epoch {epoch}")

# ================= MULTI-THRESHOLD CONFIGURATION =================
# Format: (THRESH_HIGH, THRESH_LOW)
THRESHOLD_CONFIGS = [
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
    (0.8, 0.2),
    (0.9, 0.1)
]

print(f"\n[DEBUG] Multi-threshold experiment with {len(THRESHOLD_CONFIGS)} configurations:")
for high, low in THRESHOLD_CONFIGS:
    print(f"  HIGH={high}, LOW={low}")

# ================= GROUP CLASSES BY CHECKPOINT (OPTIMIZATION 1) =================
# Map: checkpoint_epoch -> [list of classes]
CHECKPOINT_TO_CLASSES = defaultdict(list)
for cls, epoch in CLASS_TO_BEST_EPOCH.items():
    CHECKPOINT_TO_CLASSES[epoch].append(cls)

print("\n[DEBUG] Checkpoint grouping (OPTIMIZATION):")
for epoch in sorted(CHECKPOINT_TO_CLASSES.keys()):
    classes = CHECKPOINT_TO_CLASSES[epoch]
    print(f"  Epoch {epoch}: {len(classes)} classes - {', '.join(classes)}")

# ================= TRANSFORMS =================
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ================= DATASET WITH CLASS FILTERING (OPTIMIZATION 2) =================
class CheXpertDatasetFiltered(Dataset):
    """Dataset that filters to only samples with -1 for specific class(es)."""
    def __init__(self, csv_file, img_base, target_classes, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_base = img_base
        self.transform = transform
        self.target_classes = target_classes
        
        # Filter to rows that have -1 for ANY of the target classes
        mask = self.df[target_classes].isin([-1]).any(axis=1)
        self.filtered_indices = np.where(mask)[0]
        
        print(f"[DEBUG] Filtered dataset: {len(self.filtered_indices)} / {len(self.df)} samples have -1 for {target_classes}")
    
    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.filtered_indices[idx]
        row = self.df.iloc[actual_idx]
        rel_path = row["Path"].replace('CheXpert-v1.0/train/', '', 1)
        img_path = os.path.join(self.img_base, rel_path)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARNING] Error loading {img_path}: {e}")
            return self.__getitem__(np.random.randint(len(self)))
        
        if self.transform:
            image = self.transform(image)
        
        return image, actual_idx  # Return actual index for CSV alignment

# ================= MAIN PROCESS =================
print("\n[DEBUG] Loading original train.csv...")
df_orig = pd.read_csv(TRAIN_CSV_PATH)
print(f"[DEBUG] Original train.csv: {len(df_orig)} samples, {len(CHEXPERT_LABELS)} classes")

# Store soft labels for each class
soft_labels = {cls: np.zeros(len(df_orig)) for cls in EVAL_CLASSES}

# ================= GROUPED PER-CHECKPOINT PREDICTION LOOP =================
print("\n[DEBUG] Starting grouped checkpoint predictions...")
total_checkpoints = len(CHECKPOINT_TO_CLASSES)
checkpoint_count = 0

for epoch in sorted(CHECKPOINT_TO_CLASSES.keys()):
    checkpoint_count += 1
    classes_for_this_ckpt = CHECKPOINT_TO_CLASSES[epoch]
    
    print(f"\n{'='*80}")
    print(f"[CHECKPOINT {checkpoint_count}/{total_checkpoints}] Epoch {epoch}")
    print(f"[INFO] Classes to process: {classes_for_this_ckpt}")
    print(f"{'='*80}")
    
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
    
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        continue
    
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    
    # Create model ONCE for this checkpoint
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=NUM_CLASSES)
    model.head = nn.Sequential(model.head, nn.Dropout(p=0.5))
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = checkpoint['model_state_dict']
    
    # Handle DataParallel wrapper
    if list(state_dict.keys())[0].startswith('module.'):
        print("[DEBUG] Removing 'module.' prefix from DataParallel model...")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    print(f"[INFO] ✓ Model loaded successfully (reused for {len(classes_for_this_ckpt)} classes)")
    
    # Create filtered dataset for this checkpoint's classes
    print(f"\n[INFO] Creating filtered dataset...")
    pred_dataset = CheXpertDatasetFiltered(
        TRAIN_CSV_PATH, 
        TRAIN_IMAGE_BASE, 
        classes_for_this_ckpt,
        transform=transform_val
    )
    pred_loader = DataLoader(
        pred_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=28,
        pin_memory=True
    )
    print(f"[DEBUG] DataLoader created: {len(pred_loader)} batches")
    
    # Get predictions once for all classes in this checkpoint
    all_outputs = {}  # outputs[actual_idx] = [logits for all 14 classes]
    
    print(f"\n[INFO] Forward pass on filtered dataset...")
    with torch.no_grad():
        for batch_idx, (images, actual_indices) in enumerate(tqdm(pred_loader, desc=f"Epoch {epoch} forward pass")):
            images = images.to(DEVICE)
            outputs = model(images)
            logits = outputs.cpu().numpy()  # [batch_size, 14]
            
            for i, actual_idx in enumerate(actual_indices.numpy()):
                all_outputs[actual_idx] = logits[i]
            
            if batch_idx % 50 == 0:
                print(f"[DEBUG] Batch {batch_idx+1}/{len(pred_loader)} processed")
    
    # Extract soft labels for each class using the same forward pass results
    print(f"\n[INFO] Extracting per-class soft labels...")
    for cls in classes_for_this_ckpt:
        class_idx = CHEXPERT_LABELS.index(cls)
        
        for actual_idx, logits in all_outputs.items():
            prob = 1.0 / (1.0 + np.exp(-logits[class_idx]))  # Sigmoid
            soft_labels[cls][actual_idx] = prob
        
        print(f"  ✓ {cls}: predictions extracted")
    
    # Cleanup
    del model, checkpoint, state_dict, all_outputs
    torch.cuda.empty_cache()

# ================= GENERATE MULTIPLE THRESHOLD-BASED CSVs =================
print(f"\n{'='*80}")
print("[INFO] Generating pseudo-labeled CSVs with different thresholds...")
print(f"{'='*80}")

for thresh_high, thresh_low in THRESHOLD_CONFIGS:
    print(f"\n[THRESHOLD] HIGH={thresh_high}, LOW={thresh_low}")
    
    # Create copy of original dataframe
    df_pseudo = df_orig.copy()
    
    # Statistics for this threshold configuration
    confidence_stats = {cls: {"soft": 0, "hard_1": 0, "hard_0": 0, "total": 0} for cls in EVAL_CLASSES}
    
    # Apply thresholding
    for cls in EVAL_CLASSES:
        for i in range(len(df_pseudo)):
            orig_label = df_pseudo.loc[i, cls]
            
            # Only replace -1 (uncertain) labels
            if orig_label == -1:
                prob = soft_labels[cls][i]
                
                # Apply hard thresholding
                if prob >= thresh_high:
                    new_label = 1.0
                    confidence_stats[cls]["hard_1"] += 1
                elif prob <= thresh_low:
                    new_label = 0.0
                    confidence_stats[cls]["hard_0"] += 1
                else:
                    new_label = prob
                    confidence_stats[cls]["soft"] += 1
                
                confidence_stats[cls]["total"] += 1
                df_pseudo.at[i, cls] = new_label
    
    # Save output CSV with threshold in filename
    output_path = os.path.join(CSV_BASE_PATH, f"train_pseudolabels_thresh_{thresh_high}_{thresh_low}.csv")
    print(f"  Saving to: {output_path}")
    df_pseudo.to_csv(output_path, index=False)
    print(f"  ✓ CSV saved successfully")
    
    # Print statistics for this threshold
    total_replaced = 0
    print(f"\n  [STATS] Pseudo-Label Statistics for HIGH={thresh_high}, LOW={thresh_low}")
    print(f"  {'-'*76}")
    
    for cls in EVAL_CLASSES:
        s = confidence_stats[cls]
        if s["total"] > 0:
            pct_soft = (s["soft"] / s["total"]) * 100
            pct_hard_1 = (s["hard_1"] / s["total"]) * 100
            pct_hard_0 = (s["hard_0"] / s["total"]) * 100
            
            total_replaced += s["total"]
    
    print(f"  Total replaced: {total_replaced} uncertain labels")
    print(f"  Total soft: {sum(s['soft'] for s in confidence_stats.values())}")
    print(f"  Total hard 1s: {sum(s['hard_1'] for s in confidence_stats.values())}")
    print(f"  Total hard 0s: {sum(s['hard_0'] for s in confidence_stats.values())}")

# ================= FINAL SUMMARY =================
print(f"\n{'='*80}")
print("[SUMMARY] Multi-Threshold Experiment Complete")
print(f"{'='*80}")
print(f"Generated {len(THRESHOLD_CONFIGS)} pseudo-labeled datasets:")
for thresh_high, thresh_low in THRESHOLD_CONFIGS:
    output_path = os.path.join(CSV_BASE_PATH, f"train_pseudolabels_thresh_{thresh_high}_{thresh_low}.csv")
    print(f"  - train_pseudolabels_thresh_{thresh_high}_{thresh_low}.csv")

print(f"\nOriginal CSV uncertain count: {(df_orig[EVAL_CLASSES] == -1).sum().sum()}")
print(f"{'='*80}")

print(f"\n[INFO] ✓ Processing complete at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"[INFO] Next step: Use the generated CSVs for experimental student training!")