"""
Comprehensive evaluation script for CheXpert checkpoints.
- Computes AUROC for all classes per epoch
- Creates separate plots for 12 diseases (excluding No Finding and Support Devices)
- Exports results to CSV
"""

import os
import torch
import timm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt

print("[DEBUG] Starting multi-checkpoint AUC evaluation")

# ================= PATHS AND CONFIG =================
# EDIT BEFORE RUNNING: the paths below are the author's ASU Sol cluster paths.
#   CHECKPOINT_DIR  -> same folder v2_train_swin.py writes checkpoints to
#   TEST_IMAGE_BASE -> folder the "Path" column of test_labels.csv is relative to
#   TEST_CSV_PATH   -> CheXpert test set labels (668 images)
#   OUTPUT_DIR / OUTPUT_CSV -> where per-epoch AUROC CSV and plots are written
CHECKPOINT_DIR = "/scratch/smanika3/feedback/v1/checkpoints"
MODEL_NAME = "swin_base_patch4_window7_224"
NUM_CLASSES = 14
BATCH_SIZE = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_IMAGE_BASE = "/home/smanika3/scratch/CheXpert"
TEST_CSV_PATH = "/home/smanika3/scratch/CheXpert/test_labels.csv"

# Output paths
OUTPUT_CSV = "/scratch/smanika3/feedback/v1/plots/auroc_results_per_epoch.csv"
OUTPUT_DIR = "/scratch/smanika3/feedback/v1/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# All CheXpert labels
CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]

# 12 diseases to evaluate (excluding No Finding and Support Devices)
EVAL_CLASSES = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", 
    "Pleural Effusion", "Pleural Other", "Fracture"
]

# Leaderboard classes
LEADERBOARD_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"
]

leaderboard_indices = [CHEXPERT_LABELS.index(cls) for cls in LEADERBOARD_CLASSES]
eval_indices = [CHEXPERT_LABELS.index(cls) for cls in EVAL_CLASSES]

print(f"[DEBUG] Using device: {DEVICE}")
print(f"[DEBUG] Test image base: {TEST_IMAGE_BASE}")
print(f"[DEBUG] Test CSV: {TEST_CSV_PATH}")
print(f"[DEBUG] Evaluating {len(EVAL_CLASSES)} disease classes")
print(f"[DEBUG] Leaderboard classes: {LEADERBOARD_CLASSES}")

# ================= TRANSFORMS =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ================= DATASET =================
class CheXpertDataset(Dataset):
    def __init__(self, df, transform=None):
        self.image_paths = df["Path"].tolist()
        self.labels = df["labels"].tolist()
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARNING] Bad file {img_path}: {e}")
            return self.__getitem__(np.random.randint(len(self)))
        if self.transform:
            image = self.transform(image)
        return image, label

# ================= UNCERTAINTY POLICIES =================
UNCERTAINTY_POLICIES = {
    2: "U-Ones",        # Cardiomegaly
    5: "U-Ignore",      # Edema
    6: "U-Ignore",      # Consolidation
    8: "Multiclass",    # Atelectasis
    10: "U-Zeros"       # Pleural Effusion
}

def policy_label(label, policy):
    """Apply uncertainty policy to label."""
    if policy == "U-Ignore":
        return label if label != -1 else 0.0
    elif policy == "U-Ones":
        return 1.0 if label == -1 else label
    elif policy == "U-Zeros":
        return 0.0 if label == -1 else label
    else:
        return label

def prepare_labels_dataframe(csv_path):
    """Load and prepare test labels with uncertainty policies."""
    df = pd.read_csv(csv_path)
    print(f"[DEBUG] Loaded {len(df)} rows from CSV.")
    
    # Apply per-class uncertainty logic
    for i, col in enumerate(CHEXPERT_LABELS):
        policy = UNCERTAINTY_POLICIES.get(i, "U-Ignore")
        df[col] = df[col].fillna(0).apply(lambda v: policy_label(v, policy))
    
    df['Path'] = df['Path'].apply(lambda x: os.path.join(TEST_IMAGE_BASE, x))
    df['labels'] = df[CHEXPERT_LABELS].values.tolist()
    print(f"[DEBUG] Sample image path: {df['Path'].iloc[0]}")
    return df[['Path', 'labels']]

# ================= PREPARE DATA =================
test_df = prepare_labels_dataframe(TEST_CSV_PATH)
test_dataset = CheXpertDataset(test_df, transform=transform)
test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=16, 
    pin_memory=True
)
print(f"[DEBUG] Test DataLoader ready. Batches: {len(test_loader)}")

# ================= FIND CHECKPOINTS =================
def extract_epoch(path):
    return int(os.path.basename(path).split('_')[-1].split('.')[0])

all_ckpts = sorted(
    [os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint_epoch_")],
    key=extract_epoch
)
print(f"[DEBUG] Found {len(all_ckpts)} checkpoints to evaluate")

# ================= STORAGE FOR RESULTS =================
# Store AUROC for all classes per epoch
auc_results = {cls: [] for cls in CHEXPERT_LABELS}
epoch_list = []

# ================= MAIN EVALUATION LOOP =================
for idx_ckpt, ckpt_path in enumerate(all_ckpts):
    epoch = extract_epoch(ckpt_path)
    print(f"\n{'='*80}")
    print(f"[EPOCH {epoch}] Evaluating checkpoint ({idx_ckpt+1}/{len(all_ckpts)})")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*80}")
    
    # Load model
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    orig_head = model.head
    model.head = torch.nn.Sequential(orig_head, torch.nn.Dropout(p=0.5))
    
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    # Inference
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc=f"Epoch {epoch}")):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    print(f"[DEBUG] Predictions shape: {all_preds.shape}")
    
    # Compute AUROC for all classes
    epoch_list.append(epoch)
    
    print(f"\n[RESULTS] AUROC scores for epoch {epoch}:")
    for i, cls in enumerate(CHEXPERT_LABELS):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        except ValueError:
            auc = np.nan
        auc_results[cls].append(auc)
        
        if cls in EVAL_CLASSES:
            print(f"  {cls:30s}: {auc:.4f}")
    
    # Leaderboard mean
    leaderboard_aucs = [auc_results[cls][-1] for cls in LEADERBOARD_CLASSES]
    mean_leaderboard = np.nanmean(leaderboard_aucs)
    print(f"\n  {'Mean Leaderboard AUC':30s}: {mean_leaderboard:.4f}")
    
    # Cleanup
    del model, checkpoint
    torch.cuda.empty_cache()

# ================= EXPORT TO CSV =================
print(f"\n[INFO] Exporting results to CSV: {OUTPUT_CSV}")
results_df = pd.DataFrame({'Epoch': epoch_list})
for cls in CHEXPERT_LABELS:
    results_df[cls] = auc_results[cls]
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"[INFO] ✓ CSV exported successfully")

# ================= PLOTTING =================
print(f"\n[INFO] Creating plots...")

# 1. Leaderboard plot (5 classes)
print("[INFO] Creating leaderboard plot...")
plt.figure(figsize=(12, 7))
for cls in LEADERBOARD_CLASSES:
    plt.plot(epoch_list, auc_results[cls], marker='o', linewidth=2, label=cls)

mean_leaderboard_trace = [
    np.nanmean([auc_results[cls][i] for cls in LEADERBOARD_CLASSES])
    for i in range(len(epoch_list))
]
plt.plot(epoch_list, mean_leaderboard_trace, 
         marker='X', linestyle='--', linewidth=3, 
         color='black', label='Mean Leaderboard')

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("AUROC", fontsize=12)
plt.title("CheXpert Leaderboard Classes - AUROC per Epoch", fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "leaderboard_auroc.png"), dpi=300)
print(f"[INFO] ✓ Saved: {OUTPUT_DIR}/leaderboard_auroc.png")

# 2. Individual plots for each of the 12 evaluation classes
print(f"[INFO] Creating individual plots for {len(EVAL_CLASSES)} classes...")
for cls in EVAL_CLASSES:
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, auc_results[cls], 
             marker='o', linewidth=2, color='steelblue')
    
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("AUROC", fontsize=12)
    plt.title(f"{cls} - AUROC per Epoch", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create safe filename
    safe_filename = cls.replace(" ", "_").lower()
    save_path = os.path.join(OUTPUT_DIR, f"{safe_filename}_auroc.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO]   ✓ Saved: {save_path}")

# 3. Combined plot for all 12 evaluation classes
print("[INFO] Creating combined plot for all evaluation classes...")
plt.figure(figsize=(16, 10))
for cls in EVAL_CLASSES:
    plt.plot(epoch_list, auc_results[cls], marker='o', linewidth=2, label=cls)

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("AUROC", fontsize=12)
plt.title("All 12 Disease Classes - AUROC per Epoch", fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "all_classes_auroc.png"), dpi=300, bbox_inches='tight')
print(f"[INFO] ✓ Saved: {OUTPUT_DIR}/all_classes_auroc.png")

# ================= SUMMARY =================
print(f"\n{'='*80}")
print("[SUMMARY]")
print(f"{'='*80}")
print(f"Total epochs evaluated: {len(epoch_list)}")
print(f"Results CSV: {OUTPUT_CSV}")
print(f"Plots directory: {OUTPUT_DIR}/")
print(f"  - Leaderboard plot: leaderboard_auroc.png")
print(f"  - Combined plot: all_classes_auroc.png")
print(f"  - Individual plots: {len(EVAL_CLASSES)} files")
print(f"\n[INFO] ✓ Evaluation complete!")
print(f"{'='*80}")
