"""
student_train.py
Train student model on CheXpert with pseudo-labeled training data (no uncertainty labels).
Based on original train.py implementation, optimized for soft-label training.

IMPROVEMENTS:
- Save both train_loss and val_loss in checkpoints
"""

import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import datetime

# ================= CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}, DEVICE: {DEVICE}")

BATCH_SIZE = 96
NUM_CLASSES = 14
TOTAL_EPOCHS = 100
EPOCHS_PER_JOB = 10
CHECKPOINT_DIR = "/scratch/smanika3/feedback/v2-self/checkpoints/student"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

UNCOMPRESSED_DIR = "/scratch/smanika3/chexpert/full_uncompressed"
TRAIN_IMAGE_BASE = os.path.join(UNCOMPRESSED_DIR, "train")
CSV_BASE_PATH = os.path.join(UNCOMPRESSED_DIR, 'CheXpert-v1.0 batch 1 (validate & csv)')
TRAIN_CSV_PATH = os.path.join(CSV_BASE_PATH, "train_pseudolabels.csv")  # Pseudo-labeled CSV
VAL_CSV_PATH = os.path.join(CSV_BASE_PATH, "valid.csv")
VAL_IMAGE_BASE = os.path.join(UNCOMPRESSED_DIR, "valid")

CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]

# ================= DATASET =================
class CheXpertDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_base, transform=None, is_train=True):
        self.df = pd.read_csv(csv_file)
        self.img_base = img_base
        self.transform = transform
        self.is_train = is_train
        
        # Fill NaN with 0 and convert to float
        for col in CHEXPERT_LABELS:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0).astype(float)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prefix_to_remove = 'CheXpert-v1.0/train/' if self.is_train else 'CheXpert-v1.0/valid/'
        relative_path = row["Path"].replace(prefix_to_remove, '', 1)
        img_path = os.path.join(self.img_base, relative_path)
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        labels = row[CHEXPERT_LABELS].values.astype(float)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return image, labels

# ================= TRANSFORMS =================
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.05,0.05), scale=(0.95,1.05), fill=128),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ================= DATA LOADERS =================
train_dataset = CheXpertDataset(TRAIN_CSV_PATH, TRAIN_IMAGE_BASE, transform=transform_train, is_train=True)
val_dataset = CheXpertDataset(VAL_CSV_PATH, VAL_IMAGE_BASE, transform=transform_val, is_train=False)
print(f"[DEBUG] Train dataset size: {len(train_dataset)}")
print(f"[DEBUG] Validation dataset size: {len(val_dataset)}")

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=28, 
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True,
)

print(f"[DEBUG] DataLoaders created with persistent_workers=True")

# ================= MODEL =================
model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=NUM_CLASSES)
model.head = nn.Sequential(
    model.head,
    nn.Dropout(p=0.5)
)

if torch.cuda.device_count() > 1:
    print(f"[INFO] Found {torch.cuda.device_count()} GPUs, using DataParallel!")
    model = nn.DataParallel(model)
else:
    print(f"[INFO] Only one GPU found")

model = model.to(DEVICE)

# ================= OPTIMIZER & LOSS =================
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
scaler = GradScaler()

def bce_loss(outputs, labels): #Used only for Train
    """Binary cross-entropy loss for soft labels (no masking needed)."""
    return nn.functional.binary_cross_entropy_with_logits(outputs, labels)

def masked_bce_loss(outputs, labels): #Used only for Val
    """
    BCE loss for evaluation: ignores -1 labels (uncertainty) as in CheXpert.
    """
    mask = (labels != -1).float()
    safe_labels = labels.clone()
    safe_labels[labels == -1] = 0  # doesn't matter, will be masked out
    loss = nn.functional.binary_cross_entropy_with_logits(outputs, safe_labels, reduction='none')
    loss = (loss * mask).sum() / mask.sum()
    return loss


# ================= CHECKPOINT RESUME =================
def get_latest_checkpoint(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch_*.pth"))
    if not ckpts:
        return None
    def extract_epoch(path):
        return int(os.path.basename(path).split("_")[-1].split(".")[0])
    return max(ckpts, key=extract_epoch)

torch.cuda.empty_cache()
latest_ckpt = get_latest_checkpoint(CHECKPOINT_DIR)
if latest_ckpt:
    print(f"[INFO] Resuming from checkpoint: {latest_ckpt}")
    checkpoint = torch.load(latest_ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE, non_blocking=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    del checkpoint
    torch.cuda.empty_cache()
else:
    print("[INFO] No checkpoint found, starting from scratch")
    start_epoch = 1

end_epoch = min(start_epoch + EPOCHS_PER_JOB - 1, TOTAL_EPOCHS)
print(f"[INFO] Training from epoch {start_epoch} to {end_epoch}")

# ================= TRAINING LOOP =================
for epoch in range(start_epoch, end_epoch + 1):
    # ===== TRAINING PHASE =====
    model.train()
    running_loss = 0.0
    
    for images, labels in tqdm(train_loader, desc=f"[STUDENT] Epoch {epoch}/{end_epoch} [TRAIN]"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = bce_loss(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
    
    avg_train_loss = running_loss / len(train_loader.dataset)
    print(f"[INFO] Epoch {epoch} completed. Avg Train Loss: {avg_train_loss:.4f}")
    
    # ===== VALIDATION PHASE =====
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"[STUDENT] Epoch {epoch}/{end_epoch} [VAL]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = masked_bce_loss(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
    
    avg_val_loss = val_running_loss / len(val_loader.dataset)
    print(f"[INFO] Epoch {epoch} validation loss: {avg_val_loss:.4f}")
    
    # ===== SAVE CHECKPOINT =====
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss
    }, ckpt_path)
    print(f"[INFO] Saved checkpoint: {ckpt_path}")
    print(f"        Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    print(f"        Saved at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    model.train()  

print("[INFO] Student training job finished")
