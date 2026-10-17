import os
import glob
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import datetime


# === SWIN UNET V2 MODEL IMPORT ===
# Original training script of the reported run (code unchanged). EDIT BEFORE RUNNING: author's ASU Sol paths.
# The sys.path line must point at a folder containing network/swin_unet_v2.py (copy ../models/ into network/).
# TRAIN_LIST is the 70% list from ../data/split_train_test.py (the original run overwrote train_files.txt with it).
sys.path.append('/home/smanika3/ondemand/data/sys/myjobs/projects/default/6/mmdet/Pytorch-Swin-Unet-V2')
from network.swin_unet_v2 import SwinTransformerSys


# === CONFIG ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}, DEVICE: {DEVICE}")

BATCH_SIZE = 96
NUM_CLASSES = 4
TOTAL_EPOCHS = 40
EPOCHS_PER_JOB = 10
CHECKPOINT_DIR = "/scratch/smanika3/checkpoints-swin-segmentation"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

TRAIN_IMAGE_DIR = "/scratch/smanika3/chexpert/full_uncompressed/train/"
VALID_IMAGE_DIR = "/scratch/smanika3/chexpert/full_uncompressed/valid/"
TRAIN_MASK_DIR = "/scratch/smanika3/chexmask-masks/train/"
VALID_MASK_DIR = "/scratch/smanika3/chexmask-masks/valid/"
TRAIN_LIST = "/scratch/smanika3/chexmask-masks/train_files.txt"
VALID_LIST = "/scratch/smanika3/chexmask-masks/valid_files.txt"


# === DATASET ===
class CheXmaskSegDataset(Dataset):
    def __init__(self, img_base, mask_base, file_list, transform=None, mask_transform=None):
        self.img_base = img_base
        self.mask_base = mask_base
        with open(file_list) as f:
            self.images = [line.strip() for line in f]
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_base, self.images[idx])
        mask_path = os.path.join(self.mask_base, self.images[idx].replace('.jpg', '.png'))
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        if self.transform:
            image = self.transform(image)

        mask = np.array(mask, dtype=np.uint8)
        mask = Image.fromarray(mask)
        mask = mask.resize((224, 224), resample=Image.NEAREST)
        mask = np.array(mask, dtype=np.int64)
        mask = torch.from_numpy(mask)

        return image, mask


# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# === DATA LOADERS ===
train_dataset = CheXmaskSegDataset(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, TRAIN_LIST, transform=transform)
val_dataset = CheXmaskSegDataset(VALID_IMAGE_DIR, VALID_MASK_DIR, VALID_LIST, transform=transform)
print(f"[DEBUG] Train dataset size: {len(train_dataset)}")
print(f"[DEBUG] Validation dataset size: {len(val_dataset)}")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


# === MODEL ===
model = SwinTransformerSys(img_size=224, num_classes=NUM_CLASSES)

# DO NOT move model to GPU yet - will do after checkpoint loading
if torch.cuda.device_count() > 1:
    print(f"[INFO] Found {torch.cuda.device_count()} GPUs, using DataParallel!")
    model = nn.DataParallel(model)
else:
    print(f"[INFO] Only one GPU found")

# Initialize optimizer BEFORE moving to GPU
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
criterion = nn.CrossEntropyLoss()


# === CHECKPOINT RESUME ===
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
    model = model.to(DEVICE, non_blocking=True)  # Move to GPU
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # Load optimizer state AFTER moving model to GPU
    start_epoch = checkpoint["epoch"] + 1
    del checkpoint
    torch.cuda.empty_cache()
else:
    print("[INFO] No checkpoint found, starting from scratch")
    start_epoch = 1


# NOW move model to GPU after checkpoint loading
model = model.to(DEVICE)

end_epoch = min(start_epoch + EPOCHS_PER_JOB - 1, TOTAL_EPOCHS)
print(f"[INFO] Training from epoch {start_epoch} to {end_epoch}")

scaler = GradScaler()


# === DICE METRIC ===
def dice_score(pred, target, num_classes=4, eps=1e-6):
    pred = torch.argmax(pred, dim=1)
    dice = []
    for k in range(num_classes):
        pred_k = (pred == k).float()
        target_k = (target == k).float()
        intersect = (pred_k * target_k).sum()
        denom = pred_k.sum() + target_k.sum()
        dice.append((2 * intersect / (denom + eps)).item())
    return dice


# === TRAINING LOOP ===
for epoch in range(start_epoch, end_epoch + 1):
    model.train()
    running_loss = 0.0
    for i, (images, masks) in enumerate(train_loader):
        if i % 500 == 0:
            print(f"Loaded batch {i}/{len(train_loader)}")
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
    avg_loss = running_loss / len(train_loader.dataset)
    print(f"[INFO] Epoch {epoch} completed. Avg Train Loss: {avg_loss:.4f}")
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }, ckpt_path)
    print(f"[INFO] Saved checkpoint: {ckpt_path} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # === VALIDATION ===
    model.eval()
    val_loss = 0.0
    dice_accum = np.zeros(NUM_CLASSES)
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)
            dice = dice_score(outputs, masks, NUM_CLASSES)
            dice_accum += np.array(dice) * images.size(0)
    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_val_dice = dice_accum / len(val_loader.dataset)
    print(f"[INFO] Epoch {epoch} validation loss: {avg_val_loss:.4f}")
    print(f"[INFO] Epoch {epoch} validation DICE: {[f'{d:.4f}' for d in avg_val_dice]}")
    model.train()

print("[INFO] Job finished")
