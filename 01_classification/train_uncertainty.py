#!/usr/bin/env python3
"""
train_uncertainty.py
CheXpert training with clinical uncertainty policies (U-Ones, U-Zeros, U-Ignore)
for the 5 competition leaderboard pathologies.

Usage:
    python train_uncertainty.py --data-dir /path/to/chexpert --epochs 30
"""

import os
import glob
import argparse
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import timm
import pandas as pd
from PIL import Image
from tqdm import tqdm

CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]

# Paper recommended uncertainty policies:
UNCERTAINTY_POLICIES = {
    2: "U-Ones",    # Cardiomegaly
    5: "U-Ignore",  # Edema
    6: "U-Ignore",  # Consolidation
    8: "U-Ones",    # Atelectasis
    10: "U-Zeros"   # Pleural Effusion
}


def apply_policy(label: float, policy: str) -> float:
    if policy == "U-Ones":
        return 1.0 if label == -1 else label
    elif policy == "U-Zeros":
        return 0.0 if label == -1 else label
    elif policy == "U-Ignore":
        return 0.0 if label == -1 else label  # In binary BCE, ignore can be treated as negative or masked
    return label


class CheXpertUncertaintyDataset(Dataset):
    def __init__(self, csv_file: str, img_base: str, transform=None, is_train: bool = True):
        self.df = pd.read_csv(csv_file)
        self.img_base = img_base
        self.transform = transform
        self.is_train = is_train

        for i, col in enumerate(CHEXPERT_LABELS):
            if col in self.df.columns:
                policy = UNCERTAINTY_POLICIES.get(i, "U-Ignore")
                self.df[col] = self.df[col].fillna(0).apply(lambda x: apply_policy(x, policy)).astype(float)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        prefix = 'CheXpert-v1.0/train/' if self.is_train else 'CheXpert-v1.0/valid/'
        rel_path = row["Path"].replace(prefix, '', 1)
        img_path = os.path.join(self.img_base, rel_path)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        labels = row[CHEXPERT_LABELS].values.astype(float)
        labels = torch.tensor(labels, dtype=torch.float32)
        return image, labels


def parse_args():
    parser = argparse.ArgumentParser(description="CheXpert Training with Uncertainty Policies")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints_uncertainty")
    parser.add_argument("--model-name", type=str, default="swin_base_patch4_window7_224")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_csv = os.path.join(args.data_dir, "train.csv")
    val_csv = os.path.join(args.data_dir, "valid.csv")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"[{datetime.datetime.now()}] Initializing Uncertainty Policy Training...")
    train_set = CheXpertUncertaintyDataset(train_csv, os.path.join(args.data_dir, "train"), transform=transform, is_train=True)
    val_set = CheXpertUncertaintyDataset(val_csv, os.path.join(args.data_dir, "valid"), transform=transform, is_train=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = timm.create_model(args.model_name, pretrained=True, num_classes=len(CHEXPERT_LABELS))
    if torch.cuda.device_count() > 1 and "cuda" in args.device:
        model = nn.DataParallel(model)
    model.to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=("cuda" in args.device))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for images, labels in pbar:
            images = images.to(args.device)
            labels = labels.to(args.device)

            optimizer.zero_grad()
            with autocast(enabled=("cuda" in args.device)):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Epoch {epoch} finished. Checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()
