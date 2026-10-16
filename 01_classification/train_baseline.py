#!/usr/bin/env python3
"""
train_baseline.py
Baseline multi-label classification training on CheXpert using Swin Transformer Base.

Usage:
    python train_baseline.py --data-dir /path/to/chexpert --epochs 30 --batch-size 32
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


class CheXpertDataset(Dataset):
    def __init__(self, csv_file: str, img_base: str, transform=None, is_train: bool = True):
        self.df = pd.read_csv(csv_file)
        self.img_base = img_base
        self.transform = transform
        self.is_train = is_train

        # Fill NaNs with 0
        for col in CHEXPERT_LABELS:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0).astype(float)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # PATH NOTE: CSV 'Path' values look like 'CheXpert-v1.0/train/patientX/...'; this prefix is stripped
        # and the rest is joined to the image root. If your download uses another prefix
        # (e.g. 'CheXpert-v1.0-small/train/'), change it here.
        prefix = 'CheXpert-v1.0/train/' if self.is_train else 'CheXpert-v1.0/valid/'
        rel_path = row["Path"].replace(prefix, '', 1)
        img_path = os.path.join(self.img_base, rel_path)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        labels = row[CHEXPERT_LABELS].values.astype(float)
        labels = torch.tensor(labels, dtype=torch.float32)
        return image, labels


def get_latest_checkpoint(ckpt_dir: str):
    ckpts = glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch_*.pth"))
    if not ckpts:
        return None
    def extract_epoch(p):
        try:
            return int(os.path.basename(p).split('_')[-1].split('.')[0])
        except (ValueError, IndexError):
            return -1
    return max(ckpts, key=extract_epoch)


def parse_args():
    parser = argparse.ArgumentParser(description="CheXpert Swin Transformer Baseline Training")
    parser.add_argument("--data-dir", type=str, required=True, help="Base directory of CheXpert dataset")
    parser.add_argument("--train-csv", type=str, default=None, help="Path to train.csv (defaults to <data-dir>/train.csv)")
    parser.add_argument("--val-csv", type=str, default=None, help="Path to valid.csv (defaults to <data-dir>/valid.csv)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--model-name", type=str, default="swin_base_patch4_window7_224")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_csv = args.train_csv or os.path.join(args.data_dir, "train.csv")
    val_csv = args.val_csv or os.path.join(args.data_dir, "valid.csv")
    train_img_dir = os.path.join(args.data_dir, "train")
    val_img_dir = os.path.join(args.data_dir, "valid")

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"[{datetime.datetime.now()}] Initializing CheXpert Baseline Training...")
    train_set = CheXpertDataset(train_csv, train_img_dir, transform=transform, is_train=True)
    val_set = CheXpertDataset(val_csv, val_img_dir, transform=transform, is_train=False)

    print(f"Loaded {len(train_set)} train images and {len(val_set)} validation images.")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True if "cuda" in args.device else False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True if "cuda" in args.device else False)

    model = timm.create_model(args.model_name, pretrained=True, num_classes=len(CHEXPERT_LABELS))
    if torch.cuda.device_count() > 1 and "cuda" in args.device:
        print(f"Enabling DataParallel across {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)
    model.to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=("cuda" in args.device))

    start_epoch = 1
    if args.resume:
        latest = get_latest_checkpoint(args.checkpoint_dir)
        if latest:
            print(f"Resuming from: {latest}")
            ckpt = torch.load(latest, map_location="cpu")
            if hasattr(model, 'module'):
                model.module.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1

    for epoch in range(start_epoch, args.epochs + 1):
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

        epoch_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch} complete | Average Train Loss: {epoch_loss:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save({
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss
        }, ckpt_path)


if __name__ == "__main__":
    main()
