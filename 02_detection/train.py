#!/usr/bin/env python3
"""
train.py
Train a Faster R-CNN detector with a Swin Transformer FPN backbone on TBX11K (Tabular format).

Usage:
    python train.py --data-dir /path/to/tbx11k-simplified --epochs 30 --batch-size 8 --lr 1e-4
"""

import os
import glob
import argparse
import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models import build_faster_rcnn
from data import TBX11KDetectionDataset, detection_collate_fn


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
    parser = argparse.ArgumentParser(description="Train Swin Faster R-CNN on TBX11K (Simplified)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path containing data.csv, images/, and train.txt/val.txt/test.txt")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_txt = os.path.join(args.data_dir, "train.txt")
    val_txt = os.path.join(args.data_dir, "val.txt")
    csv_path = os.path.join(args.data_dir, "data.csv")
    img_dir = os.path.join(args.data_dir, "images")

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    print(f"[{datetime.datetime.now()}] Initializing TBX11K Localization Training...")
    train_dataset = TBX11KDetectionDataset(train_txt, csv_path, img_dir=img_dir, transform=transform)
    val_dataset = TBX11KDetectionDataset(val_txt, csv_path, img_dir=img_dir, transform=transform)

    print(f"Loaded {len(train_dataset)} train samples, {len(val_dataset)} val samples.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=True if "cuda" in args.device else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=True if "cuda" in args.device else False
    )

    # 3 classes: Background (0), Active TB (1), Latent TB (2)
    model = build_faster_rcnn(num_classes=3, img_size=args.img_size)
    model.to(args.device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 1
    if args.resume:
        latest = get_latest_checkpoint(args.checkpoint_dir)
        if latest:
            print(f"Resuming from checkpoint: {latest}")
            ckpt = torch.load(latest, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for images, targets in pbar:
            images = [img.to(args.device) for img in images]
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
            pbar.set_postfix({"loss": f"{losses.item():.4f}"})

        epoch_loss = running_loss / max(1, len(train_loader))

        # Validation loss. In eval() mode Faster R-CNN returns detections, not losses,
        # so (as in the original run) the model is put in train() mode under no_grad.
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(args.device) for img in images]
                targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]
                model.train()
                loss_dict = model(images, targets)
                model.eval()
                val_running_loss += sum(loss for loss in loss_dict.values()).item()
        val_loss = val_running_loss / max(1, len(val_loader))
        print(f"Epoch {epoch} finished. Train loss: {epoch_loss:.4f} | Val loss: {val_loss:.4f}")

        ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": epoch_loss,
            "val_loss": val_loss
        }, ckpt_path)


if __name__ == "__main__":
    main()
