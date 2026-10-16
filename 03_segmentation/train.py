#!/usr/bin/env python3
"""
train.py
Train Swin-Unet-V2 for multi-class anatomical segmentation on CheXmask.

Usage:
    python train.py \
        --img-dir /path/to/chexpert/train --mask-dir /path/to/chexmask-masks/train \
        --train-list /path/to/chexmask-masks/train_split.txt \
        --val-img-dir /path/to/chexpert/valid --val-mask-dir /path/to/chexmask-masks/valid \
        --val-list /path/to/chexmask-masks/valid_files.txt
"""

import os
import glob
import argparse
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from models import SwinTransformerSys
from data import CheXmaskSegDataset


def compute_dice_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 4, eps: float = 1e-6):
    """
    Computes per-class Dice coefficient.
    preds: [B, C, H, W]
    targets: [B, H, W]
    """
    pred_classes = torch.argmax(preds, dim=1)
    dice_per_class = []
    for c in range(num_classes):
        p = (pred_classes == c).float()
        t = (targets == c).float()
        intersection = (p * t).sum()
        cardinality = p.sum() + t.sum()
        score = (2.0 * intersection / (cardinality + eps)).item()
        dice_per_class.append(score)
    return dice_per_class


def get_latest_checkpoint(ckpt_dir: str):
    ckpts = glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch_*.pth"))
    if not ckpts:
        return None
    def extract_epoch(path):
        try:
            return int(os.path.basename(path).split("_")[-1].split(".")[0])
        except (ValueError, IndexError):
            return -1
    return max(ckpts, key=extract_epoch)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Swin-Unet on CheXmask")
    parser.add_argument("--img-dir", type=str, required=True, help="Training image directory (CheXpert train/)")
    parser.add_argument("--mask-dir", type=str, required=True, help="Training mask directory (chexmask-masks/train)")
    parser.add_argument("--train-list", type=str, required=True, help="Text file listing training images")
    parser.add_argument("--val-list", type=str, required=True, help="Text file listing validation images")
    parser.add_argument("--val-img-dir", type=str, default=None,
                        help="Validation image directory (CheXpert valid/). Defaults to --img-dir")
    parser.add_argument("--val-mask-dir", type=str, default=None,
                        help="Validation mask directory (chexmask-masks/valid). Defaults to --mask-dir")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=11,
                        help="Last epoch to train. The reported run stopped at epoch 11 (it was configured for 40)")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.val_img_dir = args.val_img_dir or args.img_dir
    args.val_mask_dir = args.val_mask_dir or args.mask_dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"[{datetime.datetime.now()}] Initializing Swin-Unet Segmentation Training...")
    print(f"Image Resolution: {args.img_size}x{args.img_size} | Classes: {args.num_classes} | Device: {args.device}")

    # Datasets
    train_dataset = CheXmaskSegDataset(
        img_base=args.img_dir,
        mask_base=args.mask_dir,
        file_list=args.train_list,
        img_size=args.img_size
    )
    val_dataset = CheXmaskSegDataset(
        img_base=args.val_img_dir,
        mask_base=args.val_mask_dir,
        file_list=args.val_list,
        img_size=args.img_size
    )

    print(f"Dataset summary: {len(train_dataset)} train samples, {len(val_dataset)} validation samples.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True if "cuda" in args.device else False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True if "cuda" in args.device else False)

    # Model
    model = SwinTransformerSys(img_size=args.img_size, num_classes=args.num_classes)
    if torch.cuda.device_count() > 1 and "cuda" in args.device:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)
    model.to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=("cuda" in args.device))

    start_epoch = 1
    best_val_dice = 0.0
    if args.resume:
        latest = get_latest_checkpoint(args.checkpoint_dir)
        if latest:
            print(f"Resuming from: {latest}")
            checkpoint = torch.load(latest, map_location="cpu")
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            best_val_dice = checkpoint.get("best_val_dice", 0.0)
            start_epoch = checkpoint.get("epoch", 0) + 1
            print(f"Resuming at epoch {start_epoch} (best val mean organ Dice so far: {best_val_dice:.4f})")

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")

        for images, masks in pbar:
            images = images.to(args.device)
            masks = masks.to(args.device)

            optimizer.zero_grad()
            with autocast(enabled=("cuda" in args.device)):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= max(1, len(train_dataset))

        # Validation (fp32, loss and per-batch Dice weighted by batch size, as in the original run)
        model.eval()
        val_loss = 0.0
        val_dice_scores = [0.0] * args.num_classes

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                images = images.to(args.device)
                masks = masks.to(args.device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item() * images.size(0)
                batch_dice = compute_dice_score(outputs, masks, num_classes=args.num_classes)
                for c in range(args.num_classes):
                    val_dice_scores[c] += batch_dice[c] * images.size(0)

        n_val = max(1, len(val_dataset))
        val_loss /= n_val
        val_dice_scores = [s / n_val for s in val_dice_scores]
        mean_dice = sum(val_dice_scores) / len(val_dice_scores)  # 4 classes incl. background (reported metric)
        mean_organ_dice = sum(val_dice_scores[1:]) / 3.0  # Average across Right Lung, Left Lung, Heart

        print(f"\n--- Epoch {epoch} Results ---")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Dice -> Background: {val_dice_scores[0]:.4f} | Right Lung: {val_dice_scores[1]:.4f} | "
              f"Left Lung: {val_dice_scores[2]:.4f} | Heart: {val_dice_scores[3]:.4f}")
        print(f"Val Mean Dice (4 classes incl. background, reported metric): {mean_dice:.4f}")
        print(f"Val Mean Organ Dice (right lung, left lung, heart): {mean_organ_dice:.4f}")

        is_best = mean_organ_dice > best_val_dice
        if is_best:
            best_val_dice = mean_organ_dice

        # Checkpointing
        ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save({
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mean_dice": mean_dice,
            "mean_organ_dice": mean_organ_dice,
            "best_val_dice": best_val_dice,
            "scaler_state_dict": scaler.state_dict()
        }, ckpt_path)

        if is_best:
            best_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(state_dict, best_path)
            print(f"New best model saved with val Mean Organ Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
