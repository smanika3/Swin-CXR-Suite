#!/usr/bin/env python3
"""
finetune_libauc.py
Stage-2 fine-tuning on CheXpert using LibAUC's MultiLabelAUCMLoss
for direct surrogate Area Under the ROC Curve (AUROC) maximization.

Usage:
    python finetune_libauc.py --data-dir /path/to/chexpert --checkpoint checkpoints/checkpoint_epoch_10.pth
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import timm
from tqdm import tqdm

try:
    from libauc.losses import MultiLabelAUCMLoss
    from libauc.optimizers import PESG
except ImportError:
    MultiLabelAUCMLoss = None
    PESG = None

from train_uncertainty import CheXpertUncertaintyDataset, CHEXPERT_LABELS


def parse_args():
    parser = argparse.ArgumentParser(description="CheXpert LibAUC AUROC Maximization Fine-Tuning")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Stage 1 checkpoint to fine-tune from")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints_libauc")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    if MultiLabelAUCMLoss is None:
        raise ImportError("libauc is not installed. Please run: pip install libauc")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train_csv = os.path.join(args.data_dir, "train.csv")
    val_csv = os.path.join(args.data_dir, "valid.csv")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = CheXpertUncertaintyDataset(train_csv, os.path.join(args.data_dir, "train"), transform=transform, is_train=True)
    val_set = CheXpertUncertaintyDataset(val_csv, os.path.join(args.data_dir, "valid"), transform=transform, is_train=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=len(CHEXPERT_LABELS))
    print(f"Loading Stage-1 weights: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(args.device)

    # LibAUC MultiLabelAUCMLoss
    loss_fn = MultiLabelAUCMLoss(num_classes=len(CHEXPERT_LABELS), margin=args.margin)
    optimizer = PESG(model, loss_fn=loss_fn, lr=args.lr, momentum=0.9, weight_decay=1e-5)

    print("Beginning LibAUC direct AUROC optimization...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"LibAUC Fine-tune Epoch {epoch}/{args.epochs}")

        for images, labels in pbar:
            images = images.to(args.device)
            labels = labels.to(args.device)

            logits = torch.sigmoid(model(images))
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"auc_loss": f"{loss.item():.4f}"})

        ckpt_path = os.path.join(args.checkpoint_dir, f"libauc_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Epoch {epoch} finished. Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
