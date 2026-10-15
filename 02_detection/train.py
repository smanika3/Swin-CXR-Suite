#!/usr/bin/env python3
"""
train.py
Train a Faster R-CNN detector with a Swin Transformer FPN backbone on TBX11K.

Usage:
    python train.py --data-dir /path/to/tbx11k --epochs 30 --batch-size 8 --lr 0.005
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

from models import make_fasterrcnn_swin
from datasets import TBX11KDetectionDataset, detection_collate_fn


def get_latest_checkpoint(ckpt_dir: str):
    """Finds the checkpoint with the highest epoch number."""
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
    parser = argparse.ArgumentParser(description="Train Swin Faster R-CNN on TBX11K")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to TBX11K root containing train/val/test folders and annotations/")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--model-name", type=str, default="swin_base_patch4_window7_224",
                        help="Swin Transformer model name from timm")
    parser.add_argument("--img-size", type=int, default=800,
                        help="Input image resolution")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of foreground classes (default: 2 for Active/Obsolete TB)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Total number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0001,
                        help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint if available")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on (cuda/cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"[{datetime.datetime.now()}] Initializing Training on {args.device}...")
    print(f"Model: {args.model_name} | Image Size: {args.img_size} | Epochs: {args.epochs}")

    # Dataset paths
    train_img_dir = os.path.join(args.data_dir, "train", "img")
    train_json = os.path.join(args.data_dir, "annotations", "train.json")
    val_img_dir = os.path.join(args.data_dir, "val", "img")
    val_json = os.path.join(args.data_dir, "annotations", "val.json")

    # Data transforms
    data_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = TBX11KDetectionDataset(train_img_dir, train_json, transform=data_transform)
    print(f"Loaded {len(train_set)} training images.")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=True if "cuda" in args.device else False
    )

    # Instantiate model
    model = make_fasterrcnn_swin(
        num_classes=args.num_classes,
        model_name=args.model_name,
        pretrained=True,
        img_size=args.img_size
    )
    model.to(args.device)

    # Optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    start_epoch = 1
    if args.resume:
        latest = get_latest_checkpoint(args.checkpoint_dir)
        if latest:
            print(f"Resuming from: {latest}")
            ckpt = torch.load(latest, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in pbar:
            if batch is None:
                continue
            images, targets = batch
            images = [img.to(args.device) for img in images]
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += losses.item()
            pbar.set_postfix({"loss": f"{losses.item():.4f}"})

        lr_scheduler.step()
        epoch_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch} finished. Average Loss: {epoch_loss:.4f} | LR: {lr_scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss
        }, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()
