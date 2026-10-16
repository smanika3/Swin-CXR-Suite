#!/usr/bin/env python3
"""
evaluate.py
Evaluates trained Swin-Unet-V2 model on CheXmask test dataset, calculating per-organ Dice coefficients.

Dice is computed per batch (pixels pooled over the batch) and averaged weighted by batch size, as in the
original test script; batch size 96 reproduces the reported numbers. The reported "Mean Dice" (0.9619) is the
mean over all 4 classes INCLUDING background.

Usage:
    python evaluate.py --img-dir /path/to/chexpert/train --mask-dir /path/to/chexmask-masks/train \
        --test-list /path/to/chexmask-masks/test_files.txt --checkpoint checkpoints/checkpoint_epoch_11.pth
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import SwinTransformerSys
from data import CheXmaskSegDataset
from train import compute_dice_score

ORGAN_NAMES = ["Background", "Right Lung", "Left Lung", "Heart"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Swin-Unet on CheXmask")
    parser.add_argument("--img-dir", type=str, required=True)
    parser.add_argument("--mask-dir", type=str, required=True)
    parser.add_argument("--test-list", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=96, help="Original test run used 96 (affects batch-pooled Dice)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    test_dataset = CheXmaskSegDataset(
        img_base=args.img_dir,
        mask_base=args.mask_dir,
        file_list=args.test_list,
        img_size=args.img_size
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = SwinTransformerSys(img_size=args.img_size, num_classes=4)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    total_dice = np.zeros(4)
    sample_count = 0

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating Test Set"):
            images = images.to(args.device)
            masks = masks.to(args.device)

            outputs = model(images)
            batch_dice = compute_dice_score(outputs, masks, num_classes=4)
            total_dice += np.array(batch_dice) * len(images)
            sample_count += len(images)

    mean_dice_per_class = total_dice / sample_count
    mean_dice = np.mean(mean_dice_per_class)
    mean_organ_dice = np.mean(mean_dice_per_class[1:])

    print("\n" + "=" * 60)
    print("CHEXMASK TEST SET EVALUATION RESULTS")
    print("=" * 60)
    for i, name in enumerate(ORGAN_NAMES):
        print(f"  {name:15s}: Dice = {mean_dice_per_class[i]:.4f}")
    print("-" * 60)
    print(f"  Mean Dice, 4 classes incl. background (reported metric): {mean_dice:.4f}")
    print(f"  Mean Organ Dice, right lung / left lung / heart:        {mean_organ_dice:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
