#!/usr/bin/env python3
"""
min_max_analysis.py
Identifies, ranks, and visualizes the highest-scoring and lowest-scoring segmentation predictions
to audit model strengths and edge-case failure modes.

Usage:
    python analysis/min_max_analysis.py \
        --img-dir /path/to/chexpert --mask-dir /path/to/chexmask --test-list test_files.txt \
        --checkpoint checkpoints/best_model.pth --output-dir ./analysis_results
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from tqdm import tqdm

from models import SwinTransformerSys
from data import CheXmaskSegDataset
from train import compute_dice_score

CLASS_COLORS = {
    0: (0, 0, 0),        # Background: transparent/black
    1: (0, 255, 0),      # Right Lung: Green
    2: (0, 0, 255),      # Left Lung: Blue
    3: (255, 0, 0)       # Heart: Red
}


def mask_to_color(mask_np: np.ndarray) -> np.ndarray:
    """Converts a class mask into an RGB overlay."""
    h, w = mask_np.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for c, rgb in CLASS_COLORS.items():
        color_mask[mask_np == c] = rgb
    return color_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Best & Worst Prediction Visualizer")
    parser.add_argument("--img-dir", type=str, required=True)
    parser.add_argument("--mask-dir", type=str, required=True)
    parser.add_argument("--test-list", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./analysis_results")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = CheXmaskSegDataset(
        img_base=args.img_dir,
        mask_base=args.mask_dir,
        file_list=args.test_list,
        img_size=args.img_size
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    model = SwinTransformerSys(img_size=args.img_size, num_classes=4)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    sample_metrics = []

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Evaluating predictions"):
            img_tensor, mask_tensor = dataset[i]
            img_batch = img_tensor.unsqueeze(0).to(args.device)
            mask_batch = mask_tensor.unsqueeze(0).to(args.device)

            pred_logits = model(img_batch)
            dice_scores = compute_dice_score(pred_logits, mask_batch, num_classes=4)
            mean_organ_dice = np.mean(dice_scores[1:])
            pred_mask = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy()

            sample_metrics.append({
                "index": i,
                "dice_scores": dice_scores,
                "mean_organ_dice": mean_organ_dice,
                "img_tensor": img_tensor.cpu(),
                "gt_mask": mask_tensor.numpy(),
                "pred_mask": pred_mask
            })

    # Sort by performance
    sample_metrics.sort(key=lambda x: x["mean_organ_dice"])
    worst_samples = sample_metrics[:3]
    best_samples = sample_metrics[-3:]

    # Plot Best & Worst Visualizations
    fig, axes = plt.subplots(6, 3, figsize=(12, 20))
    legend_patches = [
        mpatches.Patch(color='green', label='Right Lung'),
        mpatches.Patch(color='blue', label='Left Lung'),
        mpatches.Patch(color='red', label='Heart')
    ]
    fig.legend(handles=legend_patches, loc='upper center', ncol=3, fontsize=12)

    rows_to_plot = [("Worst", worst_samples), ("Best", best_samples)]
    row_idx = 0

    for label_prefix, sample_list in rows_to_plot:
        for item in sample_list:
            img_np = item["img_tensor"].permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
            gt_color = mask_to_color(item["gt_mask"])
            pred_color = mask_to_color(item["pred_mask"])

            # Input image
            axes[row_idx, 0].imshow(img_np)
            axes[row_idx, 0].set_title(f"{label_prefix} | Mean Dice: {item['mean_organ_dice']:.3f}")
            axes[row_idx, 0].axis('off')

            # Ground Truth
            axes[row_idx, 1].imshow(img_np)
            axes[row_idx, 1].imshow(gt_color, alpha=0.45)
            axes[row_idx, 1].set_title("Ground Truth Mask")
            axes[row_idx, 1].axis('off')

            # Prediction
            axes[row_idx, 2].imshow(img_np)
            axes[row_idx, 2].imshow(pred_color, alpha=0.45)
            axes[row_idx, 2].set_title(f"Predicted Mask")
            axes[row_idx, 2].axis('off')

            row_idx += 1

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_plot = os.path.join(args.output_dir, "best_worst_predictions.png")
    plt.savefig(out_plot, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nVisual inspection saved: {out_plot}")


if __name__ == "__main__":
    main()
