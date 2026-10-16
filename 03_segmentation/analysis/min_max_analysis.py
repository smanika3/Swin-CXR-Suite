#!/usr/bin/env python3
"""
min_max_analysis.py
Identifies, ranks, and visualizes the highest-scoring and lowest-scoring segmentation predictions
to audit model strengths and edge-case failure modes.

Samples are ranked by per-image mean Dice over the 3 organs (background excluded), as in the original
best/worst scripts. Only scores are kept during the pass; the best/worst images are reloaded for plotting.

Usage:
    python analysis/min_max_analysis.py \
        --img-dir /path/to/chexpert --mask-dir /path/to/chexmask --test-list test_files.txt \
        --checkpoint checkpoints/best_model.pth --output-dir ./analysis_results
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow `python analysis/min_max_analysis.py` from the module root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import SwinTransformerSys
from data import CheXmaskSegDataset
from train import compute_dice_score

# Same colours as the original scripts / published figures (RGBA).
CLASS_COLORS = {
    0: (0, 0, 0, 0),         # Background: transparent
    1: (255, 0, 0, 128),     # Right Lung: Red
    2: (0, 255, 0, 128),     # Left Lung: Green
    3: (0, 0, 255, 128)      # Heart: Blue
}


def mask_to_color(mask_np: np.ndarray) -> np.ndarray:
    """Converts a class mask into an RGBA overlay."""
    h, w = mask_np.shape
    color_mask = np.zeros((h, w, 4), dtype=np.uint8)
    for c, rgba in CLASS_COLORS.items():
        color_mask[mask_np == c] = rgba
    return color_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Best & Worst Prediction Visualizer")
    parser.add_argument("--img-dir", type=str, required=True)
    parser.add_argument("--mask-dir", type=str, required=True)
    parser.add_argument("--test-list", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./analysis_results")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
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

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Pass 1: per-image scores only (no images kept in memory)
    sample_metrics = []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating predictions"):
            logits = model(images.to(args.device))
            masks = masks.to(args.device)
            for j in range(images.size(0)):
                dice_scores = compute_dice_score(logits[j:j + 1], masks[j:j + 1], num_classes=4)
                sample_metrics.append({
                    "index": len(sample_metrics),
                    "dice_scores": dice_scores,
                    "mean_organ_dice": np.mean(dice_scores[1:])
                })

    # Sort by performance (best first), as in the original script
    sample_metrics.sort(key=lambda x: x["mean_organ_dice"], reverse=True)
    best_samples = sample_metrics[:3]
    worst_samples = sample_metrics[-3:]

    # Pass 2: reload and re-predict only the samples we plot
    with torch.no_grad():
        for item in best_samples + worst_samples:
            img_tensor, mask_tensor = dataset[item["index"]]
            logits = model(img_tensor.unsqueeze(0).to(args.device))
            item["pred_mask"] = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            item["gt_mask"] = mask_tensor.numpy()
            img_path = os.path.join(args.img_dir, dataset.images[item["index"]])
            item["orig_img"] = np.array(Image.open(img_path).convert("RGB").resize((args.img_size, args.img_size)))

    # Plot Best & Worst Visualizations
    fig, axes = plt.subplots(6, 3, figsize=(12, 20))
    legend_patches = [
        mpatches.Patch(color=(1, 0, 0, 0.5), label='Right Lung'),
        mpatches.Patch(color=(0, 1, 0, 0.5), label='Left Lung'),
        mpatches.Patch(color=(0, 0, 1, 0.5), label='Heart')
    ]
    fig.legend(handles=legend_patches, loc='upper center', ncol=3, fontsize=12)

    rows_to_plot = [("Best", best_samples), ("Worst", worst_samples)]
    row_idx = 0

    for label_prefix, sample_list in rows_to_plot:
        for item in sample_list:
            img_np = item["orig_img"]
            gt_color = mask_to_color(item["gt_mask"])
            pred_color = mask_to_color(item["pred_mask"])

            # Input image
            axes[row_idx, 0].imshow(img_np)
            axes[row_idx, 0].set_title(f"{label_prefix} | Mean Dice: {item['mean_organ_dice']:.3f}")
            axes[row_idx, 0].axis('off')

            # Ground Truth
            axes[row_idx, 1].imshow(img_np)
            axes[row_idx, 1].imshow(gt_color, alpha=0.5)
            axes[row_idx, 1].set_title("Ground Truth Mask")
            axes[row_idx, 1].axis('off')

            # Prediction
            axes[row_idx, 2].imshow(img_np)
            axes[row_idx, 2].imshow(pred_color, alpha=0.5)
            d = item["dice_scores"]
            axes[row_idx, 2].set_title(f"Predicted Mask\nRL:{d[1]:.3f} LL:{d[2]:.3f} H:{d[3]:.3f}")
            axes[row_idx, 2].axis('off')

            row_idx += 1

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_plot = os.path.join(args.output_dir, "best_worst_predictions.png")
    plt.savefig(out_plot, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nVisual inspection saved: {out_plot}")


if __name__ == "__main__":
    main()
