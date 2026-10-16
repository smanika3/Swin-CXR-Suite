#!/usr/bin/env python3
"""
visualize.py
Visualize model predictions vs. Ground Truth bounding boxes on TBX11K chest X-rays.

Usage:
    python visualize.py --data-dir /path/to/tbx11k --checkpoint /path/to/checkpoint.pth --num-samples 5
"""

import os
import argparse
import random
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms

from models import make_fasterrcnn_swin
from datasets import TBX11KDetectionDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize TB Bounding Box Predictions")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--score-thresh", type=float, default=0.25)
    parser.add_argument("--output-dir", type=str, default="./visualizations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    img_dir = os.path.join(args.data_dir, args.split, "img")
    json_path = os.path.join(args.data_dir, "annotations", f"{args.split}.json")

    # Load dataset without normalization for clean visual plotting
    dataset = TBX11KDetectionDataset(img_dir, json_path, transform=transforms.ToTensor())

    # Load model
    model = make_fasterrcnn_swin(num_classes=2)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.to(args.device)
    model.eval()

    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
    print(f"Visualizing {len(indices)} samples to {args.output_dir}...")

    for i, idx in enumerate(indices):
        img_tensor, target = dataset[idx]
        with torch.no_grad():
            pred = model([img_tensor.to(args.device)])[0]

        pred_boxes = pred["boxes"].cpu().numpy()
        pred_scores = pred["scores"].cpu().numpy()
        gt_boxes = target["boxes"].numpy()

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        img_np = img_tensor.permute(1, 2, 0).numpy()
        ax.imshow(img_np)

        # Plot Ground Truth (Green)
        for b in gt_boxes:
            x1, y1, x2, y2 = b
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

        # Plot Predictions (Red dashed)
        for b, score in zip(pred_boxes, pred_scores):
            if score >= args.score_thresh:
                x1, y1, x2, y2 = b
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', linestyle='--', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, max(0, y1 - 5), f"{score:.2f}", color='red', fontsize=10, backgroundcolor='black')

        plt.title(f"Sample {idx}: Ground Truth (Green) vs. Prediction (Red, >={args.score_thresh})")
        plt.axis('off')
        save_path = os.path.join(args.output_dir, f"sample_{idx}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
