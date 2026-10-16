#!/usr/bin/env python3
"""
visualize.py
Side-by-side visualizer comparing Ground Truth bounding boxes with model predictions.

Usage:
    python visualize.py --data-dir /path/to/tbx11k-simplified --checkpoint checkpoints/checkpoint_epoch_30.pth --num-samples 5
"""

import os
import random
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms

from models import build_faster_rcnn
from data import TBX11KDetectionDataset

CLASS_NAMES = {1: "Active TB", 2: "Latent TB"}
CLASS_COLORS = {1: "red", 2: "blue"}


def parse_args():
    parser = argparse.ArgumentParser(description="Side-by-side Visualizer")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--output-dir", type=str, default="./visualizations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    txt_file = os.path.join(args.data_dir, f"{args.split}.txt")
    csv_file = os.path.join(args.data_dir, "data.csv")
    img_dir = os.path.join(args.data_dir, "images")

    dataset = TBX11KDetectionDataset(
        txt_file,
        csv_file,
        img_dir=img_dir,
        transform=transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    )

    model = build_faster_rcnn(num_classes=3, img_size=512, pretrained=False)  # weights come from the checkpoint
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.to(args.device)
    model.eval()

    sample_indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    for idx in sample_indices:
        img_tensor, target = dataset[idx]
        with torch.no_grad():
            pred = model([img_tensor.to(args.device)])[0]

        img_np = img_tensor.permute(1, 2, 0).numpy()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

        # Ground Truth
        ax1.imshow(img_np)
        ax1.set_title("Ground Truth", fontsize=14, fontweight="bold")
        ax1.axis("off")
        for box, label in zip(target["boxes"].numpy(), target["labels"].numpy()):
            x1, y1, x2, y2 = box
            color = CLASS_COLORS.get(label, "green")
            name = CLASS_NAMES.get(label, f"Class {label}")
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none")
            ax1.add_patch(rect)
            ax1.text(x1, max(0, y1 - 4), name, color=color, fontsize=10, backgroundcolor="black")

        # Predictions
        ax2.imshow(img_np)
        ax2.set_title(f"Predictions (Threshold >= {args.score_thresh})", fontsize=14, fontweight="bold")
        ax2.axis("off")
        for box, score, label in zip(pred["boxes"].cpu().numpy(), pred["scores"].cpu().numpy(), pred["labels"].cpu().numpy()):
            if score >= args.score_thresh:
                x1, y1, x2, y2 = box
                color = CLASS_COLORS.get(label, "yellow")
                name = CLASS_NAMES.get(label, f"Class {label}")
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, linestyle="--", facecolor="none")
                ax2.add_patch(rect)
                ax2.text(x1, max(0, y1 - 4), f"{name}: {score:.2f}", color=color, fontsize=10, backgroundcolor="black")

        save_p = os.path.join(args.output_dir, f"vis_sample_{idx}.png")
        plt.tight_layout()
        plt.savefig(save_p, dpi=200)
        plt.close()
        print(f"Saved visualization: {save_p}")


if __name__ == "__main__":
    main()
