#!/usr/bin/env python3
"""
evaluate.py
Evaluates multi-label CheXpert classification checkpoints and computes AUROC scores.

Usage:
    python evaluate.py --data-dir /path/to/chexpert --checkpoint checkpoints/checkpoint_epoch_30.pth
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from train_baseline import CheXpertDataset, CHEXPERT_LABELS

COMPETITION_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CheXpert AUROC")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--val-csv", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="swin_base_patch4_window7_224")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    val_csv = args.val_csv or os.path.join(args.data_dir, "valid.csv")
    val_img_dir = os.path.join(args.data_dir, "valid")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_set = CheXpertDataset(val_csv, val_img_dir, transform=transform, is_train=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Loading model: {args.model_name} from {args.checkpoint}")
    model = timm.create_model(args.model_name, pretrained=False, num_classes=len(CHEXPERT_LABELS))
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating CheXpert"):
            images = images.to(args.device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    print("\n" + "=" * 65)
    print("CHEXPERT MULTI-LABEL AUROC EVALUATION")
    print("=" * 65)

    aurocs = {}
    for i, label in enumerate(CHEXPERT_LABELS):
        try:
            auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
            aurocs[label] = auc
            marker = " *" if label in COMPETITION_LABELS else ""
            print(f"  {label:<30}: AUROC = {auc:.4f}{marker}")
        except ValueError:
            print(f"  {label:<30}: AUROC = N/A (single class present)")

    comp_scores = [aurocs[l] for l in COMPETITION_LABELS if l in aurocs]
    if comp_scores:
        print("-" * 65)
        print(f"  {'Mean Leaderboard 5 Pathologies':<30}: AUROC = {np.mean(comp_scores):.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
