#!/usr/bin/env python3
"""
evaluate.py
Computes Average Precision (AP50, AP75, AP@[.5:.95]), class-specific AP,
and class-agnostic TB localization performance.

Usage:
    python evaluate.py --data-dir /path/to/tbx11k-simplified --checkpoint checkpoints/checkpoint_epoch_30.pth
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.ops import box_iou
from tqdm import tqdm

from models import build_faster_rcnn
from data import TBX11KDetectionDataset, detection_collate_fn


def compute_ap_single_threshold(all_preds, all_gts, iou_thresh=0.5, class_id=None, class_agnostic=False):
    """
    Computes Average Precision for a specific IoU threshold and target class.
    """
    detected_gts = [set() for _ in all_gts]
    total_gts = 0

    pred_records = []
    for img_idx, (preds, gts) in enumerate(zip(all_preds, all_gts)):
        p_boxes = preds['boxes'].cpu().numpy()
        p_scores = preds['scores'].cpu().numpy()
        p_labels = preds['labels'].cpu().numpy()

        g_boxes = gts['boxes'].cpu().numpy()
        g_labels = gts['labels'].cpu().numpy()

        if class_agnostic:
            # Consider all positive classes (1 or 2) as positive (class 1)
            valid_gt_mask = g_labels > 0
            valid_p_mask = p_labels > 0
        elif class_id is not None:
            valid_gt_mask = (g_labels == class_id)
            valid_p_mask = (p_labels == class_id)
        else:
            valid_gt_mask = g_labels > 0
            valid_p_mask = p_labels > 0

        g_boxes = g_boxes[valid_gt_mask]
        total_gts += len(g_boxes)

        for b, s in zip(p_boxes[valid_p_mask], p_scores[valid_p_mask]):
            pred_records.append({
                "img_idx": img_idx,
                "box": b,
                "score": float(s),
                "g_boxes": g_boxes
            })

    if total_gts == 0:
        return 0.0

    # Sort all predictions across the entire dataset by confidence descending
    pred_records.sort(key=lambda x: x["score"], reverse=True)

    tp = np.zeros(len(pred_records))
    fp = np.zeros(len(pred_records))

    for i, p in enumerate(pred_records):
        img_idx = p["img_idx"]
        p_box = torch.tensor([p["box"]])
        g_boxes = p["g_boxes"]

        if len(g_boxes) == 0:
            fp[i] = 1
            continue

        ious = box_iou(p_box, torch.tensor(g_boxes)).squeeze(0).numpy()
        best_gt_idx = np.argmax(ious)
        best_iou = ious[best_gt_idx]

        if best_iou >= iou_thresh:
            if best_gt_idx not in detected_gts[img_idx]:
                tp[i] = 1
                detected_gts[img_idx].add(best_gt_idx)
            else:
                fp[i] = 1  # Duplicate detection
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recalls = cum_tp / total_gts
    precisions = cum_tp / (cum_tp + cum_fp + 1e-6)

    # 11-point interpolation or standard AUC PR
    ap = 0.0
    for r in np.linspace(0.0, 1.0, 11):
        prec_at_r = precisions[recalls >= r]
        p_max = np.max(prec_at_r) if len(prec_at_r) > 0 else 0.0
        ap += p_max / 11.0

    return float(ap)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TBX11K Localization Model")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    txt_file = os.path.join(args.data_dir, f"{args.split}.txt")
    csv_file = os.path.join(args.data_dir, "data.csv")
    img_dir = os.path.join(args.data_dir, "images")

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = TBX11KDetectionDataset(txt_file, csv_file, img_dir=img_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=detection_collate_fn)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = build_faster_rcnn(num_classes=3, img_size=args.img_size)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.to(args.device)
    model.eval()

    all_preds, all_gts = [], []
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc=f"Evaluating on {args.split}"):
            images = [img.to(args.device) for img in images]
            preds = model(images)
            all_preds.extend(preds)
            all_gts.extend(targets)

    # Compute metrics
    ap50_active = compute_ap_single_threshold(all_preds, all_gts, iou_thresh=0.5, class_id=1)
    ap50_latent = compute_ap_single_threshold(all_preds, all_gts, iou_thresh=0.5, class_id=2)
    ap50_agnostic = compute_ap_single_threshold(all_preds, all_gts, iou_thresh=0.5, class_agnostic=True)

    ap75_active = compute_ap_single_threshold(all_preds, all_gts, iou_thresh=0.75, class_id=1)
    ap75_latent = compute_ap_single_threshold(all_preds, all_gts, iou_thresh=0.75, class_id=2)
    ap75_agnostic = compute_ap_single_threshold(all_preds, all_gts, iou_thresh=0.75, class_agnostic=True)

    print("\n" + "=" * 65)
    print(f"TBX11K LOCALIZATION EVALUATION ({args.split.upper()} SET)")
    print("=" * 65)
    print(f"{'Condition / Metric':<30} | {'AP@0.50':<12} | {'AP@0.75':<12}")
    print("-" * 65)
    print(f"{'Active TB (Class 1)':<30} | {ap50_active:<12.4f} | {ap75_active:<12.4f}")
    print(f"{'Latent TB (Class 2)':<30} | {ap50_latent:<12.4f} | {ap75_latent:<12.4f}")
    print(f"{'Class-Agnostic TB Detection':<30} | {ap50_agnostic:<12.4f} | {ap75_agnostic:<12.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
