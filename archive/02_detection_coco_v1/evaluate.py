#!/usr/bin/env python3
"""
evaluate.py
Evaluates TBX11K Swin Faster R-CNN checkpoints using COCO mAP and Free-Response ROC (FROC).

Usage:
    python evaluate.py --data-dir /path/to/tbx11k --checkpoint /path/to/checkpoint.pth
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.ops import box_iou
from tqdm import tqdm
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from models import make_fasterrcnn_swin
from datasets import TBX11KDetectionDataset, detection_collate_fn


def run_inference(model, loader, device):
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Running Validation"):
            if not batch or len(batch) == 0:
                continue
            imgs, targets = batch
            imgs = [img.to(device) for img in imgs]
            preds = model(imgs)
            all_preds.extend(preds)
            all_targets.extend(targets)
    return all_preds, all_targets


def pred_to_coco_format(all_preds, all_targets):
    results = []
    for idx, pred in enumerate(all_preds):
        image_id = all_targets[idx]["image_id"].item()
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            results.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score)
            })
    return results


def compute_coco_map(val_json_path, results_list):
    """Computes COCO mAP metrics using pycocotools."""
    if not results_list:
        print("[WARN] No predictions generated.")
        return 0.0

    coco_gt = COCO(val_json_path)
    tmp_pred_file = "/tmp/eval_preds_tmp.json"
    with open(tmp_pred_file, "w") as f:
        json.dump(results_list, f)

    coco_dt = coco_gt.loadRes(tmp_pred_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    if os.path.exists(tmp_pred_file):
        os.remove(tmp_pred_file)

    return coco_eval.stats[0]  # AP @ [.5:.95]


def compute_froc(all_preds, all_targets, iou_thresh=0.3, target_fps=(0.5, 1.0, 2.0, 4.0, 8.0)):
    """Computes Free-Response ROC (FROC) sensitivity at target false positive rates per image."""
    all_gts = [t["boxes"].cpu().numpy() for t in all_targets]
    total_gts = sum(len(b) for b in all_gts)
    if total_gts == 0:
        return 0.0, []

    sensitivity_list = []
    fp_per_img_list = []

    for thresh in np.linspace(0.05, 0.95, 20):
        found, false_pos = 0, 0
        for i, pred in enumerate(all_preds):
            keep = pred["scores"].cpu().numpy() >= thresh
            boxes_pred = pred["boxes"].cpu().numpy()[keep]
            gts = all_gts[i]
            detected_gt = set()

            for b_pred in boxes_pred:
                matched = False
                for j, b_gt in enumerate(gts):
                    iou = box_iou(torch.tensor([b_pred]), torch.tensor([b_gt])).item()
                    if iou >= iou_thresh and j not in detected_gt:
                        detected_gt.add(j)
                        found += 1
                        matched = True
                        break
                if not matched:
                    false_pos += 1

        fp_per_img_list.append(false_pos / max(1, len(all_gts)))
        sensitivity_list.append(found / total_gts)

    # Calculate average sensitivity at target FP rates
    sens_at_targets = []
    for fp in target_fps:
        idx = np.searchsorted(fp_per_img_list, fp, side='right')
        if idx >= len(sensitivity_list):
            sens_at_targets.append(sensitivity_list[-1])
        else:
            sens_at_targets.append(sensitivity_list[idx])

    froc_score = float(np.mean(sens_at_targets))
    return froc_score, sens_at_targets


def main():
    parser = argparse.ArgumentParser(description="Evaluate TBX11K Swin Faster R-CNN")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to TBX11K dataset directory")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pth file")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                        help="Split to evaluate on")
    parser.add_argument("--img-size", type=int, default=800,
                        help="Input image resolution")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of foreground classes")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-plot", type=str, default="evaluation_froc.png")
    args = parser.parse_args()

    img_dir = os.path.join(args.data_dir, args.split, "img")
    json_path = os.path.join(args.data_dir, "annotations", f"{args.split}.json")

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_set = TBX11KDetectionDataset(img_dir, json_path, transform=transform)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=detection_collate_fn)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = make_fasterrcnn_swin(num_classes=args.num_classes, img_size=args.img_size)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.to(args.device)
    model.eval()

    print("Running inference...")
    preds, targets = run_inference(model, val_loader, args.device)
    coco_results = pred_to_coco_format(preds, targets)

    print("\n" + "=" * 60)
    print("COCO EVALUATION METRICS")
    print("=" * 60)
    mAP = compute_coco_map(json_path, coco_results)

    print("\n" + "=" * 60)
    print("FROC EVALUATION METRICS")
    print("=" * 60)
    froc_score, sens_points = compute_froc(preds, targets)
    print(f"Overall FROC Score: {froc_score:.4f}")
    print(f"Sensitivity at [0.5, 1.0, 2.0, 4.0, 8.0] FP/Image: {sens_points}")


if __name__ == "__main__":
    main()
