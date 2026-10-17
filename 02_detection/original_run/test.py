import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
from collections import defaultdict

# Import from train.py
from train import TBX11KDetectionDataset, transform, SwinBackboneWithFPN
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Original test script of the reported run (code unchanged): scores every checkpoint on test.txt and writes
# evaluation_results.csv, the AP50/AP75/FROC-vs-epoch plots and one FROC curve per epoch.
# EDIT BEFORE RUNNING: author's ASU Sol paths. Run from this folder (it imports train.py).
# -------- PATHS --------
DATA_DIR = "/scratch/smanika3/v2TBX11k-Dataset/tbx11k-simplified"
CKPT_DIR = "/scratch/smanika3/swin-localization-logs/checkpoints"
OUTPUT_DIR = "/scratch/smanika3/swin-localization-logs/test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_TXT = os.path.join(DATA_DIR, "test.txt")
CSV_PATH = os.path.join(DATA_DIR, "data.csv")

# -------- DATASET --------
test_dataset = TBX11KDetectionDataset(TEST_TXT, CSV_PATH, transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, num_workers=16, shuffle=False,
    collate_fn=lambda x: tuple(zip(*x))
)

# -------- MODEL BUILDER --------
def build_model():
    backbone = SwinBackboneWithFPN()
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )
    model = FasterRCNN(
        backbone=backbone,
        num_classes=3,
        rpn_anchor_generator=anchor_generator,
        box_score_thresh=0.05,
        min_size=512,
        max_size=512
    ).to(DEVICE)
    return model

# -------- IoU CALCULATION --------
def compute_iou(box1, box2):
    """Compute IoU between two boxes [xmin, ymin, xmax, ymax]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

# -------- mAP CALCULATION --------
def calculate_ap(all_predictions, all_groundtruths, iou_threshold=0.5, class_filter=None, agnostic_tb=False):
    """
    Calculate Average Precision (AP) at given IoU threshold.
    
    Args:
        all_predictions: List of predictions per image
        all_groundtruths: List of ground truths per image
        iou_threshold: IoU threshold for TP
        class_filter: If specified, only evaluate this class (1 or 2)
        agnostic_tb: If True, treat both class 1 and 2 as "TB positive" (class-agnostic)
    
    Returns:
        AP score
    """
    # Flatten all predictions and ground truths with image indices
    all_dets = []
    all_gts = []
    
    for img_idx, (preds, gts) in enumerate(zip(all_predictions, all_groundtruths)):
        for pred in preds:
            # pred: [xmin, ymin, xmax, ymax, score, class]
            pred_class = int(pred[5])
            
            # Apply class filter
            if class_filter is not None and pred_class != class_filter:
                continue
            
            # For agnostic TB, convert class 1 and 2 to unified "TB"
            if agnostic_tb and pred_class in [1, 2]:
                pred_class = 1  # Treat all TB as class 1
            
            all_dets.append({
                'img_idx': img_idx,
                'box': pred[:4],
                'score': pred[4],
                'class': pred_class
            })
        
        for gt in gts:
            # gt: [xmin, ymin, xmax, ymax, class]
            gt_class = int(gt[4])
            
            # Apply class filter
            if class_filter is not None and gt_class != class_filter:
                continue
            
            # For agnostic TB, convert class 1 and 2 to unified "TB"
            if agnostic_tb and gt_class in [1, 2]:
                gt_class = 1  # Treat all TB as class 1
            
            all_gts.append({
                'img_idx': img_idx,
                'box': gt[:4],
                'class': gt_class,
                'matched': False
            })
    
    if len(all_dets) == 0 or len(all_gts) == 0:
        return 0.0
    
    # Sort detections by confidence
    all_dets.sort(key=lambda x: x['score'], reverse=True)
    
    # Calculate TP/FP for each detection
    tp = np.zeros(len(all_dets))
    fp = np.zeros(len(all_dets))
    
    for det_idx, det in enumerate(all_dets):
        # Find GTs in same image and same class
        img_gts = [gt for gt in all_gts 
                   if gt['img_idx'] == det['img_idx'] and gt['class'] == det['class']]
        
        max_iou = 0
        max_gt_idx = -1
        
        for gt_idx, gt in enumerate(img_gts):
            iou = compute_iou(det['box'], gt['box'])
            if iou > max_iou:
                max_iou = iou
                max_gt_idx = gt_idx
        
        if max_iou >= iou_threshold and max_gt_idx >= 0 and not img_gts[max_gt_idx]['matched']:
            tp[det_idx] = 1
            img_gts[max_gt_idx]['matched'] = True
        else:
            fp[det_idx] = 1
    
    # Compute precision/recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    total_gts = len(all_gts)
    
    recalls = tp_cumsum / total_gts if total_gts > 0 else np.zeros_like(tp_cumsum)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    
    # Compute AP using 11-point interpolation
    ap = 0
    for t in np.linspace(0, 1, 11):
        p = precisions[recalls >= t]
        ap += (p.max() if len(p) > 0 else 0) / 11
    
    return ap

# -------- FROC CALCULATION --------
def calculate_froc(all_predictions, all_groundtruths, iou_threshold=0.5, max_fppi=2.0):
    """Calculate FROC curve (sensitivity vs FPs per image)"""
    thresholds = np.linspace(0.05, 0.95, 50)
    sensitivities = []
    fps_per_img = []
    n_images = len(all_groundtruths)
    
    for thresh in thresholds:
        tp = 0
        fp = 0
        fn = 0
        
        for preds, gts in zip(all_predictions, all_groundtruths):
            # Filter by threshold
            filtered_preds = preds[preds[:, 4] >= thresh] if len(preds) > 0 else np.zeros((0, 6))
            
            matched_gts = set()
            matched_preds = set()
            
            # Match predictions to GTs
            for pred_idx, pred in enumerate(filtered_preds):
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gts):
                    if gt_idx in matched_gts:
                        continue
                    # Check class match
                    if int(pred[5]) != int(gt[4]):
                        continue
                    iou = compute_iou(pred[:4], gt[:4])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold:
                    tp += 1
                    matched_gts.add(best_gt_idx)
                    matched_preds.add(pred_idx)
                else:
                    fp += 1
            
            # Unmatched GTs are false negatives
            fn += len(gts) - len(matched_gts)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        fppi = fp / n_images
        
        if fppi <= max_fppi:  # Only keep points within max_fppi
            sensitivities.append(sensitivity)
            fps_per_img.append(fppi)
    
    # Calculate FROC score (average sensitivity at FPPI <= 1.0)
    froc_score = np.mean([s for f, s in zip(fps_per_img, sensitivities) if f <= 1.0]) if any(f <= 1.0 for f in fps_per_img) else 0
    
    return fps_per_img, sensitivities, froc_score

# -------- EVALUATION --------
def evaluate_checkpoint(ckpt_path):
    """Evaluate a single checkpoint"""
    print(f"\nEvaluating: {os.path.basename(ckpt_path)}")
    
    # Load model
    model = build_model()
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_predictions = []
    all_groundtruths = []
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Inference"):
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)
            
            # Process output (batch_size=1)
            out = outputs[0]
            boxes = out['boxes'].cpu().numpy()
            scores = out['scores'].cpu().numpy()
            labels = out['labels'].cpu().numpy()
            
            # Combine: [xmin, ymin, xmax, ymax, score, class]
            preds = np.column_stack([boxes, scores, labels]) if len(boxes) > 0 else np.zeros((0, 6))
            
            # Ground truth: [xmin, ymin, xmax, ymax, class]
            gt_boxes = targets[0]['boxes'].cpu().numpy()
            gt_labels = targets[0]['labels'].cpu().numpy()
            gts = np.column_stack([gt_boxes, gt_labels]) if len(gt_boxes) > 0 else np.zeros((0, 5))
            
            all_predictions.append(preds)
            all_groundtruths.append(gts)
    
    # Calculate AP metrics at different IoU thresholds
    # AP50 (IoU=0.5)
    ap50_agnostic = calculate_ap(all_predictions, all_groundtruths, iou_threshold=0.5, agnostic_tb=True)
    ap50_class1 = calculate_ap(all_predictions, all_groundtruths, iou_threshold=0.5, class_filter=1)
    ap50_class2 = calculate_ap(all_predictions, all_groundtruths, iou_threshold=0.5, class_filter=2)
    
    # AP75 (IoU=0.75)
    ap75_agnostic = calculate_ap(all_predictions, all_groundtruths, iou_threshold=0.75, agnostic_tb=True)
    ap75_class1 = calculate_ap(all_predictions, all_groundtruths, iou_threshold=0.75, class_filter=1)
    ap75_class2 = calculate_ap(all_predictions, all_groundtruths, iou_threshold=0.75, class_filter=2)
    
    # FROC
    fppi, sensitivity, froc_score = calculate_froc(all_predictions, all_groundtruths, iou_threshold=0.5, max_fppi=2.0)
    
    epoch = int(os.path.basename(ckpt_path).split("_")[-1].split(".")[0])
    
    return {
        'epoch': epoch,
        'ap50_agnostic': ap50_agnostic,
        'ap50_class1': ap50_class1,
        'ap50_class2': ap50_class2,
        'ap75_agnostic': ap75_agnostic,
        'ap75_class1': ap75_class1,
        'ap75_class2': ap75_class2,
        'froc_score': froc_score,
        'fppi': fppi,
        'sensitivity': sensitivity,
        'all_predictions': all_predictions,
        'all_groundtruths': all_groundtruths
    }

# -------- MAIN --------
if __name__ == "__main__":
    # Create plots directory
    PLOTS_DIR = os.path.join("/scratch/smanika3/swin-localization-logs", "plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Find all checkpoints
    ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, "checkpoint_epoch_*.pth")),
                       key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]))
    
    if not ckpt_files:
        print(f"No checkpoints found in {CKPT_DIR}")
        exit(1)
    
    print(f"Found {len(ckpt_files)} checkpoints")
    
    # Evaluate all checkpoints
    results = []
    for ckpt_path in ckpt_files:
        result = evaluate_checkpoint(ckpt_path)
        results.append(result)
        print(f"Epoch {result['epoch']}: AP50_TB={result['ap50_agnostic']:.4f}, AP75_TB={result['ap75_agnostic']:.4f}, FROC={result['froc_score']:.4f}")
        
        # Save individual FROC curve for this epoch
        plt.figure(figsize=(10, 8))
        plt.plot(result['fppi'], result['sensitivity'], linewidth=3, marker='o', markersize=6, color='blue')
        plt.xlabel('False Positives Per Image (FPPI)', fontsize=14)
        plt.ylabel('Sensitivity (TPR)', fontsize=14)
        plt.title(f'FROC Curve - Epoch {result["epoch"]}\nFROC Score (Avg Sens @ FPPI≤1.0): {result["froc_score"]:.4f}', fontsize=16)
        plt.xlim([0, 2.0])
        plt.ylim([0, 1.0])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        froc_path = os.path.join(PLOTS_DIR, f'froc_epoch_{result["epoch"]}.png')
        plt.savefig(froc_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Sort by epoch
    results.sort(key=lambda x: x['epoch'])
    
    # Extract metrics
    epochs = [r['epoch'] for r in results]
    ap50_agnostic = [r['ap50_agnostic'] for r in results]
    ap50_class1 = [r['ap50_class1'] for r in results]
    ap50_class2 = [r['ap50_class2'] for r in results]
    ap75_agnostic = [r['ap75_agnostic'] for r in results]
    ap75_class1 = [r['ap75_class1'] for r in results]
    ap75_class2 = [r['ap75_class2'] for r in results]
    froc_scores = [r['froc_score'] for r in results]
    
    # Plot AP50 metrics vs Epoch
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, ap50_agnostic, marker='o', linewidth=2, markersize=8, label='TB Agnostic (Class 1+2)')
    plt.plot(epochs, ap50_class1, marker='s', linewidth=2, markersize=8, label='Active TB (Class 1)')
    plt.plot(epochs, ap50_class2, marker='^', linewidth=2, markersize=8, label='Latent TB (Class 2)')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('AP @ IoU=0.5', fontsize=14)
    plt.title('AP50 vs Epoch (Per-Class and Category-Agnostic)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'ap50_vs_epoch.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/ap50_vs_epoch.png")
    
    # Plot AP75 metrics vs Epoch
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, ap75_agnostic, marker='o', linewidth=2, markersize=8, label='TB Agnostic (Class 1+2)')
    plt.plot(epochs, ap75_class1, marker='s', linewidth=2, markersize=8, label='Active TB (Class 1)')
    plt.plot(epochs, ap75_class2, marker='^', linewidth=2, markersize=8, label='Latent TB (Class 2)')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('AP @ IoU=0.75', fontsize=14)
    plt.title('AP75 vs Epoch (Per-Class and Category-Agnostic)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'ap75_vs_epoch.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/ap75_vs_epoch.png")
    
    # Plot FROC Score vs Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, froc_scores, marker='s', linewidth=2, markersize=8, color='green')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('FROC Score (Avg Sensitivity @ FPPI≤1.0)', fontsize=14)
    plt.title('FROC Score vs Epoch', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'froc_vs_epoch.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/froc_vs_epoch.png")
    
    # Plot FROC curves for all epochs
    plt.figure(figsize=(12, 8))
    for r in results:
        plt.plot(r['fppi'], r['sensitivity'], label=f"Epoch {r['epoch']}", alpha=0.7)
    plt.xlabel('False Positives Per Image (FPPI)', fontsize=14)
    plt.ylabel('Sensitivity (TPR)', fontsize=14)
    plt.title('FROC Curves - All Epochs', fontsize=16)
    plt.xlim([0, 2.0])
    plt.ylim([0, 1.0])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'froc_curves_all.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/froc_curves_all.png")
    
    # Find best epochs
    best_ap50_agnostic_idx = np.argmax(ap50_agnostic)
    best_ap75_agnostic_idx = np.argmax(ap75_agnostic)
    best_froc_idx = np.argmax(froc_scores)
    
    # Print summary table
    print("\n" + "="*120)
    print("EVALUATION SUMMARY - PER-CLASS AND CATEGORY-AGNOSTIC AP")
    print("="*120)
    print(f"{'Epoch':<8} {'AP50_TB':<10} {'AP50_Act':<10} {'AP50_Lat':<10} {'AP75_TB':<10} {'AP75_Act':<10} {'AP75_Lat':<10} {'FROC':<10}")
    print("-"*120)
    for e, a50_tb, a50_c1, a50_c2, a75_tb, a75_c1, a75_c2, fs in zip(
        epochs, ap50_agnostic, ap50_class1, ap50_class2, 
        ap75_agnostic, ap75_class1, ap75_class2, froc_scores
    ):
        marker = ""
        if e == epochs[best_ap50_agnostic_idx]:
            marker += " ★AP50"
        if e == epochs[best_ap75_agnostic_idx]:
            marker += " ★AP75"
        if e == epochs[best_froc_idx]:
            marker += " ★FROC"
        
        print(f"{e:<8} {a50_tb:<10.6f} {a50_c1:<10.6f} {a50_c2:<10.6f} {a75_tb:<10.6f} {a75_c1:<10.6f} {a75_c2:<10.6f} {fs:<10.6f} {marker}")
    print("="*120)
    print(f"Best AP50 (TB Agnostic): {ap50_agnostic[best_ap50_agnostic_idx]:.6f} at Epoch {epochs[best_ap50_agnostic_idx]}")
    print(f"Best AP75 (TB Agnostic): {ap75_agnostic[best_ap75_agnostic_idx]:.6f} at Epoch {epochs[best_ap75_agnostic_idx]}")
    print(f"Best FROC Score: {froc_scores[best_froc_idx]:.6f} at Epoch {epochs[best_froc_idx]}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Individual FROC curves saved to: {PLOTS_DIR}")
    print("="*120)
    
    # Save results to CSV
    import csv
    csv_path = os.path.join(OUTPUT_DIR, 'evaluation_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'AP50_TB_Agnostic', 'AP50_ActiveTB', 'AP50_LatentTB', 
                        'AP75_TB_Agnostic', 'AP75_ActiveTB', 'AP75_LatentTB', 'FROC_Score',
                        'Best_AP50', 'Best_AP75', 'Best_FROC'])
        for e, a50_tb, a50_c1, a50_c2, a75_tb, a75_c1, a75_c2, fs in zip(
            epochs, ap50_agnostic, ap50_class1, ap50_class2,
            ap75_agnostic, ap75_class1, ap75_class2, froc_scores
        ):
            is_best_ap50 = '✓' if e == epochs[best_ap50_agnostic_idx] else ''
            is_best_ap75 = '✓' if e == epochs[best_ap75_agnostic_idx] else ''
            is_best_froc = '✓' if e == epochs[best_froc_idx] else ''
            writer.writerow([e, f'{a50_tb:.6f}', f'{a50_c1:.6f}', f'{a50_c2:.6f}',
                           f'{a75_tb:.6f}', f'{a75_c1:.6f}', f'{a75_c2:.6f}', f'{fs:.6f}',
                           is_best_ap50, is_best_ap75, is_best_froc])
    
    print(f"\nCSV results saved to: {csv_path}")
    print("\nTest evaluation complete!")