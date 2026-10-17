import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

# Import from train.py
from train import TBX11KDetectionDataset, transform, SwinBackboneWithFPN
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Original script behind the best/worst predictions per category (code unchanged).
# EDIT BEFORE RUNNING: author's ASU Sol paths below. Run from this folder (it imports train.py).
# -------- CONFIG --------
CHECKPOINT_EPOCH = 11  # SET YOUR EPOCH HERE
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
NUM_BEST = 3  # Best/worst per class
NUM_WORST = 3

# -------- PATHS --------
DATA_DIR = "/scratch/smanika3/v2TBX11k-Dataset/tbx11k-simplified"
CKPT_DIR = "/scratch/smanika3/swin-localization-logs/checkpoints"
OUTPUT_DIR = "/scratch/smanika3/swin-localization-logs/min_max_predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_TXT = os.path.join(DATA_DIR, "test.txt")
CSV_PATH = os.path.join(DATA_DIR, "data.csv")

# -------- DATASET --------
test_dataset_vis = TBX11KDetectionDataset(TEST_TXT, CSV_PATH, transforms=None)  # For visualization
test_dataset = TBX11KDetectionDataset(TEST_TXT, CSV_PATH, transforms=transform)  # For inference

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

# -------- F1 SCORE CALCULATION --------
def calculate_f1_score(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, 
                       conf_threshold=0.3, iou_threshold=0.5, target_class=None):
    """
    Calculate F1 score for a single image, optionally filtered by class.
    
    Args:
        target_class: If specified, only evaluate this class (1 or 2). If None, evaluate all.
    
    Returns: F1 score, precision, recall, TP, FP, FN
    """
    # Filter by target class if specified
    if target_class is not None:
        # Filter predictions
        pred_mask = pred_labels == target_class
        pred_boxes = pred_boxes[pred_mask]
        pred_scores = pred_scores[pred_mask]
        pred_labels = pred_labels[pred_mask]
        
        # Filter ground truths
        gt_mask = gt_labels == target_class
        gt_boxes = gt_boxes[gt_mask]
        gt_labels = gt_labels[gt_mask]
    
    # Filter predictions by confidence threshold
    mask = pred_scores >= conf_threshold
    filtered_pred_boxes = pred_boxes[mask]
    filtered_pred_labels = pred_labels[mask]
    
    if len(gt_boxes) == 0 and len(filtered_pred_boxes) == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0  # Perfect score if both empty
    
    if len(gt_boxes) == 0:
        return 0.0, 0.0, 0.0, 0, len(filtered_pred_boxes), 0  # All FP
    
    if len(filtered_pred_boxes) == 0:
        return 0.0, 0.0, 0.0, 0, 0, len(gt_boxes)  # All FN
    
    # Match predictions to ground truths using IoU
    matched_gts = set()
    tp = 0
    fp = 0
    
    for pred_box, pred_label in zip(filtered_pred_boxes, filtered_pred_labels):
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if gt_idx in matched_gts:
                continue
            
            # Class must match
            if int(pred_label) != int(gt_label):
                continue
            
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            tp += 1
            matched_gts.add(best_gt_idx)
        else:
            fp += 1
    
    fn = len(gt_boxes) - len(matched_gts)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1, precision, recall, tp, fp, fn

# -------- IMAGE CATEGORIZATION --------
def categorize_image(gt_labels):
    """Categorize image based on ground truth labels"""
    if len(gt_labels) == 0:
        return 'No_TB'
    elif 1 in gt_labels and 2 not in gt_labels:
        return 'Active_TB'
    elif 2 in gt_labels and 1 not in gt_labels:
        return 'Latent_TB'
    else:
        return 'Mixed_TB'  # Has both classes

# -------- VISUALIZATION --------
def visualize_prediction(image, gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels, 
                        f1, precision, recall, tp, fp, fn, category, save_path, conf_threshold=0.3):
    """Visualize ground truth and predictions with metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Display image
    ax1.imshow(image)
    ax1.set_title(f'Ground Truth - {category}', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(image)
    title = f'Predictions (conf > {conf_threshold}) - {category}\nF1={f1:.3f}, P={precision:.3f}, R={recall:.3f} | TP={tp}, FP={fp}, FN={fn}'
    ax2.set_title(title, fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    class_names = {0: 'Unknown', 1: 'Active TB', 2: 'Latent TB'}
    class_colors = {0: 'gray', 1: 'red', 2: 'blue'}
    
    # Draw ground truth boxes
    for box, label in zip(gt_boxes, gt_labels):
        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=3, edgecolor=class_colors.get(int(label), 'yellow'),
            facecolor='none', linestyle='-'
        )
        ax1.add_patch(rect)
        ax1.text(box[0], box[1] - 5, class_names.get(int(label), 'Unknown'),
                color='white', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=class_colors.get(int(label), 'yellow'), alpha=0.8))
    
    # Draw predictions above threshold
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        if score >= conf_threshold:
            rect = patches.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=3, edgecolor=class_colors.get(int(label), 'yellow'),
                facecolor='none', linestyle='--'
            )
            ax2.add_patch(rect)
            ax2.text(box[0], box[1] - 5, f"{class_names.get(int(label), 'Unknown')} {score:.2f}",
                    color='white', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor=class_colors.get(int(label), 'yellow'), alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# -------- MAIN --------
if __name__ == "__main__":
    import csv
    
    print(f"Loading checkpoint for epoch {CHECKPOINT_EPOCH}...")
    
    # Load model
    ckpt_path = os.path.join(CKPT_DIR, f"checkpoint_epoch_{CHECKPOINT_EPOCH}.pth")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found: {ckpt_path}")
        exit(1)
    
    model = build_model()
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Evaluating all {len(test_dataset)} test images...")
    
    # Store results by category
    categorized_scores = {
        'No_TB': [],
        'Active_TB': [],
        'Latent_TB': [],
        'Mixed_TB': []
    }
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset))):
            # Get sample for inference
            img_tensor, target = test_dataset[idx]
            img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
            
            # Predict
            output = model(img_tensor)[0]
            
            # Extract predictions
            pred_boxes = output['boxes'].cpu().numpy()
            pred_scores = output['scores'].cpu().numpy()
            pred_labels = output['labels'].cpu().numpy()
            
            # Extract ground truth
            gt_boxes = target['boxes'].numpy()
            gt_labels = target['labels'].numpy()
            
            # Categorize image
            category = categorize_image(gt_labels)
            
            # Calculate F1 score (overall for the image)
            f1, precision, recall, tp, fp, fn = calculate_f1_score(
                pred_boxes, pred_scores, pred_labels,
                gt_boxes, gt_labels,
                conf_threshold=CONF_THRESHOLD,
                iou_threshold=IOU_THRESHOLD,
                target_class=None  # Evaluate all classes
            )
            
            categorized_scores[category].append({
                'idx': idx,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'pred_boxes': pred_boxes,
                'pred_scores': pred_scores,
                'pred_labels': pred_labels,
                'gt_boxes': gt_boxes,
                'gt_labels': gt_labels,
                'category': category
            })
    
    print(f"\n{'='*100}")
    print(f"BEST and WORST Predictions by Category (Epoch {CHECKPOINT_EPOCH})")
    print(f"{'='*100}")
    print(f"Metric: F1 Score @ IoU={IOU_THRESHOLD}, Confidence={CONF_THRESHOLD}")
    print(f"{'='*100}\n")
    
    # Print category statistics
    for cat in ['No_TB', 'Active_TB', 'Latent_TB', 'Mixed_TB']:
        print(f"{cat}: {len(categorized_scores[cat])} images")
    print()
    
    all_results = []
    
    # Process each category
    for category in ['No_TB', 'Active_TB', 'Latent_TB', 'Mixed_TB']:
        images = categorized_scores[category]
        
        if len(images) == 0:
            print(f"No images in category: {category}")
            continue
        
        # Sort by F1 score
        images.sort(key=lambda x: x['f1'])
        
        num_worst = min(NUM_WORST, len(images))
        num_best = min(NUM_BEST, len(images))
        
        worst_images = images[:num_worst]
        best_images = images[-num_best:][::-1]  # Reverse to get highest first
        
        # Visualize worst
        print(f"\n{'='*100}")
        print(f"WORST {num_worst} Predictions - {category}")
        print("-" * 100)
        for rank, img_data in enumerate(worst_images, 1):
            idx = img_data['idx']
            f1 = img_data['f1']
            p = img_data['precision']
            r = img_data['recall']
            print(f"Rank {rank}: Image {idx} | F1={f1:.4f}, P={p:.4f}, R={r:.4f} | TP={img_data['tp']}, FP={img_data['fp']}, FN={img_data['fn']}")
            
            # Get original image
            img_pil, _ = test_dataset_vis[idx]
            save_path = os.path.join(OUTPUT_DIR, f"worst_{category}_{rank}_img{idx}_f1{f1:.3f}_e{CHECKPOINT_EPOCH}.png")
            
            visualize_prediction(
                np.array(img_pil),
                img_data['gt_boxes'], img_data['gt_labels'],
                img_data['pred_boxes'], img_data['pred_scores'], img_data['pred_labels'],
                f1, p, r, img_data['tp'], img_data['fp'], img_data['fn'],
                category, save_path, conf_threshold=CONF_THRESHOLD
            )
            
            all_results.append(['WORST', category, rank, idx, f1, p, r, img_data['tp'], img_data['fp'], img_data['fn']])
        
        # Visualize best
        print(f"\nBEST {num_best} Predictions - {category}")
        print("-" * 100)
        for rank, img_data in enumerate(best_images, 1):
            idx = img_data['idx']
            f1 = img_data['f1']
            p = img_data['precision']
            r = img_data['recall']
            print(f"Rank {rank}: Image {idx} | F1={f1:.4f}, P={p:.4f}, R={r:.4f} | TP={img_data['tp']}, FP={img_data['fp']}, FN={img_data['fn']}")
            
            # Get original image
            img_pil, _ = test_dataset_vis[idx]
            save_path = os.path.join(OUTPUT_DIR, f"best_{category}_{rank}_img{idx}_f1{f1:.3f}_e{CHECKPOINT_EPOCH}.png")
            
            visualize_prediction(
                np.array(img_pil),
                img_data['gt_boxes'], img_data['gt_labels'],
                img_data['pred_boxes'], img_data['pred_scores'], img_data['pred_labels'],
                f1, p, r, img_data['tp'], img_data['fp'], img_data['fn'],
                category, save_path, conf_threshold=CONF_THRESHOLD
            )
            
            all_results.append(['BEST', category, rank, idx, f1, p, r, img_data['tp'], img_data['fp'], img_data['fn']])
    
    # Save summary CSV
    summary_csv = os.path.join(OUTPUT_DIR, f"min_max_summary_by_category_epoch{CHECKPOINT_EPOCH}.csv")
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Type', 'Category', 'Rank', 'Image_Index', 'F1_Score', 'Precision', 'Recall', 'TP', 'FP', 'FN'])
        for row in all_results:
            writer.writerow([row[0], row[1], row[2], row[3], 
                           f"{row[4]:.6f}", f"{row[5]:.6f}", f"{row[6]:.6f}",
                           row[7], row[8], row[9]])
    
    print(f"\n{'='*100}")
    print(f"Summary CSV saved to: {summary_csv}")
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print(f"{'='*100}")
    print("\nDone!")