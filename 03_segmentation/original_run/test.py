import os
import sys
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# === SWIN MODEL IMPORT ===
# Original test script of the reported run (code unchanged): scores every checkpoint on test_files.txt and writes
# the Dice-vs-epoch / loss plots and sample predictions. EDIT BEFORE RUNNING: author's ASU Sol paths; the
# sys.path line must point at a folder containing network/swin_unet_v2.py (copy ../models/ into network/).
sys.path.append('/home/smanika3/ondemand/data/sys/myjobs/projects/default/6/mmdet/Pytorch-Swin-Unet-V2')
from network.swin_unet_v2 import SwinTransformerSys

# === CONFIG ===
CHECKPOINT_DIR = "/scratch/smanika3/checkpoints-swin-segmentation"
BATCH_SIZE = 96
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_IMAGE_DIR = "/scratch/smanika3/chexpert/full_uncompressed/train/"
TEST_MASK_DIR = "/scratch/smanika3/chexmask-masks/train/"
TEST_LIST = "/scratch/smanika3/chexmask-masks/test_files.txt"

NUM_CLASSES = 4  # [background, right lung, left lung, heart]
IMG_SIZE = 224

print(f"[INFO] Using device: {DEVICE}")
print(f"[INFO] Test set: {TEST_LIST}")

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

class CheXmaskSegDataset(torch.utils.data.Dataset):
    def __init__(self, img_base, mask_base, file_list, transform=None):
        self.img_base = img_base
        self.mask_base = mask_base
        with open(file_list) as f:
            self.images = [line.strip() for line in f]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_base, self.images[idx])
        mask_path = os.path.join(self.mask_base, self.images[idx].replace('.jpg', '.png'))
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        # Store original image for visualization
        orig_img = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
        
        if self.transform:
            image = self.transform(image)
        
        mask = np.array(mask, dtype=np.uint8)
        mask = Image.fromarray(mask)
        mask = mask.resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)
        mask = np.array(mask, dtype=np.int64)
        mask = torch.from_numpy(mask)
        
        return image, mask, orig_img, self.images[idx]

test_dataset = CheXmaskSegDataset(TEST_IMAGE_DIR, TEST_MASK_DIR, TEST_LIST, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
print(f"[INFO] Test dataset size: {len(test_dataset)}")

def dice_score(pred, target, num_classes=4, eps=1e-6):
    pred = torch.argmax(pred, dim=1)
    dice = []
    for k in range(num_classes):
        pred_k = (pred == k).float()
        target_k = (target == k).float()
        intersect = (pred_k * target_k).sum()
        denom = pred_k.sum() + target_k.sum()
        dice.append((2 * intersect / (denom + eps)).item())
    return dice

def extract_epoch(path):
    return int(os.path.basename(path).split('_')[-1].split('.')[0])

all_ckpts = sorted(
    [os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint_epoch_")],
    key=extract_epoch
)
print(f"[INFO] Found {len(all_ckpts)} checkpoints to evaluate")

# Storage for metrics
dice_traces = [[] for _ in range(NUM_CLASSES)]
train_loss_traces = []
epoch_traces = []
mean_dice_per_epoch = []

# === STEP 1: EVALUATE ALL CHECKPOINTS ===
print("\n[STEP 1] Evaluating all checkpoints...")
for idx_ckpt, ckpt_path in enumerate(all_ckpts):
    epoch = extract_epoch(ckpt_path)
    print(f"\n[EPOCH {epoch}] Evaluating ({idx_ckpt+1}/{len(all_ckpts)})")
    
    model = SwinTransformerSys(img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    # Store train loss from checkpoint
    train_loss_traces.append(checkpoint.get('loss', np.nan))
    
    dice_accum = np.zeros(NUM_CLASSES)
    n_samples = 0
    
    with torch.no_grad():
        for images, masks, _, _ in tqdm(test_loader, desc=f"Epoch {epoch}"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            dice = dice_score(outputs, masks, NUM_CLASSES)
            dice_accum += np.array(dice) * images.size(0)
            n_samples += images.size(0)
    
    avg_dice = dice_accum / n_samples
    mean_dice = np.mean(avg_dice)
    mean_dice_per_epoch.append(mean_dice)
    
    for k in range(NUM_CLASSES):
        dice_traces[k].append(avg_dice[k])
    
    print(f"[RESULT] Class DICE: {[f'{d:.4f}' for d in avg_dice]}")
    print(f"[RESULT] Mean DICE: {mean_dice:.4f}")
    epoch_traces.append(epoch)
    
    del model
    torch.cuda.empty_cache()

# === STEP 2: FIND BEST CHECKPOINT ===
best_idx = np.argmax(mean_dice_per_epoch)
best_epoch = epoch_traces[best_idx]
best_mean_dice = mean_dice_per_epoch[best_idx]
best_ckpt_path = all_ckpts[best_idx]

print(f"\n[STEP 2] Best checkpoint identified:")
print(f"  Epoch: {best_epoch}")
print(f"  Mean DICE: {best_mean_dice:.4f}")
print(f"  Path: {best_ckpt_path}")

# === STEP 3: GENERATE PLOTS ===
print("\n[STEP 3] Generating plots...")

class_names = ['Background', 'Right Lung', 'Left Lung', 'Heart']

# 3.1 DICE vs Epoch
plt.figure(figsize=(10, 6))
for k in range(NUM_CLASSES):
    plt.plot(epoch_traces, dice_traces[k], marker='o', label=class_names[k])
plt.plot(epoch_traces, mean_dice_per_epoch, marker='X', linestyle='--', linewidth=2, label="Mean DICE", color='black')
plt.axvline(x=best_epoch, color='red', linestyle=':', linewidth=2, label=f'Best Epoch ({best_epoch})')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("DICE Score", fontsize=12)
plt.title("DICE Score vs Epoch (Test Set)", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("dice_vs_epoch.png", dpi=150)
print("[INFO] Saved: dice_vs_epoch.png")

# 3.2 Train Loss vs Epoch
plt.figure(figsize=(10, 6))
plt.plot(epoch_traces, train_loss_traces, marker='o', label="Train Loss", color='blue')
plt.axvline(x=best_epoch, color='red', linestyle=':', linewidth=2, label=f'Best Epoch ({best_epoch})')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Training Loss vs Epoch", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_vs_epoch.png", dpi=150)
print("[INFO] Saved: loss_vs_epoch.png")

# === STEP 4: GENERATE SAMPLE PREDICTIONS FROM BEST MODEL ===
print(f"\n[STEP 4] Generating sample predictions from best model (Epoch {best_epoch})...")

model = SwinTransformerSys(img_size=IMG_SIZE, num_classes=NUM_CLASSES)
checkpoint = torch.load(best_ckpt_path, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

sample_originals = []
sample_masks_gt = []
sample_masks_pred = []

with torch.no_grad():
    for images, masks, orig_imgs, _ in test_loader:
        if len(sample_originals) >= 6:
            break
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
        for i in range(min(6 - len(sample_originals), images.size(0))):
            sample_originals.append(orig_imgs[i])
            sample_masks_gt.append(masks[i].cpu().numpy())
            sample_masks_pred.append(preds[i])

# Color map for masks
def colorize_mask(mask):
    """Convert class mask to RGBA with colors"""
    color_map = {
        0: [0, 0, 0, 0],        # Background - transparent
        1: [255, 0, 0, 128],    # Right lung - red
        2: [0, 255, 0, 128],    # Left lung - green
        3: [0, 0, 255, 128],    # Heart - blue
    }
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for cls_id, color in color_map.items():
        rgba[mask == cls_id] = color
    return rgba

# Generate visualization
fig, axes = plt.subplots(len(sample_originals), 3, figsize=(12, 4 * len(sample_originals)))
if len(sample_originals) == 1:
    axes = axes.reshape(1, -1)

for i in range(len(sample_originals)):
    # Original image
    axes[i, 0].imshow(sample_originals[i])
    axes[i, 0].set_title("Original Image", fontsize=10)
    axes[i, 0].axis('off')
    
    # Ground truth mask overlay
    gt_colored = colorize_mask(sample_masks_gt[i])
    axes[i, 1].imshow(sample_originals[i])
    axes[i, 1].imshow(gt_colored, alpha=0.5)
    axes[i, 1].set_title("Ground Truth", fontsize=10)
    axes[i, 1].axis('off')
    
    # Predicted mask overlay
    pred_colored = colorize_mask(sample_masks_pred[i])
    axes[i, 2].imshow(sample_originals[i])
    axes[i, 2].imshow(pred_colored, alpha=0.5)
    axes[i, 2].set_title(f"Predicted (Epoch {best_epoch})", fontsize=10)
    axes[i, 2].axis('off')

# Add legend
legend_elements = [
    mpatches.Patch(color=(1, 0, 0, 0.5), label='Right Lung'),
    mpatches.Patch(color=(0, 1, 0, 0.5), label='Left Lung'),
    mpatches.Patch(color=(0, 0, 1, 0.5), label='Heart')
]
fig.suptitle(f"Sample Predictions from Best Model (Epoch {best_epoch}, Mean DICE: {best_mean_dice:.4f})", 
             fontsize=14, y=1.0)
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=3)

plt.tight_layout()
plt.savefig("sample_predictions_best_model.png", dpi=150, bbox_inches='tight')
print("[INFO] Saved: sample_predictions_best_model.png")

print("\n[SUCCESS] All evaluations and visualizations complete!")
print(f"[SUMMARY] Best model: Epoch {best_epoch} with mean DICE {best_mean_dice:.4f}")
