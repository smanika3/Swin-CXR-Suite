import os
import torch
import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import ast
import timm
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from collections import OrderedDict
import glob

# Original training script of the reported run (code unchanged). EDIT BEFORE RUNNING: author's ASU Sol paths.
# DATA_DIR must hold data.csv, images/ and the train.txt / val.txt from the split (see ../data/split_dataset.py).
# -------- CONFIG & PATHS --------
DATA_DIR = "/scratch/smanika3/v2TBX11k-Dataset/tbx11k-simplified"
TRAIN_TXT = os.path.join(DATA_DIR, "train.txt")
VAL_TXT = os.path.join(DATA_DIR, "val.txt")
CSV_PATH = os.path.join(DATA_DIR, "data.csv")
IMG_DIR = os.path.join(DATA_DIR, "images")
CHECKPOINT_DIR = "/scratch/smanika3/swin-localization-logs/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 30

# -------- DATASET --------
class TBX11KDetectionDataset(Dataset):
    def __init__(self, txt_file, csv_file, transforms=None):
        with open(txt_file) as f:
            self.image_files = [line.strip().split('/')[-1] for line in f]
        self.img_dir = os.path.dirname(txt_file) + "/images"
        self.csv_df = pd.read_csv(csv_file)
        self.transforms = transforms

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        img_path = os.path.join(self.img_dir, fname)
        image = Image.open(img_path).convert("RGB")
        ann_rows = self.csv_df[self.csv_df['fname'] == fname]
        boxes, labels = [], []
        for _, row in ann_rows.iterrows():
            if row['bbox'] != 'none':
                bbox = ast.literal_eval(row['bbox'])
                boxes.append([
                    bbox['xmin'],
                    bbox['ymin'],
                    bbox['xmin'] + bbox['width'],
                    bbox['ymin'] + bbox['height']
                ])
                if row['tb_type'] == 'active_tb':
                    labels.append(1)
                elif row['tb_type'] == 'latent_tb':
                    labels.append(2)
                else:
                    labels.append(0)
        target = {}
        target['boxes'] = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4), dtype=torch.float32)
        target['labels'] = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        target['image_id'] = torch.tensor([idx])
        if self.transforms:
            image = self.transforms(image)
        return image, target

    def __len__(self):
        return len(self.image_files)

# -------- TRANSFORMS --------
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# -------- SWIN BACKBONE --------
# -------- SWIN BACKBONE WITH FPN --------
class SwinBackboneWithFPN(torch.nn.Module):
    def __init__(self, model_name="swin_base_patch4_window7_224", pretrained=True):
        super().__init__()
        self.body = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            features_only=True, 
            out_indices=[1, 2, 3],  # Use middle layers for FPN
            img_size=512
        )
        
        # Get channel dimensions from Swin output
        # Swin base: [256, 512, 1024] for indices [1,2,3]
        feature_info = self.body.feature_info.get_dicts()
        in_channels_list = [f['num_chs'] for f in feature_info]
        
        # Build FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=256,
            extra_blocks=LastLevelMaxPool()
        )
        self.out_channels = 256

    def forward(self, x):
        # Get features from Swin backbone
        features = self.body(x)
        
        # Swin outputs NHWC format, convert to NCHW for FPN
        # Before: [batch, height, width, channels]
        # After:  [batch, channels, height, width]
        features = [feat.permute(0, 3, 1, 2).contiguous() for feat in features]
        
        # Convert list to OrderedDict for FPN
        # Use string numbers as keys for compatibility with FasterRCNN
        features_dict = OrderedDict()
        for i, feat in enumerate(features):
            features_dict[str(i)] = feat
        
        # Pass through FPN
        fpn_features = self.fpn(features_dict)
        return fpn_features

# -------- RESUME CHECKPOINT --------
def get_latest_checkpoint(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch_*.pth"))
    if not ckpts:
        return None
    def extract_epoch(path):
        return int(os.path.basename(path).split("_")[-1].split(".")[0])
    return max(ckpts, key=extract_epoch)

if __name__ == "__main__":
    # -------- MODEL SETUP --------
    backbone = SwinBackboneWithFPN()

    # FPN produces 4 feature levels, need anchors for each
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),  # 4 levels
        aspect_ratios=((0.5, 1.0, 2.0),) * 4    # Same ratios for all 4 levels
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=3,
        rpn_anchor_generator=anchor_generator,
        box_score_thresh=0.05,
        min_size=512,
        max_size=512
    )
    model.to(DEVICE)

    # -------- LOADERS --------
    train_dataset = TBX11KDetectionDataset(TRAIN_TXT, CSV_PATH, transforms=transform)
    val_dataset = TBX11KDetectionDataset(VAL_TXT, CSV_PATH, transforms=transform)
    def collate_fn(batch): return tuple(zip(*batch))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=20, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


    latest_ckpt = get_latest_checkpoint(CHECKPOINT_DIR)
    if latest_ckpt:
        print(f"[INFO] Resuming from checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        print("[INFO] No checkpoint found, starting from scratch")
        start_epoch = 1

    # -------- TRAINING LOOP --------
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0
        for images, targets in train_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            train_loss += losses.item()
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch}: Train loss {avg_train_loss:.4f}")

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                
                # During eval(), model returns predictions not losses
                # Temporarily switch to train mode to get validation loss
                model.train()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                model.eval()
                
                val_loss += losses.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch}: Val loss {avg_val_loss:.4f}")

        # -------- SAVE CHECKPOINT --------
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }, ckpt_path)
        print(f"[INFO] Saved checkpoint: {ckpt_path} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("Training complete!")
