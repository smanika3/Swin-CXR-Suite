"""
dataset.py
Dataset loader for TBX11K using simplified tabular CSV annotations and split text files.
"""

import os
import ast
import warnings
from typing import Optional, Callable
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TBX11KDetectionDataset(Dataset):
    """
    TBX11K Dataset supporting dictionary-encoded bounding boxes from a CSV table.

    Args:
        txt_file: Path to train.txt / val.txt / test.txt containing image file paths.
        csv_file: Path to data.csv containing image annotations.
        img_dir: Directory containing images. If None, derived from txt_file directory.
        transform: Torchvision image transforms.
    """
    def __init__(
        self,
        txt_file: str,
        csv_file: str,
        img_dir: Optional[str] = None,
        transform: Optional[Callable] = None
    ):
        with open(txt_file, "r") as f:
            self.image_files = [line.strip().split('/')[-1] for line in f if line.strip()]

        if img_dir is not None:
            self.img_dir = img_dir
        else:
            self.img_dir = os.path.join(os.path.dirname(txt_file), "images")

        self.csv_df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        fname = self.image_files[idx]
        img_path = os.path.join(self.img_dir, fname)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        ann_rows = self.csv_df[self.csv_df['fname'] == fname]

        boxes, labels = [], []
        for _, row in ann_rows.iterrows():
            if row['bbox'] != 'none':
                try:
                    bbox = ast.literal_eval(row['bbox'])
                    box = [
                        bbox['xmin'],
                        bbox['ymin'],
                        bbox['xmin'] + bbox['width'],
                        bbox['ymin'] + bbox['height']
                    ]
                except Exception as e:
                    warnings.warn(f"{fname}: could not parse bbox {row['bbox']!r} ({e}); skipping this box")
                    continue
                boxes.append(box)
                if row['tb_type'] == 'active_tb':
                    labels.append(1)  # Class 1: Active TB
                elif row['tb_type'] == 'latent_tb':
                    labels.append(2)  # Class 2: Latent TB
                else:
                    labels.append(0)

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            'image_id': torch.tensor([idx])
        }

        if self.transform:
            image = self.transform(image)

        return image, target


def detection_collate_fn(batch):
    return tuple(zip(*batch))
