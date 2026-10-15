"""
dataset.py
PyTorch Dataset for CheXmask Multi-Organ Segmentation.
"""

import os
from typing import Optional, Callable
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CheXmaskSegDataset(Dataset):
    """
    CheXmask Segmentation Dataset.

    Args:
        img_base: Base directory for image files.
        mask_base: Base directory for mask files.
        file_list: Text file containing relative image paths.
        img_size: Target square image dimension (e.g., 224).
        transform: Image transform.
    """
    def __init__(
        self,
        img_base: str,
        mask_base: str,
        file_list: str,
        img_size: int = 224,
        transform: Optional[Callable] = None
    ):
        self.img_base = img_base
        self.mask_base = mask_base
        self.img_size = img_size

        with open(file_list, "r") as f:
            self.images = [line.strip() for line in f if line.strip()]

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        rel_img_path = self.images[idx]
        rel_mask_path = rel_img_path.replace(".jpg", ".png")

        img_path = os.path.join(self.img_base, rel_img_path)
        mask_path = os.path.join(self.mask_base, rel_mask_path)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Nearest neighbor resize for masks to preserve integer class indices
        mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        mask_arr = np.array(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_arr)

        if self.transform:
            image = self.transform(image)

        return image, mask_tensor
