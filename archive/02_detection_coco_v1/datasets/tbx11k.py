"""
tbx11k.py
PyTorch Dataset and DataLoader collation for the TBX11K COCO-formatted dataset.
"""

import os
import json
from collections import defaultdict
from typing import Optional, Callable, List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TBX11KDetectionDataset(Dataset):
    """
    TBX11K Object Detection Dataset in COCO format.

    Args:
        images_dir: Directory containing images.
        annotation_file: Path to the COCO JSON annotations.
        transform: Optional torchvision transforms applied to PIL images.
    """
    def __init__(self, images_dir: str, annotation_file: str, transform: Optional[Callable] = None):
        self.images_dir = images_dir
        self.transform = transform if transform else transforms.ToTensor()

        with open(annotation_file, 'r') as f:
            ann = json.load(f)

        self.img_id_to_file = {img['id']: img['file_name'] for img in ann['images']}
        self.img_ids = list(self.img_id_to_file.keys())

        self.annotations = defaultdict(list)
        for obj in ann.get('annotations', []):
            self.annotations[obj['image_id']].append(obj)

        # Foreground categories mapped to labels 1..N (0 is background in Faster R-CNN)
        categories = ann.get('categories', [])
        self.cat_id_to_label = {c['id']: idx + 1 for idx, c in enumerate(categories)}
        self.categories = [c['name'] for c in categories]

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        img_id = self.img_ids[idx]
        file_name = self.img_id_to_file[img_id]
        img_path = os.path.join(self.images_dir, file_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert('RGB')
        objs = self.annotations.get(img_id, [])

        boxes, labels = [], []
        for obj in objs:
            x, y, w, h = obj['bbox']
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                labels.append(self.cat_id_to_label.get(obj['category_id'], 1))

        if len(boxes) == 0:
            target_boxes = torch.zeros((0, 4), dtype=torch.float32)
            target_labels = torch.zeros((0,), dtype=torch.int64)
        else:
            target_boxes = torch.as_tensor(boxes, dtype=torch.float32)
            target_labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': target_boxes,
            'labels': target_labels,
            'image_id': torch.tensor([img_id])
        }

        image = self.transform(image)
        return image, target


def detection_collate_fn(batch):
    """Filters out invalid samples and collates tuples of images and targets."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return tuple(zip(*batch))
