#!/usr/bin/env python3
"""
convert_coco.py
Convert raw TBX11K dataset annotations to standard COCO detection format.

Usage:
    python tools/convert_coco.py --dataset-root /path/to/tbx11k --output-dir /path/to/tbx11k/annotations
"""

import os
import json
import argparse
from tqdm import tqdm

CATEGORIES = ["ActiveTuberculosis", "ObsoletePulmonaryTuberculosis"]
CATEGORY_MAP = {name: idx + 1 for idx, name in enumerate(CATEGORIES)}


def process_split(split, dataset_root, output_dir):
    """Processes images and annotation JSONs for a given split (train/val/test)."""
    split_dir = os.path.join(dataset_root, split)
    img_dir = os.path.join(split_dir, "img")
    ann_dir = os.path.join(split_dir, "ann")

    if not os.path.exists(img_dir):
        print(f"[WARN] Image directory not found: {img_dir}. Skipping split '{split}'.")
        return

    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    valid_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Processing split: '{split}' ({len(valid_files)} candidate images)...")

    for fname in tqdm(valid_files, desc=f"Converting {split}"):
        img_path = os.path.join(img_dir, fname)
        # Annotation files in TBX11K typically correspond to <fname>.json or <basename>.json
        ann_filename = fname + ".json" if not fname.endswith(".json") else fname
        ann_path = os.path.join(ann_dir, ann_filename)

        if not os.path.exists(ann_path):
            base_name, _ = os.path.splitext(fname)
            ann_path = os.path.join(ann_dir, base_name + ".json")
            if not os.path.exists(ann_path):
                continue

        try:
            with open(ann_path, "r") as f:
                ann_data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read {ann_path}: {e}")
            continue

        width = ann_data.get("size", {}).get("width", 0)
        height = ann_data.get("size", {}).get("height", 0)

        images.append({
            "id": img_id,
            "file_name": fname,
            "width": width,
            "height": height
        })

        for obj in ann_data.get("objects", []):
            class_title = obj.get("classTitle")
            if class_title not in CATEGORY_MAP:
                continue

            points = obj.get("points", {}).get("exterior", [])
            if len(points) < 2:
                continue

            x1, y1 = points[0]
            x2, y2 = points[1]
            w = max(0.0, float(x2 - x1))
            h = max(0.0, float(y2 - y1))

            if w <= 0 or h <= 0:
                continue

            bbox = [float(x1), float(y1), w, h]
            area = w * h

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CATEGORY_MAP[class_title],
                "bbox": bbox,
                "area": area,
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

    coco_data = {
        "info": {
            "description": f"TBX11K Tuberculosis Dataset - {split} split",
            "year": 2024
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": idx + 1, "name": name} for idx, name in enumerate(CATEGORIES)
        ]
    }

    out_json = os.path.join(output_dir, f"{split}.json")
    with open(out_json, "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f" Saved {out_json} | Images: {len(images)}, Bounding Boxes: {len(annotations)}")


def main():
    parser = argparse.ArgumentParser(description="Convert TBX11K Annotations to COCO format.")
    parser.add_argument("--dataset-root", type=str, required=True,
                        help="Root directory of TBX11K containing train/val/test subdirectories.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Destination directory for COCO JSON files. Defaults to <dataset-root>/annotations.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Dataset splits to process.")
    args = parser.parse_args()

    out_dir = args.output_dir if args.output_dir else os.path.join(args.dataset_root, "annotations")
    os.makedirs(out_dir, exist_ok=True)

    for split in args.splits:
        process_split(split, args.dataset_root, out_dir)


if __name__ == "__main__":
    main()
