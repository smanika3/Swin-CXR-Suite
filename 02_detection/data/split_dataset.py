#!/usr/bin/env python3
"""
split_dataset.py
Builds train/val/test split files for TBX11K (simplified CSV format).

- val  = every image with source == 'val' (the official TBX11K validation set).
- test = carved out of the source == 'train' images, per category
         (No_TB, Active_TB, Latent_TB, Mixed_TB): int(0.3 * n) images,
         with at least 20 Latent_TB and 5 Mixed_TB images.
- train = the remaining source == 'train' images.
Images whose category is 'Unknown' (source == 'train') are left out of every split.

Usage:
    python data/split_dataset.py --csv-path /path/to/data.csv --output-dir /path/to/output
"""

import os
import random
import argparse
from collections import defaultdict
import pandas as pd


def categorize_image(group: pd.DataFrame) -> str:
    """Categorizes an image based on the set of TB types present in its lesions."""
    tb_types = set(group['tb_type'].values)
    if 'none' in tb_types and len(tb_types) == 1:
        return 'No_TB'
    elif 'active_tb' in tb_types and 'latent_tb' not in tb_types:
        return 'Active_TB'
    elif 'latent_tb' in tb_types and 'active_tb' not in tb_types:
        return 'Latent_TB'
    elif 'active_tb' in tb_types and 'latent_tb' in tb_types:
        return 'Mixed_TB'
    else:
        return 'Unknown'


# Minimum number of test images for the rare categories
MIN_TEST = {'Latent_TB': 20, 'Mixed_TB': 5}
TRAIN_SPLIT_RATIO = 0.7  # 70% train, 30% test from the source == 'train' images


def split(df: pd.DataFrame, seed: int = 42):
    """Returns (train_fnames, val_fnames, test_fnames)."""
    random.seed(seed)

    train_source_images = df[df['source'] == 'train']['fname'].unique()
    val_images = list(df[df['source'] == 'val']['fname'].unique())

    image_categories = defaultdict(list)
    for fname in train_source_images:
        image_categories[categorize_image(df[df['fname'] == fname])].append(fname)

    # Shuffle each category (all categories, incl. Unknown, to keep the RNG sequence)
    for cat in image_categories:
        random.shuffle(image_categories[cat])

    test_images, train_images = [], []
    print("=" * 60)
    print(f"STRATIFIED SPLIT RESULTS (Seed = {seed})")
    print("=" * 60)
    for cat in ['No_TB', 'Active_TB', 'Latent_TB', 'Mixed_TB']:
        cat_images = image_categories[cat]
        n_total = len(cat_images)
        n_test = max(MIN_TEST.get(cat, 0), int(n_total * (1 - TRAIN_SPLIT_RATIO)))
        n_test = min(n_test, n_total)

        test_images.extend(cat_images[:n_test])
        train_images.extend(cat_images[n_test:])
        print(f"  {cat:10s}: total {n_total:5d} | train {n_total - n_test:5d} | test {n_test:5d}")
    if image_categories.get('Unknown'):
        print(f"  Dropped {len(image_categories['Unknown'])} 'Unknown' images")

    random.shuffle(train_images)
    random.shuffle(test_images)

    print(f"\n  Train Set: {len(train_images)} images")
    print(f"  Val Set:   {len(val_images)} images")
    print(f"  Test Set:  {len(test_images)} images")
    return train_images, val_images, test_images


def main():
    parser = argparse.ArgumentParser(description="Stratified TBX11K Dataset Splitter")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to data.csv")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save train/val/test txt files")
    parser.add_argument("--img-dir", type=str, default=None,
                        help="Images directory written as the path prefix in the txt files (default: <csv dir>/images)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    img_dir = args.img_dir or os.path.join(os.path.dirname(os.path.abspath(args.csv_path)), "images")
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} annotation rows from {args.csv_path}")
    train_files, val_files, test_files = split(df, args.seed)

    for name, flist in [("train.txt", train_files), ("val.txt", val_files), ("test.txt", test_files)]:
        out_p = os.path.join(args.output_dir, name)
        with open(out_p, "w") as f:
            f.writelines(os.path.join(img_dir, fn) + "\n" for fn in flist)
        print(f"Saved: {out_p}")


if __name__ == "__main__":
    main()
