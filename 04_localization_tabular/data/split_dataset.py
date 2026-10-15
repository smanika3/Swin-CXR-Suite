#!/usr/bin/env python3
"""
split_dataset.py
Performs balanced, stratified dataset splitting across 4 diagnostic disease categories:
1. No_TB (Healthy / non-TB controls)
2. Active_TB (Active tuberculosis lesions only)
3. Latent_TB (Obsolete/latent tuberculosis lesions only)
4. Mixed_TB (Concurrent active and obsolete lesions)

Usage:
    python data/split_dataset.py --csv-path /path/to/data.csv --img-dir /path/to/images --output-dir /path/to/output
"""

import os
import random
import argparse
from collections import defaultdict
import pandas as pd


def categorize_image(group: pd.DataFrame) -> str:
    """Categorizes an image based on the set of TB types present in its lesions."""
    tb_types = set(group['tb_type'].dropna().values)
    if 'none' in tb_types and len(tb_types) == 1:
        return 'No_TB'
    elif 'active_tb' in tb_types and 'latent_tb' not in tb_types:
        return 'Active_TB'
    elif 'latent_tb' in tb_types and 'active_tb' not in tb_types:
        return 'Latent_TB'
    elif 'active_tb' in tb_types and 'latent_tb' in tb_types:
        return 'Mixed_TB'
    else:
        return 'No_TB' if len(tb_types) == 0 else 'Unknown'


def main():
    parser = argparse.ArgumentParser(description="Stratified TBX11K Dataset Splitter")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to data.csv")
    parser.add_argument("--img-dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save train/val/test txt files")
    parser.add_argument("--test-size", type=int, default=600, help="Target number of test images")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Proportion of remaining data for train vs val")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} annotation rows from {args.csv_path}")

    # Group by unique filename
    unique_fnames = df['fname'].unique()
    categories = defaultdict(list)

    for fname in unique_fnames:
        img_rows = df[df['fname'] == fname]
        cat = categorize_image(img_rows)
        categories[cat].append(fname)

    print("\n" + "=" * 60)
    print("DATASET STRATIFICATION CATEGORY TOTALS")
    print("=" * 60)
    for cat, fnames in categories.items():
        print(f"  {cat:12s}: {len(fnames)} images")

    # Shuffle within each category
    for cat in categories:
        random.shuffle(categories[cat])

    # Sample test set proportionally
    total_imgs = len(unique_fnames)
    test_files, train_files, val_files = [], [], []

    for cat in ['No_TB', 'Active_TB', 'Latent_TB', 'Mixed_TB', 'Unknown']:
        fnames = categories.get(cat, [])
        if not fnames:
            continue
        cat_test_n = int(round(len(fnames) / total_imgs * args.test_size))
        cat_test = fnames[:cat_test_n]
        remaining = fnames[cat_test_n:]

        split_idx = int(round(len(remaining) * args.train_ratio))
        cat_train = remaining[:split_idx]
        cat_val = remaining[split_idx:]

        test_files.extend(cat_test)
        train_files.extend(cat_train)
        val_files.extend(cat_val)

    print("\n" + "=" * 60)
    print(f"SPLIT RESULTS (Seed = {args.seed})")
    print("=" * 60)
    print(f"  Train Set: {len(train_files)} images")
    print(f"  Val Set:   {len(val_files)} images")
    print(f"  Test Set:  {len(test_files)} images")

    # Write output lists
    for name, flist in [("train.txt", train_files), ("val.txt", val_files), ("test.txt", test_files)]:
        out_p = os.path.join(args.output_dir, name)
        with open(out_p, "w") as f:
            f.writelines(f"{fn}\n" for fn in flist)
        print(f"Saved: {out_p}")


if __name__ == "__main__":
    main()
