#!/usr/bin/env python3
"""
analyze_dataset.py
Audits dataset splits and outputs distributions of images and bounding boxes per TB category.

Usage:
    python data/analyze_dataset.py --csv-path /path/to/data.csv --splits-dir /path/to/splits
"""

import os
import argparse
import pandas as pd
from split_dataset import categorize_image


def main():
    parser = argparse.ArgumentParser(description="Analyze TBX11K Dataset Distributions")
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--splits-dir", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    split_basenames = {}
    for s in ["train", "val", "test"]:
        txt_path = os.path.join(args.splits_dir, f"{s}.txt")
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                split_basenames[s] = set(line.strip().split('/')[-1] for line in f if line.strip())
        else:
            split_basenames[s] = set()

    print("=" * 80)
    print("TBX11K SPLIT AND CATEGORY ANALYSIS")
    print("=" * 80)

    for s, files in split_basenames.items():
        if not files:
            continue
        sub_df = df[df['fname'].isin(files)]
        unique_fnames = sub_df['fname'].unique()
        print(f"\n{s.upper()} SET: {len(unique_fnames)} Images")
        print("-" * 50)

        cat_counts = {}
        for fn in unique_fnames:
            c = categorize_image(sub_df[sub_df['fname'] == fn])
            cat_counts[c] = cat_counts.get(c, 0) + 1

        for c, count in sorted(cat_counts.items()):
            pct = (count / len(unique_fnames)) * 100
            print(f"  {c:12s}: {count:5d} ({pct:5.1f}%)")

        boxes = sub_df[sub_df['bbox'] != 'none']
        print(f"  Total Bounding Boxes: {len(boxes)}")


if __name__ == "__main__":
    main()
