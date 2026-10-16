#!/usr/bin/env python3
"""
prep_mask.py
High-throughput parallel RLE decoding tool to convert CheXmask CSV annotations
into multi-class segmentation PNG masks (Background, Right Lung, Left Lung, Heart).

The reported run used the CheXmask v1.0.0 CheXpert file:
    chexmask-cxr-segmentation-data/1.0.0/Preprocessed/CheXpert.csv
(columns used: Path, Height, Width, Right Lung, Left Lung, Heart, Dice RCA (Mean)).
Rows with Dice RCA (Mean) < 0.7 are dropped, as in the original run.

Usage:
    python data/prep_mask.py \
        --csv-path /path/to/1.0.0/Preprocessed/CheXpert.csv \
        --train-img-dir /path/to/chexpert/train \
        --val-img-dir /path/to/chexpert/valid \
        --output-dir /path/to/chexmask-masks \
        --num-workers 16
"""

import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def rle_decode(rle: str, shape: tuple) -> np.ndarray:
    """Decodes Run-Length Encoded (RLE) string into a binary mask."""
    if pd.isna(rle) or rle == "" or rle == "nan":
        return np.zeros(shape, dtype=np.uint8)
    s = list(map(int, rle.split()))
    starts, lengths = s[::2], s[1::2]
    starts = np.array(starts) - 1
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1
    return mask.reshape(shape)


SPLIT_PREFIX = re.compile(r"(?:^|/)(?:train|valid)/")


def process_row(args):
    try:
        return _process_row(*args)
    except Exception as e:  # bad RLE / shape: count it instead of killing the pool
        return "error", f"{args[0]['Path']}: {e}"


def _process_row(row, train_img_dir, val_img_dir, train_mask_dir, val_mask_dir):
    rel_path = row["Path"]
    # Strip everything up to and including the first 'train/' or 'valid/'
    # ('train/patientX/...' -> 'patientX/...', also 'CheXpert-v1.0/train/patientX/...').
    m = SPLIT_PREFIX.search(rel_path)
    if m:
        rel_path = rel_path[m.end():]

    if os.path.exists(os.path.join(train_img_dir, rel_path)):
        split = "train"
        target_dir = train_mask_dir
    elif os.path.exists(os.path.join(val_img_dir, rel_path)):
        split = "valid"
        target_dir = val_mask_dir
    else:
        return None, None

    mask_shape = (int(row["Height"]), int(row["Width"]))
    right_lung = rle_decode(row.get("Right Lung", ""), mask_shape)
    left_lung = rle_decode(row.get("Left Lung", ""), mask_shape)
    heart = rle_decode(row.get("Heart", ""), mask_shape)

    mask = np.zeros(mask_shape, dtype=np.uint8)
    mask[right_lung == 1] = 1
    mask[left_lung == 1] = 2
    mask[heart == 1] = 3

    out_path = os.path.join(target_dir, rel_path.replace(".jpg", ".png"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(mask).save(out_path)

    return split, rel_path


def main():
    parser = argparse.ArgumentParser(description="Prepare CheXmask PNG Segmentation Masks")
    parser.add_argument("--csv-path", type=str, required=True,
                        help="CheXmask CSV for CheXpert. The reported run used "
                             "chexmask-cxr-segmentation-data/1.0.0/Preprocessed/CheXpert.csv")
    parser.add_argument("--train-img-dir", type=str, required=True, help="Path to CheXpert train images")
    parser.add_argument("--val-img-dir", type=str, required=True, help="Path to CheXpert valid images")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for generated masks")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--min-rca", type=float, default=0.7,
                        help="Keep rows with 'Dice RCA (Mean)' >= this value (original run: 0.7; 0 disables)")
    args = parser.parse_args()

    train_mask_dir = os.path.join(args.output_dir, "train")
    val_mask_dir = os.path.join(args.output_dir, "valid")
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)

    print(f"Reading CheXmask metadata: {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} metadata rows.")
    if args.min_rca > 0:
        n_before = len(df)
        df = df[df["Dice RCA (Mean)"] >= args.min_rca]
        print(f"RCA filter (Dice RCA (Mean) >= {args.min_rca}): kept {len(df)}, dropped {n_before - len(df)} rows.")

    tasks = [
        (row, args.train_img_dir, args.val_img_dir, train_mask_dir, val_mask_dir)
        for _, row in df.iterrows()
    ]

    train_files, val_files = [], []
    n_unmatched, errors = 0, []
    print(f"Starting parallel mask extraction with {args.num_workers} processes...")
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for split, rel_path in tqdm(executor.map(process_row, tasks), total=len(tasks), desc="Decoding Masks"):
            if split == "train":
                train_files.append(rel_path)
            elif split == "valid":
                val_files.append(rel_path)
            elif split == "error":
                errors.append(rel_path)
            else:
                n_unmatched += 1

    # Save file lists
    with open(os.path.join(args.output_dir, "train_files.txt"), "w") as f:
        f.writelines(f"{p}\n" for p in train_files)
    with open(os.path.join(args.output_dir, "valid_files.txt"), "w") as f:
        f.writelines(f"{p}\n" for p in val_files)

    print(f"\nCompleted! Generated {len(train_files)} train masks and {len(val_files)} valid masks.")
    print(f"Skipped {n_unmatched} rows whose image was not found under --train-img-dir/--val-img-dir, "
          f"{len(errors)} rows that failed to decode.")
    for e in errors[:5]:
        print(f"  decode error: {e}")
    if not train_files and not val_files:
        sys.exit("ERROR: no masks were written. Check that --train-img-dir/--val-img-dir point at the CheXpert "
                 "train/ and valid/ folders and that the CSV 'Path' column contains .../train/... or .../valid/... paths.")


if __name__ == "__main__":
    main()
