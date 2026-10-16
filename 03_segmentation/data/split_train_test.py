#!/usr/bin/env python3
"""
split_train_test.py
Split the CheXmask train image list (train_files.txt from prep_mask.py) 70/30 into train and test lists.

This is the split used for the reported results: sklearn train_test_split(test_size=0.3, random_state=42)
over the list of image paths. It is an IMAGE-level split, not patient-independent: images of the same
patient can appear in both train and test.

Unlike the original script, the input list is not overwritten; the 70% part goes to --train-out.

Usage:
    python data/split_train_test.py \
        --input /path/to/chexmask-masks/train_files.txt \
        --train-out /path/to/chexmask-masks/train_split.txt \
        --test-out /path/to/chexmask-masks/test_files.txt
"""

import argparse
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="Image-level 70/30 train/test split of the CheXmask train list")
    parser.add_argument("--input", type=str, default="train_files.txt", help="Image list written by prep_mask.py")
    parser.add_argument("--train-out", type=str, default="train_split.txt", help="Output list for the 70%% train part")
    parser.add_argument("--test-out", type=str, default="test_files.txt", help="Output list for the 30%% test part")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        train_files = [line.strip() for line in f if line.strip()]

    train_part, test_part = train_test_split(train_files, test_size=args.test_size, random_state=args.seed)

    with open(args.train_out, "w") as f:
        f.writelines(f"{p}\n" for p in train_part)
    with open(args.test_out, "w") as f:
        f.writelines(f"{p}\n" for p in test_part)

    print(f"Train: {len(train_part)}, Test: {len(test_part)}")


if __name__ == "__main__":
    main()
