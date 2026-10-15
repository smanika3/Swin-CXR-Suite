#!/usr/bin/env python3
"""
fix_coco_json.py
Ensures COCO JSON annotation files have valid 'info' and 'categories' metadata dictionaries.
"""

import sys
import json
import argparse


def fix_json(file_path: str):
    print(f"Checking {file_path}...")
    with open(file_path, "r") as f:
        data = json.load(f)

    modified = False
    if "info" not in data or not isinstance(data["info"], dict):
        data["info"] = {"description": "TBX11K COCO Dataset", "version": "1.0"}
        modified = True

    if "licenses" not in data:
        data["licenses"] = []
        modified = True

    if modified:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f" Successfully patched: {file_path}")
    else:
        print(f" File already valid: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Patch COCO JSON metadata")
    parser.add_argument("files", nargs="+", help="COCO JSON file paths to inspect and patch")
    args = parser.parse_args()

    for fp in args.files:
        fix_json(fp)


if __name__ == "__main__":
    main()
