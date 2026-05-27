#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.dataset import IMAGE_EXTS, DEFAULT_SPLIT_RATIOS
from src.utils import ensure_dir


def _resolve_dir(base: str, subdir: str) -> str:
    if not subdir:
        return base
    if os.path.isabs(subdir):
        return subdir
    if base:
        return os.path.join(base, subdir)
    return subdir


def _list_images(root: str) -> List[str]:
    if not root or not os.path.isdir(root):
        return []
    return [
        os.path.join(root, name)
        for name in sorted(os.listdir(root))
        if name.lower().endswith(IMAGE_EXTS)
    ]


def _read_split_file(path: str) -> List[str]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _split_counts(
    dataset_root: str,
    total_images: int,
    ratios: Tuple[float, float, float],
    split_dir: str | None,
) -> Tuple[int, int, int, str]:
    split_dir = split_dir or dataset_root or "."
    train_path = os.path.join(split_dir, "train.txt")
    val_path = os.path.join(split_dir, "val.txt")
    test_path = os.path.join(split_dir, "test.txt")

    if os.path.isfile(train_path) and os.path.isfile(val_path) and os.path.isfile(test_path):
        train_count = len(_read_split_file(train_path))
        val_count = len(_read_split_file(val_path))
        test_count = len(_read_split_file(test_path))
        return train_count, val_count, test_count, "files"

    train_count = int(total_images * ratios[0])
    val_count = int(total_images * ratios[1])
    test_count = max(total_images - train_count - val_count, 0)
    return train_count, val_count, test_count, "ratios"


def _match_masks(image_paths: List[str], mask_paths: List[str]) -> int:
    name_map = {os.path.basename(path): path for path in mask_paths}
    stem_map = {os.path.splitext(os.path.basename(path))[0]: path for path in mask_paths}
    matched = 0
    for image_path in image_paths:
        name = os.path.basename(image_path)
        stem = os.path.splitext(name)[0]
        if name in name_map or stem in stem_map:
            matched += 1
    return matched


def _resolution_stats(image_paths: List[str]) -> Tuple[float, float, int, int, int, int]:
    heights: List[int] = []
    widths: List[int] = []
    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        h, w = image.shape[:2]
        heights.append(int(h))
        widths.append(int(w))

    if not heights or not widths:
        return 0.0, 0.0, 0, 0, 0, 0

    mean_h = float(np.mean(heights))
    mean_w = float(np.mean(widths))
    min_h = int(min(heights))
    min_w = int(min(widths))
    max_h = int(max(heights))
    max_w = int(max(widths))
    return mean_h, mean_w, min_h, min_w, max_h, max_w


def _dataset_name(path: str) -> str:
    return os.path.basename(os.path.normpath(path)) or "dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset summary table.")
    parser.add_argument("--datasets", nargs="+", default=[])
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--mask-dir", default="masks")
    parser.add_argument("--split-dir", default=None)
    parser.add_argument(
        "--split-ratios",
        default=None,
        help="Comma-separated ratios for train,val,test (default: 0.8,0.1,0.1)",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/tables/dataset_summary.csv",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/tables/dataset_summary.json",
    )
    args = parser.parse_args()

    dataset_roots = list(args.datasets)
    if args.dataset_root:
        dataset_roots.append(args.dataset_root)

    if not dataset_roots:
        raise ValueError("Provide --datasets or --dataset-root.")

    if args.split_ratios:
        parts = [float(p.strip()) for p in args.split_ratios.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError("--split-ratios must have 3 comma-separated values.")
        ratios = (parts[0], parts[1], parts[2])
    else:
        ratios = DEFAULT_SPLIT_RATIOS

    rows: List[Dict[str, object]] = []
    for root in dataset_roots:
        dataset_name = _dataset_name(root)
        images_dir = _resolve_dir(root, args.image_dir)
        masks_dir = _resolve_dir(root, args.mask_dir)

        image_paths = _list_images(images_dir)
        mask_paths = _list_images(masks_dir)

        matched_masks = _match_masks(image_paths, mask_paths)
        train_count, val_count, test_count, split_source = _split_counts(
            root,
            len(image_paths),
            ratios,
            args.split_dir,
        )
        mean_h, mean_w, min_h, min_w, max_h, max_w = _resolution_stats(image_paths)

        rows.append(
            {
                "dataset_name": dataset_name,
                "num_images": len(image_paths),
                "num_masks": len(mask_paths),
                "matched_masks": matched_masks,
                "train_count": train_count,
                "val_count": val_count,
                "test_count": test_count,
                "split_source": split_source,
                "mean_height": round(mean_h, 2),
                "mean_width": round(mean_w, 2),
                "min_height": min_h,
                "min_width": min_w,
                "max_height": max_h,
                "max_width": max_w,
            }
        )

    output_csv = args.output_csv
    ensure_dir(os.path.dirname(output_csv))
    fieldnames = [
        "dataset_name",
        "num_images",
        "num_masks",
        "matched_masks",
        "train_count",
        "val_count",
        "test_count",
        "split_source",
        "mean_height",
        "mean_width",
        "min_height",
        "min_width",
        "max_height",
        "max_width",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    output_json = args.output_json
    ensure_dir(os.path.dirname(output_json))
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump({"datasets": rows}, handle, indent=2)

    print(f"Saved dataset summary to {output_csv}")


if __name__ == "__main__":
    main()
