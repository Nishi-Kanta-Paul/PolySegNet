#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Iterable, List, Tuple

import torch

from src.utils import ensure_dir


DEFAULT_METHODS = {
    "U-Net": "baseline_smp_unet",
    "U-Net++": "baseline_smp_unetpp",
    "DeepLabV3+": "baseline_smp_deeplabv3plus",
    "FPN": "baseline_smp_fpn",
    "PSPNet": "baseline_smp_pspnet",
    "LinkNet": "baseline_smp_linknet",
    "BGD-SF PolySegNet": "bgdsf_polysegnet_full",
}


METRIC_COLUMNS = [
    "dice",
    "iou",
    "precision",
    "recall",
    "f_measure",
    "mae",
    "boundary_f1",
    "hausdorff",
    "hd95",
    "asd",
    "assd",
]


def _dataset_name_from_path(path: str) -> str:
    return os.path.basename(os.path.normpath(path)) or "dataset"


def _load_checkpoint_dataset_name(exp_name: str) -> str:
    ckpt_dir = os.path.join("experiments", exp_name, "checkpoints")
    for name in ("best.pth", "latest.pth", "last.pth"):
        path = os.path.join(ckpt_dir, name)
        if not os.path.isfile(path):
            continue
        checkpoint = torch.load(path, map_location="cpu")
        cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        roots = cfg.get("dataset_roots") or []
        if isinstance(roots, list) and roots:
            return _dataset_name_from_path(roots[0])
        root = cfg.get("dataset_root") or ""
        if root:
            return _dataset_name_from_path(root)
    return "dataset"


def _find_results_paths(exp_name: str) -> List[Tuple[str, str]]:
    eval_dir = os.path.join("experiments", exp_name, "evaluation")
    if not os.path.isdir(eval_dir):
        return []

    single_results = os.path.join(eval_dir, "results.csv")
    if os.path.isfile(single_results):
        dataset_name = _load_checkpoint_dataset_name(exp_name)
        return [(dataset_name, single_results)]

    paths: List[Tuple[str, str]] = []
    for name in sorted(os.listdir(eval_dir)):
        subdir = os.path.join(eval_dir, name)
        if not os.path.isdir(subdir):
            continue
        csv_path = os.path.join(subdir, "results.csv")
        if os.path.isfile(csv_path):
            paths.append((name, csv_path))
    return paths


def _read_results_csv(path: str) -> Iterable[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def _resolve_method_list(
    experiments: List[str] | None,
    method_names: List[str] | None,
) -> Dict[str, str]:
    if experiments:
        if method_names and len(method_names) != len(experiments):
            raise ValueError("--method-names must match --experiments length.")
        mapping: Dict[str, str] = {}
        for idx, exp in enumerate(experiments):
            method = method_names[idx] if method_names else exp
            mapping[method] = exp
        return mapping
    return dict(DEFAULT_METHODS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-image results table.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Experiment names to include.",
    )
    parser.add_argument(
        "--method-names",
        nargs="+",
        default=None,
        help="Optional display method names aligned with --experiments.",
    )
    parser.add_argument(
        "--output",
        default="outputs/tables/per_image_results.csv",
    )
    args = parser.parse_args()

    mapping = _resolve_method_list(args.experiments, args.method_names)
    rows: List[Dict[str, str]] = []

    for method_name, exp_name in mapping.items():
        result_paths = _find_results_paths(exp_name)
        if not result_paths:
            print(f"Warning: no evaluation results for {exp_name}")
            continue

        for dataset_name, csv_path in result_paths:
            for row in _read_results_csv(csv_path):
                image_name = row.get("image") or row.get("image_name") or ""
                if not image_name:
                    image_path = row.get("image_path") or ""
                    image_name = os.path.splitext(os.path.basename(image_path))[0]
                entry = {
                    "method_name": method_name,
                    "dataset_name": dataset_name,
                    "image_name": image_name,
                }
                for metric in METRIC_COLUMNS:
                    entry[metric] = row.get(metric, "")
                rows.append(entry)

    output_path = args.output
    ensure_dir(os.path.dirname(output_path))
    fieldnames = [
        "method_name",
        "dataset_name",
        "image_name",
        *METRIC_COLUMNS,
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved per-image results to {output_path}")


if __name__ == "__main__":
    main()
