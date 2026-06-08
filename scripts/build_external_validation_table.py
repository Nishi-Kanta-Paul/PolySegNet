#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from types import SimpleNamespace
from typing import Dict, List

import torch

from src.utils import ensure_dir, load_json


METRICS = [
    "dice",
    "iou",
    "precision",
    "recall",
    "f_measure",
    "mae",
    "mask_boundary_f1",
    "mask_hd",
    "mask_hd95",
    "mask_asd",
    "mask_assd",
]


def _dataset_name_from_path(path: str) -> str:
    return os.path.basename(os.path.normpath(path)) or "dataset"


def _infer_train_dataset(exp_name: str) -> str:
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
        break
    return "dataset"


def _load_eval_summary(exp_name: str) -> Dict[str, Dict[str, float]]:
    eval_dir = os.path.join("experiments", exp_name, "evaluation")
    summary_path = os.path.join(eval_dir, "summary.json")
    results_path = os.path.join(eval_dir, "results.json")

    if os.path.isfile(summary_path):
        payload = load_json(summary_path)
        datasets = payload.get("datasets", {}) if isinstance(payload, dict) else {}
        return {name: dict(values) for name, values in datasets.items()}

    if os.path.isfile(results_path):
        payload = load_json(results_path)
        summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        return {"dataset": dict(summary)}

    return {}


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build external validation summary table.")
    parser.add_argument("--experiment", required=True, help="Experiment name.")
    parser.add_argument("--train-dataset", default=None, help="Override training dataset label.")
    parser.add_argument(
        "--output-csv",
        default="outputs/tables/external_validation.csv",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/tables/external_validation.json",
    )
    args = parser.parse_args()

    per_dataset = _load_eval_summary(args.experiment)
    if not per_dataset:
        raise FileNotFoundError(f"No evaluation summary found for {args.experiment}.")

    train_dataset = args.train_dataset or _infer_train_dataset(args.experiment)

    rows: List[Dict[str, str]] = []
    for dataset_name, metrics in per_dataset.items():
        row: Dict[str, str] = {
            "train_dataset": train_dataset,
            "target_dataset": dataset_name,
        }
        for metric in METRICS:
            row[metric] = _format_number(metrics.get(metric))
        rows.append(row)

    output_csv = args.output_csv
    ensure_dir(os.path.dirname(output_csv))
    fieldnames = ["train_dataset", "target_dataset", *METRICS]
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    output_json = args.output_json
    ensure_dir(os.path.dirname(output_json))
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump({"results": rows}, handle, indent=2)

    print(f"Saved external validation table to {output_csv}")


if __name__ == "__main__":
    main()
