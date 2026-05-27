#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from src.utils import ensure_dir, load_json


SEEDS = [42, 123, 2025]
EXPERIMENTS = {
    42: "bgdsf_polysegnet_seed42",
    123: "bgdsf_polysegnet_seed123",
    2025: "bgdsf_polysegnet_seed2025",
}

METRICS = [
    "dice",
    "iou",
    "precision",
    "recall",
    "f_measure",
    "mae",
    "boundary_f1",
    "hausdorff",
]

PAPER_METRICS = [
    "dice",
    "iou",
    "mae",
    "boundary_f1",
    "hausdorff",
]


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


def _format_mean_std(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return ""
    return f"{mean:.3f} ± {std:.3f}"


def _collect_datasets(results_by_seed: Dict[int, Dict[str, Dict[str, float]]]) -> List[str]:
    datasets = set()
    for data in results_by_seed.values():
        datasets.update(data.keys())
    return sorted(datasets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize multi-seed BGD-SF results.")
    parser.add_argument(
        "--output-csv",
        default="outputs/tables/bgdsf_multiseed_summary.csv",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/tables/bgdsf_multiseed_summary.json",
    )
    parser.add_argument(
        "--paper-csv",
        default="outputs/tables/bgdsf_multiseed_paper_table.csv",
    )
    args = parser.parse_args()

    results_by_seed: Dict[int, Dict[str, Dict[str, float]]] = {}
    for seed, exp_name in EXPERIMENTS.items():
        data = _load_eval_summary(exp_name)
        if not data:
            print(f"Warning: missing evaluation results for {exp_name}")
        results_by_seed[seed] = data

    datasets = _collect_datasets(results_by_seed)
    rows: List[Dict[str, object]] = []
    paper_rows: List[Dict[str, str]] = []

    for dataset_name in datasets:
        for metric in METRICS:
            seed_values: Dict[int, float] = {}
            for seed in SEEDS:
                value = results_by_seed.get(seed, {}).get(dataset_name, {}).get(metric)
                if value is None:
                    continue
                seed_values[seed] = float(value)

            values = list(seed_values.values())
            if not values:
                continue

            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            rows.append(
                {
                    "model_name": "bgdsf_polysegnet",
                    "dataset_name": dataset_name,
                    "metric": metric,
                    "seed_42": seed_values.get(42),
                    "seed_123": seed_values.get(123),
                    "seed_2025": seed_values.get(2025),
                    "mean": mean,
                    "std": std,
                    "formatted_mean_std": _format_mean_std(mean, std),
                    "n_seeds_used": len(values),
                }
            )

        paper_row: Dict[str, str] = {"dataset_name": dataset_name}
        for metric in PAPER_METRICS:
            metric_values = [
                row for row in rows if row["dataset_name"] == dataset_name and row["metric"] == metric
            ]
            if not metric_values:
                paper_row[metric] = ""
                continue
            row = metric_values[0]
            paper_row[metric] = row["formatted_mean_std"]
        paper_rows.append(paper_row)

    ensure_dir(os.path.dirname(args.output_csv))
    with open(args.output_csv, "w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "model_name",
            "dataset_name",
            "metric",
            "seed_42",
            "seed_123",
            "seed_2025",
            "mean",
            "std",
            "formatted_mean_std",
            "n_seeds_used",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    ensure_dir(os.path.dirname(args.output_json))
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump({"results": rows}, handle, indent=2)

    ensure_dir(os.path.dirname(args.paper_csv))
    with open(args.paper_csv, "w", encoding="utf-8", newline="") as handle:
        fieldnames = ["dataset_name", *PAPER_METRICS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in paper_rows:
            writer.writerow(row)

    print(f"Saved summary CSV to {args.output_csv}")
    print(f"Saved summary JSON to {args.output_json}")
    print(f"Saved paper table CSV to {args.paper_csv}")


if __name__ == "__main__":
    main()
