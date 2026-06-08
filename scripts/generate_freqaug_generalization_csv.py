#!/usr/bin/env python3
"""Generate the FreqAug generalization CSV required by make_figure5_generalization.py.

Reads evaluation summaries for:
  - bgdsf_polysegnet_no_freqaug   (without FreqAug)
  - bgdsf_polysegnet_full         (with FreqAug)

and writes a CSV with columns:
  target_dataset, without_freqaug_dice, with_freqaug_dice

Usage:
  python scripts/generate_freqaug_generalization_csv.py \
      --datasets CVC-ClinicDB CVC-ColonDB ETIS-Larib CVC-300 \
      --output outputs/tables/freqaug_generalization.csv

Then generate the figure:
  python scripts/make_figure5_generalization.py \
      --csv outputs/tables/freqaug_generalization.csv \
      --train-dataset Kvasir-SEG \
      --output paper_figures/fig_generalization.png
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

EXP_WITHOUT = "bgdsf_polysegnet_no_freqaug"
EXP_WITH = "bgdsf_polysegnet_full"


def _load_results_json(path: str) -> Dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _get_dice(exp_name: str, dataset_name: str) -> Optional[float]:
    """Try per-dataset subdir first, then summary.json, then flat results.json."""
    eval_dir = os.path.join("experiments", exp_name, "evaluation")

    # 1. Per-dataset subdir
    subdir = os.path.join(eval_dir, dataset_name, "results.json")
    payload = _load_results_json(subdir)
    if payload:
        summary = payload.get("summary", {})
        if isinstance(summary, dict) and "dice" in summary:
            return float(summary["dice"])

    # 2. summary.json (multi-dataset run)
    summary_path = os.path.join(eval_dir, "summary.json")
    payload = _load_results_json(summary_path)
    if payload:
        datasets = payload.get("datasets", {})
        if isinstance(datasets, dict) and dataset_name in datasets:
            d = datasets[dataset_name]
            if isinstance(d, dict) and "dice" in d:
                return float(d["dice"])

    # 3. Flat results.json (only if single dataset and matches)
    flat_path = os.path.join(eval_dir, "results.json")
    payload = _load_results_json(flat_path)
    if payload:
        summary = payload.get("summary", {})
        if isinstance(summary, dict) and "dice" in summary:
            return float(summary["dice"])

    return None


def _auto_discover_datasets(exp_name: str) -> List[str]:
    eval_dir = os.path.join("experiments", exp_name, "evaluation")
    if not os.path.isdir(eval_dir):
        return []

    # From per-dataset subdirs
    datasets = []
    for name in sorted(os.listdir(eval_dir)):
        if os.path.isdir(os.path.join(eval_dir, name)):
            datasets.append(name)

    if datasets:
        return datasets

    # From summary.json
    summary_path = os.path.join(eval_dir, "summary.json")
    payload = _load_results_json(summary_path)
    if payload:
        d = payload.get("datasets", {})
        if isinstance(d, dict):
            return sorted(d.keys())

    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate FreqAug generalization CSV for Figure 5."
    )
    parser.add_argument(
        "--without-exp",
        default=EXP_WITHOUT,
        help=f"Experiment name without FreqAug (default: {EXP_WITHOUT}).",
    )
    parser.add_argument(
        "--with-exp",
        default=EXP_WITH,
        help=f"Experiment name with FreqAug (default: {EXP_WITH}).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset names to include. Auto-discovered if omitted.",
    )
    parser.add_argument(
        "--output",
        default="outputs/tables/freqaug_generalization.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    datasets = args.datasets
    if not datasets:
        datasets = _auto_discover_datasets(args.with_exp)
        if not datasets:
            datasets = _auto_discover_datasets(args.without_exp)
        if not datasets:
            print(
                "ERROR: Could not auto-discover datasets. "
                "Specify them with --datasets <name1> <name2> ..."
            )
            sys.exit(1)
        print(f"Auto-discovered datasets: {datasets}")

    rows = []
    any_missing = False
    for ds in datasets:
        dice_without = _get_dice(args.without_exp, ds)
        dice_with = _get_dice(args.with_exp, ds)

        if dice_without is None:
            print(
                f"WARNING: No Dice score found for '{args.without_exp}' on dataset '{ds}'. "
                f"Expected: experiments/{args.without_exp}/evaluation/{ds}/results.json"
            )
            any_missing = True
        if dice_with is None:
            print(
                f"WARNING: No Dice score found for '{args.with_exp}' on dataset '{ds}'. "
                f"Expected: experiments/{args.with_exp}/evaluation/{ds}/results.json"
            )
            any_missing = True

        rows.append({
            "target_dataset": ds,
            "without_freqaug_dice": f"{dice_without:.6f}" if dice_without is not None else "",
            "with_freqaug_dice": f"{dice_with:.6f}" if dice_with is not None else "",
        })

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = ["target_dataset", "without_freqaug_dice", "with_freqaug_dice"]
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {args.output}")
    if any_missing:
        print(
            "\nSome evaluation outputs are missing. Run evaluation first:\n"
            f"  ./scripts/evaluate_ablation.sh\n"
            "or evaluate individual experiments:\n"
            f"  python src/main.py --mode eval --experiment-name {args.without_exp} ...\n"
            f"  python src/main.py --mode eval --experiment-name {args.with_exp} ..."
        )


if __name__ == "__main__":
    main()
