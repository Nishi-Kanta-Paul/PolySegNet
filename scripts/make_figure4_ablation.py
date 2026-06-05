#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

EXPECTED_ORDER = [
    "EfficientNet-B4 U-Net Baseline",
    "+ Original MSCA (parallel)",
    "+ Cascaded CMSCA (static)",
    "+ BGD-CMSCA (dynamic)",
    "+ SAGF (plain)",
    "+ BG-SAGF",
    "+ MBGH",
    "Full BGD-SF (FreqAug)",
]
FULL_MODEL_LABEL = "Full BGD-SF (FreqAug)"


def _load_csv(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if "Model" not in fieldnames:
            raise ValueError("CSV must include 'Model' column.")
        if "Dice" not in fieldnames:
            raise ValueError("CSV must include 'Dice' column.")

        results: Dict[str, Dict[str, float]] = {}
        for row in reader:
            variant = row["Model"].strip()
            dice_str = row.get("Dice", "").strip()
            results[variant] = {"Dice": float(dice_str) if dice_str else float("nan")}

    missing = [name for name in EXPECTED_ORDER if name not in results]
    if missing:
        raise ValueError(f"Missing variants in CSV: {missing}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 4 ablation bar chart.")
    parser.add_argument("--csv", required=True, help="Path to ablation CSV.")
    parser.add_argument("--output", default="paper_figures/fig_ablation.png")
    args = parser.parse_args()

    results = _load_csv(args.csv)
    datasets = list(next(iter(results.values())).keys())

    x = np.arange(len(EXPECTED_ORDER))
    width = 0.8 / max(len(datasets), 1)

    plt.rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
    })

    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, dataset in enumerate(datasets):
        offsets = x - 0.4 + width / 2 + idx * width
        values = [results[variant][dataset] for variant in EXPECTED_ORDER]
        bars = ax.bar(offsets, values, width=width, label=dataset)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    full_model_values = [results[FULL_MODEL_LABEL][dataset] for dataset in datasets]
    full_model_avg = float(np.mean(full_model_values)) if full_model_values else 0.0
    ax.axhline(full_model_avg, color="red", linestyle="--", linewidth=1.0)
    ax.text(
        len(EXPECTED_ORDER) - 0.5,
        full_model_avg + 0.01,
        "Full model",
        color="red",
        ha="right",
        va="bottom",
        fontsize=9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(EXPECTED_ORDER, rotation=15, ha="right")
    ax.set_ylabel("Dice score")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
