#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_csv(path: str) -> Tuple[List[str], List[float], List[float]]:
    targets: List[str] = []
    without_vals: List[float] = []
    with_vals: List[float] = []

    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"target_dataset", "without_freqaug_dice", "with_freqaug_dice"}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = required.difference(set(reader.fieldnames or []))
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row in reader:
            targets.append(row["target_dataset"])
            without_vals.append(float(row["without_freqaug_dice"]))
            with_vals.append(float(row["with_freqaug_dice"]))

    if not targets:
        raise ValueError("CSV file has no data rows.")
    return targets, without_vals, with_vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 5 cross-dataset generalization.")
    parser.add_argument("--csv", required=True, help="Path to generalization CSV.")
    parser.add_argument("--train-dataset", required=True, help="Training dataset name.")
    parser.add_argument("--output", default="paper_figures/fig_generalization.png")
    args = parser.parse_args()

    targets, without_vals, with_vals = _load_csv(args.csv)

    x = np.arange(len(targets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars1 = ax.bar(x - width / 2, without_vals, width, label="Without FreqAug", color="#9ecae1")
    bars2 = ax.bar(x + width / 2, with_vals, width, label="With FreqAug", color="#3182bd")

    for idx, (wout, wth) in enumerate(zip(without_vals, with_vals)):
        if wout == 0:
            improvement = 0.0
        else:
            improvement = (wth - wout) / wout * 100.0
        y_pos = max(wout, wth) + 0.01
        ax.text(
            x[idx],
            y_pos,
            f"\u2191 {improvement:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#225ea8",
        )

    ax.set_title(f"Cross-dataset generalization (train: {args.train_dataset})", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=0)
    ax.set_ylabel("Dice score")
    ax.legend(loc="upper left", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    fig.tight_layout()
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
