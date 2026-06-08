#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

EXPECTED_ORDER = [
    "EffB4 encoder-decoder baseline",
    "EffB4 + Original MSCA",
    "EffB4 + CMSCA (static)",
    "EffB4 + BGD-CMSCA",
    "BGD-CMSCA + SAGF (plain)",
    "BGD-CMSCA + BG-SAGF",
    "BGD-SF (no FreqAug)",
    "BGD-SF (FreqAug)",
]
FULL_MODEL_LABEL = "BGD-SF (FreqAug)"

# Short x-axis labels to save horizontal space
SHORT_LABELS = [
    "Baseline",
    "+Orig.MSCA",
    "+CMSCA\n(static)",
    "+BGD-\nCMSCA",
    "+SAGF\n(plain)",
    "+BG-SAGF",
    "BGD-SF\n(no FA)",
    "BGD-SF\n(Full)",
]

DICE_COLOR = "#4878cf"
BF1_COLOR = "#e87c1e"
Y_MIN = 0.85


def _load_csv(path: str) -> Dict[str, Dict[str, float]]:
    """Load ablation CSV; expects Model, Dice, and optionally mask_boundary_f1 columns."""
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if "Model" not in fieldnames:
            raise ValueError("CSV must include 'Model' column.")
        if "Dice" not in fieldnames:
            raise ValueError("CSV must include 'Dice' column.")

        has_bf1 = "mask_boundary_f1" in fieldnames

        results: Dict[str, Dict[str, float]] = {}
        for row in reader:
            variant = row["Model"].strip()
            dice_str = row.get("Dice", "").strip()
            entry: Dict[str, float] = {
                "Dice": float(dice_str) if dice_str else float("nan"),
            }
            if has_bf1:
                bf1_str = row.get("mask_boundary_f1", "").strip()
                entry["BF1"] = float(bf1_str) if bf1_str else float("nan")
            results[variant] = entry

    missing = [name for name in EXPECTED_ORDER if name not in results]
    if missing:
        raise ValueError(f"Missing variants in CSV: {missing}")
    return results


def _annotate_bar(ax: plt.Axes, bar, y_min: float) -> None:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.001,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=7.5,
        rotation=90,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 4 ablation bar chart.")
    parser.add_argument("--csv", required=True, help="Path to ablation CSV.")
    parser.add_argument("--output", default="paper_figures/fig_ablation.png")
    parser.add_argument(
        "--ymin",
        type=float,
        default=Y_MIN,
        help="Y-axis lower bound (default: 0.85).",
    )
    args = parser.parse_args()

    results = _load_csv(args.csv)
    has_bf1 = "BF1" in next(iter(results.values()))

    n = len(EXPECTED_ORDER)
    x = np.arange(n)

    # Two metrics → two bars per model; otherwise single bar
    n_metrics = 2 if has_bf1 else 1
    bar_width = 0.35 if n_metrics == 2 else 0.55

    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
    })

    fig, ax = plt.subplots(figsize=(12, 5.5))

    if n_metrics == 2:
        offsets_dice = x - bar_width / 2
        offsets_bf1 = x + bar_width / 2
    else:
        offsets_dice = x

    dice_vals = [results[v]["Dice"] for v in EXPECTED_ORDER]
    bars_dice = ax.bar(offsets_dice, dice_vals, bar_width, label="Dice", color=DICE_COLOR, zorder=3)
    for bar in bars_dice:
        _annotate_bar(ax, bar, args.ymin)

    if has_bf1:
        bf1_vals = [results[v]["BF1"] for v in EXPECTED_ORDER]
        bars_bf1 = ax.bar(offsets_bf1, bf1_vals, bar_width, label="Boundary F1", color=BF1_COLOR, zorder=3)
        for bar in bars_bf1:
            _annotate_bar(ax, bar, args.ymin)

    # Reference line at full model Dice
    full_dice = results[FULL_MODEL_LABEL]["Dice"]
    ax.axhline(full_dice, color=DICE_COLOR, linestyle="--", linewidth=1.0, alpha=0.7, zorder=2)
    ax.text(
        n - 0.25,
        full_dice + 0.002,
        "Full Dice",
        color=DICE_COLOR,
        ha="right",
        va="bottom",
        fontsize=8.5,
    )

    if has_bf1:
        full_bf1 = results[FULL_MODEL_LABEL]["BF1"]
        ax.axhline(full_bf1, color=BF1_COLOR, linestyle="--", linewidth=1.0, alpha=0.7, zorder=2)
        ax.text(
            n - 0.25,
            full_bf1 - 0.004,
            "Full BF1",
            color=BF1_COLOR,
            ha="right",
            va="top",
            fontsize=8.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_LABELS, ha="center", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(bottom=args.ymin)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    output_path = os.path.abspath(args.output)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
