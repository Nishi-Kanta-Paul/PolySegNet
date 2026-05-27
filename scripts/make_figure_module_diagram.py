#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.utils import ensure_dir


def _box(ax, xy, text, width=2.4, height=0.5, color="#f8fafc"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.0,
        edgecolor="#0f172a",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=8,
    )


def _stack(ax, x, y, title, items, color):
    ax.text(x, y + 2.1, title, fontsize=9, weight="bold")
    for idx, item in enumerate(items):
        _box(ax, (x, y + 1.5 - idx * 0.55), item, color=color)


def main() -> None:
    parser = argparse.ArgumentParser(description="Module diagram (placeholder).")
    parser.add_argument("--output", default="paper_figures/fig_modules.png")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    _stack(
        ax,
        0.6,
        0.2,
        "BGD-CMSCA",
        [
            "1x1 Conv (Reduce)",
            "Dilated Conv x4",
            "Boundary Prior",
            "Dynamic Scale Weights",
            "Fuse + Residual",
        ],
        color="#e0f2fe",
    )

    _stack(
        ax,
        5.2,
        0.2,
        "BG-SAGF",
        [
            "Upsample Decoder",
            "Boundary-guided Gate",
            "Channel Attention",
            "Spatial Attention",
            "Fuse Skip + Decoder",
        ],
        color="#fef3c7",
    )

    output_path = os.path.abspath(args.output)
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved module diagram to {output_path}")


if __name__ == "__main__":
    main()
