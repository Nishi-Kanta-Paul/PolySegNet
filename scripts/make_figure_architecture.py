#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.utils import ensure_dir


def _box(ax, xy, text, width=1.6, height=0.55, color="#e0f2fe"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.2,
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
        fontsize=9,
    )
    return box


def _arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", linewidth=1.2, color="#0f172a"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Architecture diagram (placeholder).")
    parser.add_argument("--output", default="paper_figures/fig_architecture.png")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    encoder = _box(ax, (0.4, 1.2), "EfficientNet-B4\nEncoder", width=1.8)
    proj = _box(ax, (2.6, 1.2), "1x1\nProjection", width=1.4, color="#fef3c7")
    cmsca = _box(ax, (4.4, 1.2), "BGD-CMSCA\nBottleneck", width=1.7)
    decoder = _box(ax, (6.5, 1.2), "Decoder\nBG-SAGF", width=1.6, color="#e9d5ff")
    mbgh = _box(ax, (8.4, 1.2), "MBGH\nBoundary Head", width=1.3, color="#dcfce7")

    _arrow(ax, (2.2, 1.5), (2.6, 1.5))
    _arrow(ax, (4.0, 1.5), (4.4, 1.5))
    _arrow(ax, (6.1, 1.5), (6.5, 1.5))
    _arrow(ax, (8.1, 1.5), (8.4, 1.5))

    ax.text(6.6, 2.35, "Skip Fusion", fontsize=8, color="#7c3aed")
    ax.text(8.35, 2.35, "Mask + Boundary", fontsize=8, color="#14532d")

    output_path = os.path.abspath(args.output)
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved architecture diagram to {output_path}")


if __name__ == "__main__":
    main()
