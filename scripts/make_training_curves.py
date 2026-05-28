#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from src.utils import ensure_dir


@dataclass
class ExperimentLog:
    name: str
    log_path: str
    entries: List[Dict[str, float]]
    available_keys: set


def _default_log_path(experiment: str) -> str:
    return os.path.join("experiments", experiment, "logs", "train_log.json")


def _load_entries(path: str) -> List[Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]

    if isinstance(data, dict):
        if isinstance(data.get("history"), list):
            return [entry for entry in data["history"] if isinstance(entry, dict)]

        list_keys = [key for key, value in data.items() if isinstance(value, list)]
        if list_keys:
            length = max(len(data[key]) for key in list_keys)
            entries: List[Dict[str, float]] = []
            for idx in range(length):
                entry: Dict[str, float] = {}
                for key in list_keys:
                    values = data[key]
                    if idx < len(values):
                        entry[key] = values[idx]
                entries.append(entry)
            return entries

    raise ValueError(f"Unsupported log format: {path}")


def _collect_keys(entries: Sequence[Dict[str, float]]) -> set:
    keys: set = set()
    for entry in entries:
        keys.update(entry.keys())
    return keys


def _series(entries: Sequence[Dict[str, float]], key: str) -> Tuple[List[int], List[float]]:
    epochs: List[int] = []
    values: List[float] = []
    for idx, entry in enumerate(entries):
        if key in entry:
            epoch_value = entry.get("epoch", idx + 1)
            epochs.append(int(epoch_value))
            values.append(float(entry[key]))
    return epochs, values


def _first_available_series(
    entries: Sequence[Dict[str, float]],
    keys: Iterable[str],
) -> Tuple[Optional[str], List[int], List[float]]:
    for key in keys:
        epochs, values = _series(entries, key)
        if values:
            return key, epochs, values
    return None, [], []


def _configure_plotting() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _apply_axis_style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)


def _save_figure(
    fig,
    output_dir: str,
    name: str,
    fmt: str,
    dpi: int,
    also_pdf: bool,
) -> List[str]:
    ensure_dir(output_dir)
    saved_paths: List[str] = []
    main_path = os.path.join(output_dir, f"{name}.{fmt}")
    fig.savefig(main_path, dpi=dpi, bbox_inches="tight")
    saved_paths.append(main_path)
    if also_pdf and fmt.lower() != "pdf":
        pdf_path = os.path.join(output_dir, f"{name}.pdf")
        fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
        saved_paths.append(pdf_path)
    return saved_paths


def _warn_missing(metric: str, experiment: str) -> None:
    print(f"Warning: '{metric}' not found for experiment '{experiment}'.")


def _load_experiment_logs(
    experiments: Sequence[str],
    log_path_override: Optional[str],
) -> List[ExperimentLog]:
    logs: List[ExperimentLog] = []
    for exp in experiments:
        log_path = log_path_override or _default_log_path(exp)
        if not os.path.isfile(log_path):
            raise FileNotFoundError(f"Log file not found: {log_path}")
        entries = _load_entries(log_path)
        logs.append(
            ExperimentLog(
                name=exp,
                log_path=log_path,
                entries=entries,
                available_keys=_collect_keys(entries),
            )
        )
    return logs


def _prefix_from_experiment(experiment: str) -> str:
    if experiment == "bgdsf_polysegnet_full":
        return "bgdsf"
    if "bgdsf" in experiment:
        return "bgdsf"
    return experiment


def _plot_single_experiment(
    exp_log: ExperimentLog,
    output_dir: str,
    fmt: str,
    dpi: int,
    also_pdf: bool,
) -> List[str]:
    saved: List[str] = []
    prefix = _prefix_from_experiment(exp_log.name)

    _configure_plotting()

    train_epochs, train_loss = _series(exp_log.entries, "train_loss")
    val_epochs, val_loss = _series(exp_log.entries, "val_loss")
    if not train_loss:
        _warn_missing("train_loss", exp_log.name)
    if not val_loss:
        _warn_missing("val_loss", exp_log.name)

    if train_loss or val_loss:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        if train_loss:
            ax.plot(train_epochs, train_loss, label="Train loss", linewidth=2)
        if val_loss:
            ax.plot(val_epochs, val_loss, label="Val loss", linewidth=2)
        _apply_axis_style(ax, "Loss Curves", "Loss")
        if train_loss or val_loss:
            ax.legend()
        fig.tight_layout()
        saved += _save_figure(fig, output_dir, f"{prefix}_training_loss", fmt, dpi, also_pdf)
        plt.close(fig)

    dice_epochs, val_dice = _series(exp_log.entries, "val_dice")
    iou_epochs, val_iou = _series(exp_log.entries, "val_iou")
    if not val_dice:
        _warn_missing("val_dice", exp_log.name)
    if not val_iou:
        _warn_missing("val_iou", exp_log.name)

    if val_dice or val_iou:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        if val_dice:
            ax.plot(dice_epochs, val_dice, label="Val Dice", linewidth=2)
        if val_iou:
            ax.plot(iou_epochs, val_iou, label="Val IoU", linewidth=2)
        _apply_axis_style(ax, "Validation Metrics", "Metric")
        ax.legend()
        fig.tight_layout()
        saved += _save_figure(
            fig, output_dir, f"{prefix}_validation_metrics", fmt, dpi, also_pdf
        )
        plt.close(fig)

    boundary_epochs, boundary_loss = _series(exp_log.entries, "boundary_loss")
    aux_epochs, aux_boundary_loss = _series(exp_log.entries, "aux_boundary_loss")
    if not boundary_loss:
        _warn_missing("boundary_loss", exp_log.name)
    if not aux_boundary_loss:
        _warn_missing("aux_boundary_loss", exp_log.name)

    if boundary_loss or aux_boundary_loss:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        if boundary_loss:
            ax.plot(boundary_epochs, boundary_loss, label="Boundary loss", linewidth=2)
        if aux_boundary_loss:
            ax.plot(aux_epochs, aux_boundary_loss, label="Aux boundary loss", linewidth=2)
        _apply_axis_style(ax, "Boundary Loss", "Loss")
        ax.legend()
        fig.tight_layout()
        saved += _save_figure(fig, output_dir, f"{prefix}_boundary_loss", fmt, dpi, also_pdf)
        plt.close(fig)

    lr_key, lr_epochs, lr_values = _first_available_series(
        exp_log.entries, ("learning_rate", "lr")
    )
    if not lr_values:
        _warn_missing("learning_rate", exp_log.name)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    axes = axes.flatten()

    plotted_any = False

    if train_loss or val_loss:
        ax = axes[0]
        if train_loss:
            ax.plot(train_epochs, train_loss, label="Train loss", linewidth=2)
        if val_loss:
            ax.plot(val_epochs, val_loss, label="Val loss", linewidth=2)
        _apply_axis_style(ax, "Loss", "Loss")
        ax.legend()
        plotted_any = True
    else:
        axes[0].text(0.5, 0.5, "No loss data", ha="center", va="center")
        axes[0].set_axis_off()

    if val_dice or val_iou:
        ax = axes[1]
        if val_dice:
            ax.plot(dice_epochs, val_dice, label="Val Dice", linewidth=2)
        if val_iou:
            ax.plot(iou_epochs, val_iou, label="Val IoU", linewidth=2)
        _apply_axis_style(ax, "Validation Metrics", "Metric")
        ax.legend()
        plotted_any = True
    else:
        axes[1].text(0.5, 0.5, "No validation metrics", ha="center", va="center")
        axes[1].set_axis_off()

    if boundary_loss or aux_boundary_loss:
        ax = axes[2]
        if boundary_loss:
            ax.plot(boundary_epochs, boundary_loss, label="Boundary loss", linewidth=2)
        if aux_boundary_loss:
            ax.plot(aux_epochs, aux_boundary_loss, label="Aux boundary loss", linewidth=2)
        _apply_axis_style(ax, "Boundary Loss", "Loss")
        ax.legend()
        plotted_any = True
    else:
        axes[2].text(0.5, 0.5, "No boundary loss", ha="center", va="center")
        axes[2].set_axis_off()

    if lr_values:
        ax = axes[3]
        ax.plot(lr_epochs, lr_values, label="Learning rate", linewidth=2)
        _apply_axis_style(ax, "Learning Rate", "LR")
        ax.legend()
        plotted_any = True
    else:
        axes[3].text(0.5, 0.5, "No learning rate", ha="center", va="center")
        axes[3].set_axis_off()

    fig.tight_layout()
    if plotted_any:
        saved += _save_figure(
            fig, output_dir, f"{prefix}_full_training_curves", fmt, dpi, also_pdf
        )
    plt.close(fig)

    return saved


def _plot_comparisons(
    experiment_logs: Sequence[ExperimentLog],
    output_dir: str,
    fmt: str,
    dpi: int,
    also_pdf: bool,
) -> List[str]:
    saved: List[str] = []
    _configure_plotting()

    compare_metrics = {
        "comparison_val_dice": "val_dice",
        "comparison_val_loss": "val_loss",
    }

    for name, metric in compare_metrics.items():
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        plotted = False
        for exp_log in experiment_logs:
            epochs, values = _series(exp_log.entries, metric)
            if not values:
                _warn_missing(metric, exp_log.name)
                continue
            ax.plot(epochs, values, label=exp_log.name, linewidth=2)
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        _apply_axis_style(ax, metric.replace("_", " ").title(), "Metric")
        ax.legend()
        fig.tight_layout()
        saved += _save_figure(fig, output_dir, name, fmt, dpi, also_pdf)
        plt.close(fig)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training curve figures.")
    parser.add_argument("--experiment", default="bgdsf_polysegnet_full")
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--output-dir", default="outputs/figures/training_curves")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--format", default="png")
    parser.add_argument("--also-pdf", action="store_true")
    args = parser.parse_args()

    if args.experiments:
        experiments = args.experiments
        if args.log_path:
            print("Warning: --log-path is ignored when using --experiments.")
        experiment_logs = _load_experiment_logs(experiments, None)
        _plot_comparisons(experiment_logs, args.output_dir, args.format, args.dpi, args.also_pdf)
    else:
        experiments = [args.experiment]
        experiment_logs = _load_experiment_logs(experiments, args.log_path)

    saved_paths: List[str] = []
    for exp_log in experiment_logs:
        saved_paths.extend(
            _plot_single_experiment(exp_log, args.output_dir, args.format, args.dpi, args.also_pdf)
        )

    if saved_paths:
        print("Saved training curve figures:")
        for path in saved_paths:
            print(f"- {path}")


if __name__ == "__main__":
    main()
