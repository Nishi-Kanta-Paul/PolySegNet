#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from types import SimpleNamespace
from typing import Dict, List

import torch

from src.model import build_model
from src.utils import count_parameters, ensure_dir, load_json


METHODS = {
    "U-Net": {
        "experiment": "baseline_smp_unet",
    },
    "U-Net++": {
        "experiment": "baseline_smp_unetpp",
    },
    "DeepLabV3+": {
        "experiment": "baseline_smp_deeplabv3plus",
    },
    "FPN": {
        "experiment": "baseline_smp_fpn",
    },
    "PSPNet": {
        "experiment": "baseline_smp_pspnet",
    },
    "LinkNet": {
        "experiment": "baseline_smp_linknet",
    },
    "BGD-SF PolySegNet": {
        "experiment": "bgdsf_polysegnet_full",
    },
}


def _dataset_name_from_path(path: str) -> str:
    return os.path.basename(os.path.normpath(path)) or "dataset"


def _infer_dataset_names_from_checkpoint(exp_name: str) -> List[str]:
    ckpt_dir = os.path.join("experiments", exp_name, "checkpoints")
    for name in ("best.pth", "latest.pth", "last.pth"):
        path = os.path.join(ckpt_dir, name)
        if os.path.isfile(path):
            checkpoint = torch.load(path, map_location="cpu")
            cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
            roots = cfg.get("dataset_roots") or []
            if isinstance(roots, list) and roots:
                return [_dataset_name_from_path(root) for root in roots]
            root = cfg.get("dataset_root") or ""
            if root:
                return [_dataset_name_from_path(root)]
            break
    return ["dataset"]


def _load_experiment_results(exp_name: str, dataset_names_hint: List[str]) -> Dict[str, Dict[str, float]]:
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
        dataset_names = dataset_names_hint or ["dataset"]
        return {dataset_names[0]: dict(summary)}

    return {}


def _load_checkpoint_config(exp_name: str) -> SimpleNamespace | None:
    ckpt_dir = os.path.join("experiments", exp_name, "checkpoints")
    for name in ("best.pth", "latest.pth", "last.pth"):
        path = os.path.join(ckpt_dir, name)
        if os.path.isfile(path):
            checkpoint = torch.load(path, map_location="cpu")
            cfg = checkpoint.get("config") if isinstance(checkpoint, dict) else None
            if isinstance(cfg, dict):
                return SimpleNamespace(**cfg)
    return None


def _try_params_m(exp_name: str) -> float | None:
    cfg = _load_checkpoint_config(exp_name)
    if cfg is None:
        return None
    try:
        model = build_model(cfg)
        return count_parameters(model) / 1e6
    except Exception:
        return None


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _format_params(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect easy baseline results.")
    parser.add_argument(
        "--dataset-roots",
        nargs="+",
        default=[],
        help="Optional dataset roots for labeling single-dataset runs.",
    )
    parser.add_argument("--output", default="outputs/tables/easy_baseline_comparison.csv")
    args = parser.parse_args()

    dataset_names_hint = [
        _dataset_name_from_path(root) for root in args.dataset_roots if root
    ]

    rows: List[Dict[str, str]] = []
    pending: List[str] = []

    for method_name, spec in METHODS.items():
        exp_name = spec["experiment"]
        inferred_names = dataset_names_hint or _infer_dataset_names_from_checkpoint(exp_name)
        per_dataset = _load_experiment_results(exp_name, inferred_names)
        params_m = _try_params_m(exp_name)

        if not per_dataset:
            print(
                "Evaluation output missing for "
                f"{exp_name}. Please run evaluation using best.pth."
            )
            pending.append(method_name)
            dataset_list = dataset_names_hint or ["dataset"]
            for dataset_name in dataset_list:
                rows.append(
                    {
                        "method_name": method_name,
                        "dataset_name": dataset_name,
                        "dice": "",
                        "iou": "",
                        "precision": "",
                        "recall": "",
                        "f_measure": "",
                        "mae": "",
                        "boundary_f1": "",
                        "hausdorff": "",
                        "hd95": "",
                        "asd": "",
                        "assd": "",
                        "params_m": _format_params(params_m),
                        "flops_g": "",
                        "fps": "",
                    }
                )
            continue

        for dataset_name, metrics in per_dataset.items():
            rows.append(
                {
                    "method_name": method_name,
                    "dataset_name": dataset_name,
                    "dice": _format_number(metrics.get("dice")),
                    "iou": _format_number(metrics.get("iou")),
                    "precision": _format_number(metrics.get("precision")),
                    "recall": _format_number(metrics.get("recall")),
                    "f_measure": _format_number(metrics.get("f_measure")),
                    "mae": _format_number(metrics.get("mae")),
                    "boundary_f1": _format_number(metrics.get("boundary_f1")),
                    "hausdorff": _format_number(metrics.get("hausdorff")),
                    "hd95": _format_number(metrics.get("hd95")),
                    "asd": _format_number(metrics.get("asd")),
                    "assd": _format_number(metrics.get("assd")),
                    "params_m": _format_params(params_m),
                    "flops_g": "",
                    "fps": "",
                }
            )

    output_path = args.output
    ensure_dir(os.path.dirname(output_path))
    fieldnames = [
        "method_name",
        "dataset_name",
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
        "params_m",
        "flops_g",
        "fps",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if pending:
        print("Pending results:")
        for name in pending:
            print(f"- {name}")
    print(f"Saved comparison CSV to {output_path}")


if __name__ == "__main__":
    main()
