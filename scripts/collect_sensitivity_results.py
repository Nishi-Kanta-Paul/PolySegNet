#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import torch

from src.model import build_model
from src.utils import count_parameters, ensure_dir, load_json


BOUNDARY_WEIGHT_EXPERIMENTS = {
    "bgdsf_bw_0p1": "0.1",
    "bgdsf_bw_0p3": "0.3",
    "bgdsf_bw_0p5": "0.5",
    "bgdsf_bw_1p0": "1.0",
}

INPUT_SIZE_EXPERIMENTS = {
    "bgdsf_size_256": "256",
    "bgdsf_size_352": "352",
    "bgdsf_size_512": "512",
}

BACKBONE_EXPERIMENTS = {
    "bgdsf_backbone_effb0": "efficientnet_b0",
    "bgdsf_backbone_effb4": "efficientnet_b4",
    "bgdsf_backbone_resnet50": "resnet50",
    "bgdsf_backbone_mobilenetv3": "mobilenetv3_large",
    "bgdsf_backbone_convnext_tiny": "convnext_tiny",
}

FULL_FIELDS = [
    "experiment_name",
    "sensitivity_type",
    "setting_value",
    "dataset_name",
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
    "params_m",
    "flops_g",
    "fps",
    "ms_per_image",
]

PAPER_FIELDS = [
    "dataset_name",
    "setting_value",
    "dice",
    "iou",
    "mae",
    "mask_boundary_f1",
    "mask_hd95",
    "mask_assd",
    "params_m",
    "flops_g",
    "fps",
]


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


def _load_experiment_results(
    exp_name: str,
    dataset_names_hint: List[str],
) -> Dict[str, Dict[str, float]]:
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


def _load_complexity_index(path: str) -> Dict[str, Dict[str, object]]:
    if not path or not os.path.isfile(path):
        return {}
    payload = load_json(path)
    results = payload.get("results", []) if isinstance(payload, dict) else []
    index: Dict[str, Dict[str, object]] = {}
    if isinstance(results, list):
        for row in results:
            if not isinstance(row, dict):
                continue
            key = row.get("method_name") or row.get("experiment_name")
            if key:
                index[str(key)] = row
    return index


def _format_number(value: float | None, decimals: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{decimals}f}"


def _format_params(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def _collect_rows(
    experiments: Dict[str, str],
    sensitivity_type: str,
    dataset_names_hint: List[str],
    complexity_index: Dict[str, Dict[str, object]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    full_rows: List[Dict[str, str]] = []
    paper_rows: List[Dict[str, str]] = []

    for exp_name, setting in experiments.items():
        inferred_names = dataset_names_hint or _infer_dataset_names_from_checkpoint(exp_name)
        per_dataset = _load_experiment_results(exp_name, inferred_names)
        params_m = _try_params_m(exp_name)
        complexity = complexity_index.get(exp_name, {})

        flops_g = complexity.get("flops_g")
        fps = complexity.get("fps")
        ms_per_image = complexity.get("ms_per_image")
        params_from_complexity = complexity.get("params_m")

        if params_from_complexity is not None:
            params_m = float(params_from_complexity)

        if not per_dataset:
            print(
                f"WARNING: Evaluation output missing for {exp_name}. "
                "Please run evaluation using best.pth. "
                "Metrics will be empty — no fallback to training metrics."
            )
            dataset_list = inferred_names or ["dataset"]
            for dataset_name in dataset_list:
                full_rows.append(
                    {
                        "experiment_name": exp_name,
                        "sensitivity_type": sensitivity_type,
                        "setting_value": setting,
                        "dataset_name": dataset_name,
                        "dice": "",
                        "iou": "",
                        "precision": "",
                        "recall": "",
                        "f_measure": "",
                        "mae": "",
                        "mask_boundary_f1": "",
                        "mask_hd": "",
                        "mask_hd95": "",
                        "mask_asd": "",
                        "mask_assd": "",
                        "params_m": _format_params(params_m),
                        "flops_g": _format_number(
                            float(flops_g), 4
                        ) if flops_g is not None else "",
                        "fps": _format_number(float(fps), 2) if fps is not None else "",
                        "ms_per_image": _format_number(
                            float(ms_per_image), 3
                        ) if ms_per_image is not None else "",
                    }
                )
                paper_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "setting_value": setting,
                        "dice": "",
                        "iou": "",
                        "mae": "",
                        "mask_boundary_f1": "",
                        "mask_hd95": "",
                        "mask_assd": "",
                        "params_m": _format_params(params_m),
                        "flops_g": _format_number(
                            float(flops_g), 4
                        ) if flops_g is not None else "",
                        "fps": _format_number(float(fps), 2) if fps is not None else "",
                    }
                )
            continue

        for dataset_name, metrics in per_dataset.items():
            full_rows.append(
                {
                    "experiment_name": exp_name,
                    "sensitivity_type": sensitivity_type,
                    "setting_value": setting,
                    "dataset_name": dataset_name,
                    "dice": _format_number(metrics.get("dice")),
                    "iou": _format_number(metrics.get("iou")),
                    "precision": _format_number(metrics.get("precision")),
                    "recall": _format_number(metrics.get("recall")),
                    "f_measure": _format_number(metrics.get("f_measure")),
                    "mae": _format_number(metrics.get("mae")),
                    "mask_boundary_f1": _format_number(metrics.get("mask_boundary_f1")),
                    "mask_hd": _format_number(metrics.get("mask_hd")),
                    "mask_hd95": _format_number(metrics.get("mask_hd95")),
                    "mask_asd": _format_number(metrics.get("mask_asd")),
                    "mask_assd": _format_number(metrics.get("mask_assd")),
                    "params_m": _format_params(params_m),
                    "flops_g": _format_number(
                        float(flops_g), 4
                    ) if flops_g is not None else "",
                    "fps": _format_number(float(fps), 2) if fps is not None else "",
                    "ms_per_image": _format_number(
                        float(ms_per_image), 3
                    ) if ms_per_image is not None else "",
                }
            )

            paper_rows.append(
                {
                    "dataset_name": dataset_name,
                    "setting_value": setting,
                    "dice": _format_number(metrics.get("dice")),
                    "iou": _format_number(metrics.get("iou")),
                    "mae": _format_number(metrics.get("mae")),
                    "mask_boundary_f1": _format_number(metrics.get("mask_boundary_f1")),
                    "mask_hd95": _format_number(metrics.get("mask_hd95")),
                    "mask_assd": _format_number(metrics.get("mask_assd")),
                    "params_m": _format_params(params_m),
                    "flops_g": _format_number(
                        float(flops_g), 4
                    ) if flops_g is not None else "",
                    "fps": _format_number(float(fps), 2) if fps is not None else "",
                }
            )

    return full_rows, paper_rows


def _write_csv(path: str, fieldnames: Iterable[str], rows: List[Dict[str, str]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect sensitivity study results into CSV tables."
    )
    parser.add_argument(
        "--dataset-roots",
        nargs="+",
        default=[],
        help="Optional dataset roots for labeling single-dataset runs.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/tables",
        help="Output directory for sensitivity tables.",
    )
    parser.add_argument(
        "--complexity-json",
        default="outputs/tables/model_complexity.json",
        help="Optional complexity JSON file to enrich params/FLOPs/FPS.",
    )
    args = parser.parse_args()

    dataset_names_hint = [
        _dataset_name_from_path(root) for root in args.dataset_roots if root
    ]
    complexity_index = _load_complexity_index(args.complexity_json)

    outputs = {
        "boundary_weight": (
            BOUNDARY_WEIGHT_EXPERIMENTS,
            os.path.join(args.output_dir, "sensitivity_boundary_weight.csv"),
            os.path.join(args.output_dir, "sensitivity_boundary_weight_paper.csv"),
        ),
        "input_size": (
            INPUT_SIZE_EXPERIMENTS,
            os.path.join(args.output_dir, "sensitivity_input_size.csv"),
            os.path.join(args.output_dir, "sensitivity_input_size_paper.csv"),
        ),
        "backbone": (
            BACKBONE_EXPERIMENTS,
            os.path.join(args.output_dir, "sensitivity_backbone.csv"),
            os.path.join(args.output_dir, "sensitivity_backbone_paper.csv"),
        ),
    }

    for sensitivity_type, (experiments, full_path, paper_path) in outputs.items():
        full_rows, paper_rows = _collect_rows(
            experiments,
            sensitivity_type,
            dataset_names_hint,
            complexity_index,
        )
        _write_csv(full_path, FULL_FIELDS, full_rows)
        _write_csv(paper_path, PAPER_FIELDS, paper_rows)
        print(f"Saved {sensitivity_type} table to {full_path}")
        print(f"Saved {sensitivity_type} paper table to {paper_path}")


if __name__ == "__main__":
    main()
