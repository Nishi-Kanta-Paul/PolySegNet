import csv
import os
import sys
from types import SimpleNamespace
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import build_model
from src.utils import count_parameters, ensure_dir, load_json


CHECK = "✓"
DASH = "–"


VARIANTS = [
    {
        "experiment": "effb4_unet_baseline",
        "label": "EffB4 U-Net",
        "cmsca": False,
        "bgsagf": False,
        "mbgh": False,
        "freq_aug": False,
        "config": {
            "model_name": "bgdsf_polysegnet",
            "use_msca": False,
            "use_csaf": False,
            "use_mbgh": False,
            "use_boundary_loss": False,
            "use_dynamic_weighting": False,
            "use_boundary_guidance": False,
            "use_multilevel_boundary": False,
        },
    },
    {
        "experiment": "effb4_unet_original_msca",
        "label": "EffB4 + Original MSCA",
        "cmsca": False,
        "bgsagf": False,
        "mbgh": False,
        "freq_aug": False,
        "config": {
            "model_name": "original_msca",
            "use_msca": True,
            "use_csaf": False,
            "use_mbgh": False,
            "use_boundary_loss": False,
            "use_dynamic_weighting": False,
            "use_boundary_guidance": False,
            "use_multilevel_boundary": False,
        },
    },
    {
        "experiment": "effb4_unet_cmsca_static",
        "label": "EffB4 + CMSCA (static)",
        "cmsca": True,
        "bgsagf": False,
        "mbgh": False,
        "freq_aug": False,
        "config": {
            "model_name": "bgdsf_polysegnet",
            "use_msca": True,
            "use_csaf": False,
            "use_mbgh": False,
            "use_boundary_loss": False,
            "use_dynamic_weighting": False,
            "use_boundary_guidance": False,
            "use_multilevel_boundary": False,
        },
    },
    {
        "experiment": "effb4_unet_bgd_cmsca",
        "label": "EffB4 + BGD-CMSCA",
        "cmsca": True,
        "bgsagf": False,
        "mbgh": False,
        "freq_aug": False,
        "config": {
            "model_name": "bgdsf_polysegnet",
            "use_msca": True,
            "use_csaf": False,
            "use_mbgh": False,
            "use_boundary_loss": False,
            "use_dynamic_weighting": True,
            "use_boundary_guidance": False,
            "use_multilevel_boundary": False,
        },
    },
    {
        "experiment": "bgdsf_cmsca_sagf_plain",
        "label": "BGD-CMSCA + SAGF (plain)",
        "cmsca": True,
        "bgsagf": False,
        "mbgh": False,
        "freq_aug": False,
        "config": {
            "model_name": "bgdsf_polysegnet",
            "use_msca": True,
            "use_csaf": True,
            "use_mbgh": False,
            "use_boundary_loss": False,
            "use_dynamic_weighting": True,
            "use_boundary_guidance": False,
            "use_multilevel_boundary": False,
        },
    },
    {
        "experiment": "bgdsf_cmsca_bgsagf",
        "label": "BGD-CMSCA + BG-SAGF",
        "cmsca": True,
        "bgsagf": True,
        "mbgh": False,
        "freq_aug": False,
        "config": {
            "model_name": "bgdsf_polysegnet",
            "use_msca": True,
            "use_csaf": True,
            "use_mbgh": False,
            "use_boundary_loss": False,
            "use_dynamic_weighting": True,
            "use_boundary_guidance": True,
            "use_multilevel_boundary": False,
        },
    },
    {
        "experiment": "bgdsf_polysegnet_no_freqaug",
        "label": "BGD-SF (no FreqAug)",
        "cmsca": True,
        "bgsagf": True,
        "mbgh": True,
        "freq_aug": False,
        "config": {
            "model_name": "bgdsf_polysegnet",
            "use_msca": True,
            "use_csaf": True,
            "use_mbgh": True,
            "use_boundary_loss": True,
            "use_dynamic_weighting": True,
            "use_boundary_guidance": True,
            "use_multilevel_boundary": True,
        },
    },
    {
        "experiment": "bgdsf_polysegnet_full",
        "label": "BGD-SF (FreqAug)",
        "cmsca": True,
        "bgsagf": True,
        "mbgh": True,
        "freq_aug": True,
        "config": {
            "model_name": "bgdsf_polysegnet",
            "use_msca": True,
            "use_csaf": True,
            "use_mbgh": True,
            "use_boundary_loss": True,
            "use_dynamic_weighting": True,
            "use_boundary_guidance": True,
            "use_multilevel_boundary": True,
        },
    },
]


def _extract_metrics(results: Dict[str, object]) -> Dict[str, float]:
    metrics = results.get("final_metrics", {}) if isinstance(results, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}

    return {
        "dice": float(metrics.get("val_dice", results.get("best_dice", 0.0))),
        "iou": float(metrics.get("val_iou", 0.0)),
        "precision": float(metrics.get("val_precision", 0.0)),
        "recall": float(metrics.get("val_recall", 0.0)),
        "mae": float(metrics.get("val_mae", 0.0)),
        "f_measure": float(
            metrics.get("val_fbeta", metrics.get("val_fmeasure", metrics.get("val_f1", 0.0)))
        ),
    }


def _load_eval_summary(exp_name: str) -> Dict[str, float]:
    eval_dir = os.path.join("experiments", exp_name, "evaluation")
    summary_path = os.path.join(eval_dir, "summary.json")
    results_path = os.path.join(eval_dir, "results.json")

    if os.path.isfile(summary_path):
        payload = load_json(summary_path)
        if isinstance(payload, dict):
            overall = payload.get("overall")
            if isinstance(overall, dict):
                return dict(overall)
            datasets = payload.get("datasets")
            if isinstance(datasets, dict) and datasets:
                first = next(iter(datasets.values()))
                if isinstance(first, dict):
                    return dict(first)

    if os.path.isfile(results_path):
        payload = load_json(results_path)
        summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        if isinstance(summary, dict):
            return dict(summary)

    return {}


def _pick_metric(
    eval_metrics: Dict[str, float],
    train_metrics: Dict[str, float],
    key: str,
) -> float:
    if isinstance(eval_metrics, dict) and key in eval_metrics:
        return float(eval_metrics.get(key, 0.0))
    return float(train_metrics.get(key, 0.0))


def _format_flag(enabled: bool) -> str:
    return CHECK if enabled else DASH


def _count_params(config_overrides: Dict[str, object]) -> float:
    config = SimpleNamespace(
        model_name=config_overrides.get("model_name", "bgdsf_polysegnet"),
        pretrained=False,
        unified_channels=128,
        use_msca=bool(config_overrides.get("use_msca", True)),
        use_csaf=bool(config_overrides.get("use_csaf", True)),
        use_mbgh=bool(config_overrides.get("use_mbgh", True)),
        use_boundary_loss=bool(config_overrides.get("use_boundary_loss", False)),
        use_dynamic_weighting=bool(config_overrides.get("use_dynamic_weighting", True)),
        use_boundary_guidance=bool(config_overrides.get("use_boundary_guidance", True)),
        use_multilevel_boundary=bool(config_overrides.get("use_multilevel_boundary", True)),
        debug=False,
    )
    model = build_model(config)
    params = count_parameters(model) / 1e6
    return params


def compare_results() -> None:
    rows: List[List[str]] = []
    for variant in VARIANTS:
        exp_name = variant["experiment"]
        results_path = os.path.join("experiments", exp_name, "results.json")
        if not os.path.isfile(results_path):
            print(f"Warning: missing results for {exp_name}")
            continue
        train_metrics = _extract_metrics(load_json(results_path))
        eval_metrics = _load_eval_summary(exp_name)
        if not eval_metrics:
            print(f"Warning: missing evaluation results for {exp_name}")
        params_m = _count_params(variant["config"])
        rows.append(
            [
                variant["label"],
                _format_flag(variant["cmsca"]),
                _format_flag(variant["bgsagf"]),
                _format_flag(variant["mbgh"]),
                _format_flag(variant["freq_aug"]),
                f"{_pick_metric(eval_metrics, train_metrics, 'dice'):.4f}",
                f"{_pick_metric(eval_metrics, train_metrics, 'iou'):.4f}",
                f"{_pick_metric(eval_metrics, train_metrics, 'precision'):.4f}",
                f"{_pick_metric(eval_metrics, train_metrics, 'recall'):.4f}",
                f"{_pick_metric(eval_metrics, train_metrics, 'mae'):.4f}",
                f"{_pick_metric(eval_metrics, train_metrics, 'f_measure'):.4f}",
                f"{_pick_metric(eval_metrics, train_metrics, 'boundary_f1'):.4f}",
                f"{_pick_metric(eval_metrics, train_metrics, 'hausdorff'):.4f}",
                f"{_pick_metric(eval_metrics, train_metrics, 'hd95'):.4f}",
                f"{_pick_metric(eval_metrics, train_metrics, 'asd'):.4f}",
                f"{_pick_metric(eval_metrics, train_metrics, 'assd'):.4f}",
                f"{params_m:.2f}",
            ]
        )

    output_dir = os.path.join("outputs", "tables")
    ensure_dir(output_dir)

    csv_path = os.path.join(output_dir, "comparison_results.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Model",
                "CMSCA",
                "BG-SAGF",
                "MBGH",
                "FreqAug",
                "Dice",
                "IoU",
                "Precision",
                "Recall",
                "MAE",
                "F-measure",
                "Boundary F1",
                "Hausdorff",
                "HD95",
                "ASD",
                "ASSD",
                "Params(M)",
            ]
        )
        writer.writerows(rows)

    md_path = os.path.join(output_dir, "comparison_results.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(
            "| Model | CMSCA | BG-SAGF | MBGH | FreqAug | Dice | IoU | Precision | Recall | MAE | F-measure | Boundary F1 | Hausdorff | HD95 | ASD | ASSD | Params(M) |\n"
        )
        handle.write(
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        )
        for row in rows:
            handle.write("| " + " | ".join(row) + " |\n")

    print(f"Saved comparison tables to {output_dir}")


if __name__ == "__main__":
    compare_results()
