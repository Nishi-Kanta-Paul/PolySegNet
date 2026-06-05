#!/usr/bin/env python3
"""Generate corrected ablation tables using mask-derived boundary metrics.

Boundary metrics in these tables are always derived from the final predicted
mask, ensuring a fair comparison across all model variants regardless of
whether the variant includes the MBGH boundary prediction head.

Outputs:
  outputs/tables/ablation_corrected_mask_boundary.csv       Full table
  outputs/tables/ablation_corrected_mask_boundary_paper.csv Paper-ready table
  outputs/tables/ablation_boundary_diagnostic_report.md     Diagnostic report
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import build_model
from src.utils import count_parameters, ensure_dir, load_json


CHECK = "✓"
DASH = "–"

# Must match baselines/compare_results.py VARIANTS labels exactly
VARIANTS: List[Dict] = [
    {
        "experiment": "effb4_unet_baseline",
        "label": "EffB4 encoder-decoder baseline",
        "cmsca": False, "bgsagf": False, "mbgh": False, "freq_aug": False,
        "config": {
            "model_name": "bgdsf_polysegnet",
            "use_msca": False, "use_csaf": False, "use_mbgh": False,
            "use_boundary_loss": False, "use_dynamic_weighting": False,
            "use_boundary_guidance": False, "use_multilevel_boundary": False,
        },
    },
    {
        "experiment": "effb4_unet_original_msca",
        "label": "EffB4 + Original MSCA",
        "cmsca": False, "bgsagf": False, "mbgh": False, "freq_aug": False,
        "config": {
            "model_name": "original_msca",
            "use_msca": True, "use_csaf": False, "use_mbgh": False,
            "use_boundary_loss": False, "use_dynamic_weighting": False,
            "use_boundary_guidance": False, "use_multilevel_boundary": False,
        },
    },
    {
        "experiment": "effb4_unet_cmsca_static",
        "label": "EffB4 + CMSCA (static)",
        "cmsca": True, "bgsagf": False, "mbgh": False, "freq_aug": False,
        "config": {
            "model_name": "bgdsf_polysegnet",
            "use_msca": True, "use_csaf": False, "use_mbgh": False,
            "use_boundary_loss": False, "use_dynamic_weighting": False,
            "use_boundary_guidance": False, "use_multilevel_boundary": False,
        },
    },
    {
        "experiment": "effb4_unet_bgd_cmsca",
        "label": "EffB4 + BGD-CMSCA",
        "cmsca": True, "bgsagf": False, "mbgh": False, "freq_aug": False,
        "config": {
            "model_name": "bgdsf_polysegnet",
            "use_msca": True, "use_csaf": False, "use_mbgh": False,
            "use_boundary_loss": False, "use_dynamic_weighting": True,
            "use_boundary_guidance": False, "use_multilevel_boundary": False,
        },
    },
    {
        "experiment": "bgdsf_cmsca_sagf_plain",
        "label": "BGD-CMSCA + SAGF (plain)",
        "cmsca": True, "bgsagf": False, "mbgh": False, "freq_aug": False,
        "config": {
            "model_name": "bgdsf_polysegnet",
            "use_msca": True, "use_csaf": True, "use_mbgh": False,
            "use_boundary_loss": False, "use_dynamic_weighting": True,
            "use_boundary_guidance": False, "use_multilevel_boundary": False,
        },
    },
    {
        "experiment": "bgdsf_cmsca_bgsagf",
        "label": "BGD-CMSCA + BG-SAGF",
        "cmsca": True, "bgsagf": True, "mbgh": False, "freq_aug": False,
        "config": {
            "model_name": "bgdsf_polysegnet",
            "use_msca": True, "use_csaf": True, "use_mbgh": False,
            "use_boundary_loss": False, "use_dynamic_weighting": True,
            "use_boundary_guidance": True, "use_multilevel_boundary": False,
        },
    },
    {
        "experiment": "bgdsf_polysegnet_no_freqaug",
        "label": "BGD-SF (no FreqAug)",
        "cmsca": True, "bgsagf": True, "mbgh": True, "freq_aug": False,
        "config": {
            "model_name": "bgdsf_polysegnet",
            "use_msca": True, "use_csaf": True, "use_mbgh": True,
            "use_boundary_loss": True, "use_dynamic_weighting": True,
            "use_boundary_guidance": True, "use_multilevel_boundary": True,
        },
    },
    {
        "experiment": "bgdsf_polysegnet_full",
        "label": "BGD-SF (FreqAug)",
        "cmsca": True, "bgsagf": True, "mbgh": True, "freq_aug": True,
        "config": {
            "model_name": "bgdsf_polysegnet",
            "use_msca": True, "use_csaf": True, "use_mbgh": True,
            "use_boundary_loss": True, "use_dynamic_weighting": True,
            "use_boundary_guidance": True, "use_multilevel_boundary": True,
        },
    },
]

FULL_COLUMNS = [
    "Model", "CMSCA", "BG-SAGF", "MBGH", "FreqAug",
    "Dice", "IoU", "Precision", "Recall", "MAE", "F-measure",
    "mask_boundary_f1", "mask_hd", "mask_hd95", "mask_asd", "mask_assd",
    "Params(M)",
]

PAPER_COLUMNS = [
    "Model", "Dice", "IoU", "MAE",
    "mask_boundary_f1", "mask_hd95", "mask_assd",
]

BASELINE_LABEL = "EffB4 encoder-decoder baseline"
FULL_LABEL = "BGD-SF (FreqAug)"
NO_FREQAUG_LABEL = "BGD-SF (no FreqAug)"


def _load_eval_summary(exp_name: str) -> Optional[Dict[str, float]]:
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

    return None


def _count_params(config_overrides: Dict) -> Optional[float]:
    cfg = SimpleNamespace(
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
    try:
        model = build_model(cfg)
        return count_parameters(model) / 1e6
    except Exception:
        return None


def _fmt(value: Optional[float], decimals: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):.{decimals}f}"


def _get(metrics: Optional[Dict[str, float]], key: str) -> Optional[float]:
    if not metrics:
        return None
    val = metrics.get(key)
    return float(val) if val is not None else None


def _build_full_row(
    variant: Dict,
    metrics: Optional[Dict[str, float]],
    params_m: Optional[float],
) -> Dict[str, str]:
    flag = lambda b: CHECK if b else DASH
    return {
        "Model": variant["label"],
        "CMSCA": flag(variant["cmsca"]),
        "BG-SAGF": flag(variant["bgsagf"]),
        "MBGH": flag(variant["mbgh"]),
        "FreqAug": flag(variant["freq_aug"]),
        "Dice": _fmt(_get(metrics, "dice")),
        "IoU": _fmt(_get(metrics, "iou")),
        "Precision": _fmt(_get(metrics, "precision")),
        "Recall": _fmt(_get(metrics, "recall")),
        "MAE": _fmt(_get(metrics, "mae")),
        "F-measure": _fmt(_get(metrics, "f_measure")),
        "mask_boundary_f1": _fmt(_get(metrics, "mask_boundary_f1")),
        "mask_hd": _fmt(_get(metrics, "mask_hd")),
        "mask_hd95": _fmt(_get(metrics, "mask_hd95")),
        "mask_asd": _fmt(_get(metrics, "mask_asd")),
        "mask_assd": _fmt(_get(metrics, "mask_assd")),
        "Params(M)": _fmt(params_m, 2) if params_m is not None else "",
    }


def _build_paper_row(
    variant: Dict,
    metrics: Optional[Dict[str, float]],
) -> Dict[str, str]:
    return {
        "Model": variant["label"],
        "Dice": _fmt(_get(metrics, "dice")),
        "IoU": _fmt(_get(metrics, "iou")),
        "MAE": _fmt(_get(metrics, "mae")),
        "mask_boundary_f1": _fmt(_get(metrics, "mask_boundary_f1")),
        "mask_hd95": _fmt(_get(metrics, "mask_hd95")),
        "mask_assd": _fmt(_get(metrics, "mask_assd")),
    }


def _write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def _safe_float(s: str) -> Optional[float]:
    try:
        return float(s) if s else None
    except ValueError:
        return None


def _compare_pair(
    rows_dict: Dict[str, Dict[str, str]],
    label_a: str,
    label_b: str,
    col: str,
    higher_is_better: bool = True,
) -> str:
    a = _safe_float(rows_dict.get(label_a, {}).get(col, ""))
    b = _safe_float(rows_dict.get(label_b, {}).get(col, ""))
    if a is None or b is None:
        return f"{label_b} → {label_a}: {col} = N/A (missing data)"
    diff = a - b
    if higher_is_better:
        verdict = "IMPROVEMENT" if diff > 0 else "REGRESSION" if diff < 0 else "NO CHANGE"
    else:
        verdict = "IMPROVEMENT" if diff < 0 else "REGRESSION" if diff > 0 else "NO CHANGE"
    return f"{label_b} → {label_a}: {col} {b:.4f} → {a:.4f} (Δ={diff:+.4f}) [{verdict}]"


def _generate_report(
    full_rows_dict: Dict[str, Dict[str, str]],
    pending: List[str],
) -> str:
    lines: List[str] = []

    lines += [
        "# Ablation Boundary Diagnostic Report",
        "",
        "> Auto-generated by `scripts/generate_ablation_corrected.py`.",
        "> All boundary metrics use the pipeline:",
        "> `final predicted mask → boundary extraction → boundary metrics`",
        "> This is identical for all variants. `head_boundary_f1` (MBGH head output) is",
        "> saved as a diagnostic field in `evaluation/results.json` but is **not** used here.",
        "",
    ]

    if pending:
        lines += [
            "## ⚠️ Missing Evaluation Outputs",
            "",
            "The following experiments have no evaluation output. Run evaluation before",
            "drawing conclusions. Metrics show as empty in the tables.",
            "",
        ]
        for name in pending:
            lines.append(f"- `{name}`")
        lines.append("")

    # 1. Overall improvement
    lines += [
        "## 1. Does Full BGD-SF Improve Over EffB4 Baseline?",
        "",
    ]
    for col, hib in [
        ("Dice", True), ("IoU", True), ("MAE", False),
        ("mask_boundary_f1", True), ("mask_hd95", False), ("mask_assd", False),
    ]:
        lines.append(f"- {_compare_pair(full_rows_dict, FULL_LABEL, BASELINE_LABEL, col, hib)}")
    lines.append("")

    # 2. Region vs boundary
    lines += [
        "## 2. Region-Based vs Boundary-Based Improvement",
        "",
        "| Variant | Dice | IoU | mask_boundary_f1 | mask_hd95 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for v in VARIANTS:
        lbl = v["label"]
        r = full_rows_dict.get(lbl, {})
        lines.append(
            f"| {lbl} | {r.get('Dice', 'N/A')} | {r.get('IoU', 'N/A')}"
            f" | {r.get('mask_boundary_f1', 'N/A')} | {r.get('mask_hd95', 'N/A')} |"
        )
    lines.append("")

    # 3. Fairness statement
    lines += [
        "## 3. Boundary Metric Fairness",
        "",
        "**Status: FAIR.** The boundary derivation pipeline is identical for all variants:",
        "",
        "```",
        "predicted mask → generate_boundary_target (morphological dilation − erosion) → boundary pixels",
        "```",
        "",
        "Models with MBGH: their head output is stored as `head_boundary_f1` (diagnostic only,",
        "not in this table). Models without MBGH: no head output. Both use the same mask-derived",
        "boundary for the reported `mask_boundary_*` metrics.",
        "",
    ]

    # 4. Which modules help
    lines += [
        "## 4. Which Modules Help? (Progressive Dice)",
        "",
    ]
    prev = BASELINE_LABEL
    for v in VARIANTS[1:]:
        lbl = v["label"]
        lines.append(f"- {_compare_pair(full_rows_dict, lbl, prev, 'Dice', True)}")
        prev = lbl
    lines.append("")

    # 5. Inconclusive/regression modules
    lines += [
        "## 5. Inconclusive or Regression Modules",
        "",
        "Threshold: |Δ Dice| < 0.001 is flagged as inconclusive.",
        "",
    ]
    prev = BASELINE_LABEL
    for v in VARIANTS[1:]:
        lbl = v["label"]
        a = _safe_float(full_rows_dict.get(lbl, {}).get("Dice", ""))
        b = _safe_float(full_rows_dict.get(prev, {}).get("Dice", ""))
        if a is not None and b is not None:
            diff = a - b
            if diff < -1e-9:
                verdict = "**HURTS**"
            elif abs(diff) < 0.001:
                verdict = "**Inconclusive** (Δ < 0.001)"
            else:
                verdict = "Helps"
            lines.append(f"- `{lbl}`: Δ Dice = {diff:+.4f} → {verdict}")
        else:
            lines.append(f"- `{lbl}`: N/A (missing data)")
        prev = lbl
    lines.append("")

    # 6. FreqAug recommendation
    lines += [
        "## 6. Should FreqAug Remain in the Final Model?",
        "",
        f"- {_compare_pair(full_rows_dict, FULL_LABEL, NO_FREQAUG_LABEL, 'Dice', True)}",
        f"- {_compare_pair(full_rows_dict, FULL_LABEL, NO_FREQAUG_LABEL, 'mask_boundary_f1', True)}",
        f"- {_compare_pair(full_rows_dict, FULL_LABEL, NO_FREQAUG_LABEL, 'mask_hd95', False)}",
        "",
        "_Recommendation: Keep FreqAug if Dice and mask_boundary_f1 both improve.",
        "Remove if both are neutral or if only one improves marginally._",
        "",
    ]

    # 7. Paper narrative
    lines += [
        "## 7. Recommended Paper Narrative",
        "",
        "_Fill in after your teammate runs evaluation and reviews numbers._",
        "",
        "Suggested structure:",
        "",
        "1. **Baseline (EffB4 encoder-decoder)**: Establishes the lower bound for all metrics.",
        "2. **CMSCA**: Cascaded multi-scale context attention improves feature representation.",
        "3. **Dynamic boundary guidance (BGD)**: Dynamic weighting guided by boundary priors",
        "   further sharpens segmentation.",
        "4. **BG-SAGF**: Boundary-guided skip attention fusion improves boundary adherence",
        "   (`mask_boundary_f1` ↑, `mask_hd95` ↓).",
        "5. **MBGH**: Multi-scale boundary guidance head reinforces region accuracy.",
        "6. **FreqAug**: Frequency-domain augmentation improves cross-domain generalization.",
        "7. **Fairness note**: All boundary metrics (`mask_boundary_*`) are computed from the",
        "   final predicted mask using the same morphological pipeline for every variant.",
        "   The MBGH head’s boundary output is an auxiliary diagnostic (`head_boundary_f1`)",
        "   and is excluded from comparison tables to avoid inflating boundary scores for",
        "   models that include a dedicated boundary head.",
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate corrected ablation tables using mask-derived boundary metrics. "
            "Reads best-checkpoint evaluation outputs only; never falls back to training metrics."
        )
    )
    parser.add_argument(
        "--full-output",
        default="outputs/tables/ablation_corrected_mask_boundary.csv",
    )
    parser.add_argument(
        "--paper-output",
        default="outputs/tables/ablation_corrected_mask_boundary_paper.csv",
    )
    parser.add_argument(
        "--report-output",
        default="outputs/tables/ablation_boundary_diagnostic_report.md",
    )
    args = parser.parse_args()

    full_rows: List[Dict[str, str]] = []
    paper_rows: List[Dict[str, str]] = []
    pending: List[str] = []

    for variant in VARIANTS:
        exp_name = variant["experiment"]
        metrics = _load_eval_summary(exp_name)

        if metrics is None:
            print(
                f"WARNING: Evaluation output missing for {exp_name}. "
                "Please run evaluation using best.pth. "
                "Metrics will be empty — no fallback to training metrics."
            )
            pending.append(exp_name)

        params_m = _count_params(variant["config"])
        full_rows.append(_build_full_row(variant, metrics, params_m))
        paper_rows.append(_build_paper_row(variant, metrics))

    _write_csv(args.full_output, FULL_COLUMNS, full_rows)
    _write_csv(args.paper_output, PAPER_COLUMNS, paper_rows)

    full_rows_dict = {row["Model"]: row for row in full_rows}
    report = _generate_report(full_rows_dict, pending)

    ensure_dir(os.path.dirname(args.report_output))
    with open(args.report_output, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"Saved: {args.report_output}")

    if pending:
        print("\nPending evaluations — run these first:")
        for name in pending:
            print(
                f"  python -m src.evaluate "
                f"--experiment-name {name} "
                f"--checkpoint experiments/{name}/checkpoints/best.pth "
                f"--dataset-root <DATASET_ROOT>"
            )


if __name__ == "__main__":
    main()
