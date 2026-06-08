#!/usr/bin/env python3
"""Diagnostic analysis: EffB4 encoder-decoder baseline vs Full BGD-SF on external datasets.

Reads best-checkpoint evaluation outputs and produces four outputs:
  outputs/tables/external_baseline_vs_full.csv          Full per-dataset metric table
  outputs/tables/external_baseline_vs_full_gap.csv      Delta (BGD-SF minus Baseline) per metric
  outputs/tables/external_baseline_vs_full_paper.csv    Paper-ready condensed table
  outputs/tables/external_baseline_vs_full_report.md    Diagnostic narrative report
  outputs/tables/external_baseline_vs_full_stats.csv    Paired stats (if per-image CSVs exist)

All boundary metrics are mask-derived for a fair comparison (mask_boundary_f1, mask_hd95,
mask_assd). This script never falls back to training final_metrics.

Usage:
  python scripts/analyze_external_baseline_vs_full.py \\
      --datasets CVC-ClinicDB CVC-ColonDB ETIS-Larib CVC-300
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import ensure_dir, load_json

try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASELINE_EXP = "effb4_unet_baseline"
DEFAULT_FULL_EXP = "bgdsf_polysegnet_full"

REPORT_METRICS: List[str] = [
    "dice", "iou", "precision", "recall", "f_measure",
    "mae", "mask_boundary_f1", "mask_hd95", "mask_assd",
]

STAT_METRICS: List[str] = [
    "dice", "iou", "mae", "mask_boundary_f1", "mask_hd95", "mask_assd",
]

HIGHER_IS_BETTER = frozenset({
    "dice", "iou", "precision", "recall", "f_measure", "mask_boundary_f1",
})

LOWER_IS_BETTER = frozenset({
    "mae", "mask_hd", "mask_hd95", "mask_asd", "mask_assd",
})

KNOWN_EXTERNAL_DATASETS: List[str] = [
    "CVC-ClinicDB",
    "CVC-ColonDB",
    "ETIS-Larib",
    "CVC-300",
    "BKAI-IGH",
]

# Paper table: (paper_col_prefix, internal_metric)
PAPER_METRICS: List[Tuple[str, str]] = [
    ("Dice",   "dice"),
    ("BF1",    "mask_boundary_f1"),
    ("HD95",   "mask_hd95"),
    ("ASSD",   "mask_assd"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_dataset_summary(exp_name: str, dataset_name: str) -> Optional[Dict[str, float]]:
    """Load per-dataset summary metrics from evaluation outputs.

    Search order:
    1. experiments/<exp>/evaluation/<dataset_name>/results.json  → summary key
    2. experiments/<exp>/evaluation/summary.json                 → datasets.<dataset_name>
    3. experiments/<exp>/evaluation/results.json                 → summary key (single-dataset)
    """
    eval_dir = os.path.join("experiments", exp_name, "evaluation")

    # 1. Per-dataset subdirectory
    sub_json = os.path.join(eval_dir, dataset_name, "results.json")
    if os.path.isfile(sub_json):
        payload = load_json(sub_json)
        if isinstance(payload, dict):
            s = payload.get("summary")
            if isinstance(s, dict) and s:
                return {k: float(v) for k, v in s.items() if isinstance(v, (int, float))}

    # 2. Top-level summary.json (multi-dataset run)
    summary_json = os.path.join(eval_dir, "summary.json")
    if os.path.isfile(summary_json):
        payload = load_json(summary_json)
        if isinstance(payload, dict):
            dmap = payload.get("datasets", {})
            if isinstance(dmap, dict) and dataset_name in dmap:
                return {k: float(v) for k, v in dmap[dataset_name].items()
                        if isinstance(v, (int, float))}

    # 3. Single-dataset results.json
    single_json = os.path.join(eval_dir, "results.json")
    if os.path.isfile(single_json) and not os.path.isdir(
        os.path.join(eval_dir, dataset_name)
    ):
        payload = load_json(single_json)
        if isinstance(payload, dict):
            s = payload.get("summary")
            if isinstance(s, dict) and s:
                return {k: float(v) for k, v in s.items() if isinstance(v, (int, float))}

    return None


def _load_per_image_csv(exp_name: str, dataset_name: str) -> Optional[List[Dict[str, str]]]:
    """Load per-image results CSV for a specific dataset."""
    eval_dir = os.path.join("experiments", exp_name, "evaluation")

    # Per-dataset subdirectory
    sub_csv = os.path.join(eval_dir, dataset_name, "results.csv")
    if os.path.isfile(sub_csv):
        with open(sub_csv, "r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    # Single-dataset results.csv
    single_csv = os.path.join(eval_dir, "results.csv")
    if os.path.isfile(single_csv) and not os.path.isdir(
        os.path.join(eval_dir, dataset_name)
    ):
        with open(single_csv, "r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    return None


def _discover_datasets(exp_name: str) -> List[str]:
    """Auto-discover dataset names from evaluation output directory."""
    eval_dir = os.path.join("experiments", exp_name, "evaluation")
    if not os.path.isdir(eval_dir):
        return []

    # From summary.json datasets key
    summary_json = os.path.join(eval_dir, "summary.json")
    if os.path.isfile(summary_json):
        payload = load_json(summary_json)
        if isinstance(payload, dict):
            dmap = payload.get("datasets", {})
            if isinstance(dmap, dict) and dmap:
                return sorted(dmap.keys())

    # From subdirectories containing results.json
    found = [
        name for name in sorted(os.listdir(eval_dir))
        if os.path.isdir(os.path.join(eval_dir, name))
        and os.path.isfile(os.path.join(eval_dir, name, "results.json"))
    ]
    return found


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float], decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    return f"{v:.{decimals}f}"


def _fmt_delta(v: Optional[float], decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    return f"{v:+.{decimals}f}"


def _better_method(
    metric: str,
    baseline_val: Optional[float],
    full_val: Optional[float],
    baseline_label: str,
    full_label: str,
) -> str:
    if baseline_val is None or full_val is None:
        return "N/A"
    if metric in HIGHER_IS_BETTER:
        if full_val > baseline_val:
            return full_label
        if full_val < baseline_val:
            return baseline_label
        return "Tie"
    if metric in LOWER_IS_BETTER:
        if full_val < baseline_val:
            return full_label
        if full_val > baseline_val:
            return baseline_label
        return "Tie"
    return "N/A"


def _build_full_rows(
    datasets: List[str],
    baseline_data: Dict[str, Optional[Dict[str, float]]],
    full_data: Dict[str, Optional[Dict[str, float]]],
    baseline_label: str,
    full_label: str,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for ds in datasets:
        for method_label, data_map in [
            (baseline_label, baseline_data),
            (full_label, full_data),
        ]:
            m = data_map.get(ds)
            row: Dict[str, str] = {
                "dataset_name": ds,
                "method_name": method_label,
            }
            for metric in REPORT_METRICS:
                row[metric] = _fmt(m.get(metric) if m else None)
            rows.append(row)
    return rows


def _build_gap_rows(
    datasets: List[str],
    baseline_data: Dict[str, Optional[Dict[str, float]]],
    full_data: Dict[str, Optional[Dict[str, float]]],
    baseline_label: str,
    full_label: str,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for ds in datasets:
        bm = baseline_data.get(ds)
        fm = full_data.get(ds)
        for metric in REPORT_METRICS:
            bv = bm.get(metric) if bm else None
            fv = fm.get(metric) if fm else None
            delta: Optional[float] = None
            if bv is not None and fv is not None:
                delta = fv - bv  # always full minus baseline
            rows.append({
                "dataset_name": ds,
                "metric": metric,
                "effb4_baseline": _fmt(bv),
                "full_bgdsf": _fmt(fv),
                "delta": _fmt_delta(delta),
                "better_method": _better_method(metric, bv, fv, baseline_label, full_label),
            })
    return rows


def _build_paper_rows(
    datasets: List[str],
    baseline_data: Dict[str, Optional[Dict[str, float]]],
    full_data: Dict[str, Optional[Dict[str, float]]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for ds in datasets:
        bm = baseline_data.get(ds)
        fm = full_data.get(ds)
        row: Dict[str, str] = {"Dataset": ds}
        for col_prefix, internal in PAPER_METRICS:
            bv = bm.get(internal) if bm else None
            fv = fm.get(internal) if fm else None
            delta = (fv - bv) if (bv is not None and fv is not None) else None
            row[f"EffB4 {col_prefix}"] = _fmt(bv)
            row[f"BGD-SF {col_prefix}"] = _fmt(fv)
            row[f"Δ {col_prefix}"] = _fmt_delta(delta)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def _to_float(s: str) -> Optional[float]:
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def _rank_biserial(diff: np.ndarray) -> Optional[float]:
    nonzero = diff[diff != 0]
    n = nonzero.size
    if n == 0:
        return None
    from scipy import stats
    ranks = stats.rankdata(np.abs(nonzero))
    w_pos = float(np.sum(ranks[nonzero > 0]))
    w_neg = float(np.sum(ranks[nonzero < 0]))
    total = n * (n + 1) / 2.0
    return (w_pos - w_neg) / total if total > 0 else None


def _cohens_dz(diff: np.ndarray) -> Optional[float]:
    if diff.size < 2:
        return None
    std = float(np.std(diff, ddof=1))
    return float(np.mean(diff) / std) if std > 0 else None


def _run_paired_stats(
    baseline_rows: List[Dict[str, str]],
    full_rows: List[Dict[str, str]],
    dataset_name: str,
    baseline_label: str,
    full_label: str,
) -> List[Dict[str, object]]:
    if not _HAS_SCIPY:
        print("WARNING: scipy not available — skipping statistical tests.")
        return []

    # Index by image name
    def _image_key(row: Dict[str, str]) -> str:
        val = row.get("image") or row.get("image_name") or ""
        if not val:
            ip = row.get("image_path", "")
            val = os.path.splitext(os.path.basename(ip))[0]
        return val

    baseline_idx = {_image_key(r): r for r in baseline_rows if _image_key(r)}
    full_idx = {_image_key(r): r for r in full_rows if _image_key(r)}
    common_keys = sorted(set(baseline_idx) & set(full_idx))

    if not common_keys:
        print(
            f"WARNING: No common images found between experiments for dataset "
            f"'{dataset_name}' — skipping statistical tests."
        )
        return []

    if len(common_keys) < len(baseline_idx) or len(common_keys) < len(full_idx):
        print(
            f"WARNING: Partial image overlap for '{dataset_name}': "
            f"{len(common_keys)} common out of "
            f"{len(baseline_idx)} baseline / {len(full_idx)} full images."
        )

    results: List[Dict[str, object]] = []
    from scipy import stats as scipy_stats

    for metric in STAT_METRICS:
        vals_b, vals_f = [], []
        for key in common_keys:
            bv = _to_float(baseline_idx[key].get(metric, ""))
            fv = _to_float(full_idx[key].get(metric, ""))
            if bv is not None and fv is not None:
                vals_b.append(bv)
                vals_f.append(fv)

        n_pairs = len(vals_b)
        base_row: Dict[str, object] = {
            "dataset_name": dataset_name,
            "method_a": full_label,
            "method_b": baseline_label,
            "metric": metric,
            "n_pairs": n_pairs,
            "mean_full": float(np.mean(vals_f)) if vals_f else None,
            "mean_baseline": float(np.mean(vals_b)) if vals_b else None,
        }

        if n_pairs < 2:
            for test in ("wilcoxon", "ttest_rel"):
                results.append({
                    **base_row,
                    "test_name": test,
                    "statistic": None,
                    "p_value": None,
                    "effect_size": None,
                    "significant_0_05": False,
                    "better_method": "N/A",
                    "notes": "insufficient_pairs",
                })
            continue

        arr_b = np.array(vals_b, dtype=float)
        arr_f = np.array(vals_f, dtype=float)
        is_higher = metric in HIGHER_IS_BETTER
        # diff oriented so positive = full is better
        diff = arr_f - arr_b if is_higher else arr_b - arr_f
        mean_diff = float(np.mean(diff))
        if mean_diff > 0:
            better = full_label
        elif mean_diff < 0:
            better = baseline_label
        else:
            better = "Tie"

        # Wilcoxon
        w_stat = w_p = w_effect = None
        w_note = ""
        try:
            wres = scipy_stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
            w_stat = float(wres.statistic)
            w_p = float(wres.pvalue)
            w_effect = _rank_biserial(diff)
            if w_effect is None:
                w_effect = _cohens_dz(diff)
        except Exception as exc:
            w_note = f"wilcoxon_failed:{exc}"

        results.append({
            **base_row,
            "test_name": "wilcoxon",
            "statistic": w_stat,
            "p_value": w_p,
            "effect_size": w_effect,
            "significant_0_05": bool(w_p is not None and w_p < 0.05),
            "better_method": better,
            "notes": w_note,
        })

        # Paired t-test
        t_stat = t_p = t_effect = None
        t_note = ""
        try:
            tres = scipy_stats.ttest_rel(arr_f, arr_b, nan_policy="omit")
            t_stat = float(tres.statistic)
            t_p = float(tres.pvalue)
            t_effect = _cohens_dz(diff)
        except Exception as exc:
            t_note = f"ttest_failed:{exc}"

        results.append({
            **base_row,
            "test_name": "ttest_rel",
            "statistic": t_stat,
            "p_value": t_p,
            "effect_size": t_effect,
            "significant_0_05": bool(t_p is not None and t_p < 0.05),
            "better_method": better,
            "notes": t_note,
        })

    return results


# ---------------------------------------------------------------------------
# Diagnostic report
# ---------------------------------------------------------------------------

def _delta_sign_ok(metric: str, delta: Optional[float]) -> Optional[bool]:
    """Return True if the delta indicates Full BGD-SF is better, None if missing."""
    if delta is None:
        return None
    if metric in HIGHER_IS_BETTER:
        return delta > 0
    if metric in LOWER_IS_BETTER:
        return delta < 0
    return None


def _verdict(ok: Optional[bool]) -> str:
    if ok is None:
        return "N/A"
    return "BGD-SF ✓" if ok else "Baseline ✓"


def _generate_report(
    datasets: List[str],
    baseline_data: Dict[str, Optional[Dict[str, float]]],
    full_data: Dict[str, Optional[Dict[str, float]]],
    gap_rows: List[Dict[str, str]],
    stats_rows: Optional[List[Dict[str, object]]],
    baseline_label: str,
    full_label: str,
    missing_baseline: List[str],
    missing_full: List[str],
) -> str:
    lines: List[str] = []

    def h(n: int, text: str) -> None:
        lines.append(f"{'#' * n} {text}")
        lines.append("")

    def p(text: str) -> None:
        lines.append(text)
        lines.append("")

    # Header
    h(1, "External Baseline-vs-Full BGD-SF Diagnostic Report")
    lines += [
        "> Auto-generated by `scripts/analyze_external_baseline_vs_full.py`.",
        "> Boundary metrics (`mask_boundary_f1`, `mask_hd95`, `mask_assd`) are derived",
        "> from the **final predicted mask** for both methods — fair comparison guaranteed.",
        "",
    ]

    # Missing data warnings
    if missing_baseline or missing_full:
        h(2, "⚠️ Missing Evaluation Outputs")
        if missing_baseline:
            p(f"`{baseline_label}` evaluation missing for: {', '.join(missing_baseline)}")
        if missing_full:
            p(f"`{full_label}` evaluation missing for: {', '.join(missing_full)}")
        p("Re-run evaluation for missing datasets before drawing conclusions.")

    # --- Section 1: Overall improvement summary ---
    h(2, "1. Overall Improvement: Full BGD-SF vs EffB4 Baseline")

    # Count wins per metric across all datasets
    metric_wins: Dict[str, int] = {m: 0 for m in REPORT_METRICS}
    metric_total: Dict[str, int] = {m: 0 for m in REPORT_METRICS}
    for row in gap_rows:
        m = row["metric"]
        if m not in REPORT_METRICS:
            continue
        bv = _to_float(row["effb4_baseline"])
        fv = _to_float(row["full_bgdsf"])
        if bv is None or fv is None:
            continue
        metric_total[m] += 1
        delta = fv - bv
        ok = _delta_sign_ok(m, delta)
        if ok:
            metric_wins[m] += 1

    lines.append("| Metric | BGD-SF Wins / Total Datasets | Verdict |")
    lines.append("| --- | --- | --- |")
    for m in REPORT_METRICS:
        total = metric_total[m]
        wins = metric_wins[m]
        if total == 0:
            verdict = "N/A"
        elif wins == total:
            verdict = f"✅ Improves on all {total} datasets"
        elif wins > total / 2:
            verdict = f"⬆ Improves on {wins}/{total} datasets"
        elif wins == total / 2:
            verdict = f"⟺ Split ({wins}/{total})"
        else:
            verdict = f"⬇ Worse on {total - wins}/{total} datasets"
        lines.append(f"| `{m}` | {wins}/{total} | {verdict} |")
    lines.append("")

    # --- Section 2: Region vs boundary ---
    h(2, "2. Is Improvement Region-Based or Boundary-Based?")
    region_metrics = ["dice", "iou", "f_measure"]
    boundary_metrics = ["mask_boundary_f1", "mask_hd95", "mask_assd"]

    region_wins_total = sum(metric_wins.get(m, 0) for m in region_metrics)
    region_n_total = sum(metric_total.get(m, 0) for m in region_metrics)
    boundary_wins_total = sum(metric_wins.get(m, 0) for m in boundary_metrics)
    boundary_n_total = sum(metric_total.get(m, 0) for m in boundary_metrics)

    p(
        f"- **Region metrics** (Dice, IoU, F-measure): BGD-SF wins "
        f"{region_wins_total}/{region_n_total} cases."
    )
    p(
        f"- **Boundary metrics** (mask_boundary_f1, mask_hd95, mask_assd): BGD-SF wins "
        f"{boundary_wins_total}/{boundary_n_total} cases."
    )

    if region_n_total > 0 and boundary_n_total > 0:
        region_rate = region_wins_total / region_n_total
        boundary_rate = boundary_wins_total / boundary_n_total
        if boundary_rate > region_rate + 0.1:
            p("→ Improvement is **primarily boundary-based**.")
        elif region_rate > boundary_rate + 0.1:
            p("→ Improvement is **primarily region-based**.")
        else:
            p("→ Improvement is **consistent across both region and boundary metrics**.")

    # --- Section 3: Per-dataset Dice + BF1 table ---
    h(2, "3. Per-Dataset Summary (Dice and Boundary F1)")
    lines.append("| Dataset | EffB4 Dice | BGD-SF Dice | Δ Dice | EffB4 BF1 | BGD-SF BF1 | Δ BF1 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for ds in datasets:
        bm = baseline_data.get(ds)
        fm = full_data.get(ds)
        bd = bm.get("dice") if bm else None
        fd = fm.get("dice") if fm else None
        bb = bm.get("mask_boundary_f1") if bm else None
        fb = fm.get("mask_boundary_f1") if fm else None
        dd = (fd - bd) if (bd is not None and fd is not None) else None
        db = (fb - bb) if (bb is not None and fb is not None) else None
        lines.append(
            f"| {ds} | {_fmt(bd)} | {_fmt(fd)} | {_fmt_delta(dd)} "
            f"| {_fmt(bb)} | {_fmt(fb)} | {_fmt_delta(db)} |"
        )
    lines.append("")

    # --- Section 4: Largest gain / weakest ---
    h(2, "4. Dataset-Level Gain Analysis")

    dice_deltas: Dict[str, Optional[float]] = {}
    for ds in datasets:
        bm = baseline_data.get(ds)
        fm = full_data.get(ds)
        bd = bm.get("dice") if bm else None
        fd = fm.get("dice") if fm else None
        dice_deltas[ds] = (fd - bd) if (bd is not None and fd is not None) else None

    valid_deltas = {ds: v for ds, v in dice_deltas.items() if v is not None}
    if valid_deltas:
        best_ds = max(valid_deltas, key=lambda k: valid_deltas[k])
        worst_ds = min(valid_deltas, key=lambda k: valid_deltas[k])
        p(
            f"- **Largest Dice gain**: `{best_ds}` "
            f"(Δ = {_fmt_delta(valid_deltas[best_ds])})"
        )
        p(
            f"- **Smallest Dice gain** (or regression): `{worst_ds}` "
            f"(Δ = {_fmt_delta(valid_deltas[worst_ds])})"
        )
        negative = [ds for ds, v in valid_deltas.items() if v < -1e-6]
        if negative:
            p(
                f"- **Datasets with Dice regression**: {', '.join(f'`{d}`' for d in negative)} "
                f"— investigate whether test set distribution differs significantly from training."
            )
        else:
            p("- No datasets show Dice regression.")
    else:
        p("Insufficient data to determine dataset-level gains.")

    # --- Section 5: External robustness claim ---
    h(2, "5. Does This Support the External Robustness Claim?")

    overall_dice_wins = metric_wins.get("dice", 0)
    overall_dice_total = metric_total.get("dice", 0)
    overall_bf1_wins = metric_wins.get("mask_boundary_f1", 0)
    overall_bf1_total = metric_total.get("mask_boundary_f1", 0)

    if overall_dice_total == 0:
        p("Cannot evaluate — no data available.")
    elif overall_dice_wins == overall_dice_total and overall_bf1_wins >= overall_bf1_total * 0.5:
        p(
            "✅ **Yes, the claim is supported.** Full BGD-SF improves Dice on all evaluated "
            "external datasets and shows consistent boundary metric improvements, supporting "
            "the claim that the proposed modules improve external generalization."
        )
    elif overall_dice_wins > overall_dice_total / 2:
        p(
            "⚠️ **Partially supported.** Full BGD-SF improves Dice on the majority of "
            f"external datasets ({overall_dice_wins}/{overall_dice_total}). The claim "
            "should be qualified to acknowledge datasets where gains are marginal or absent."
        )
    else:
        p(
            "❌ **Not clearly supported.** Full BGD-SF does not consistently outperform "
            "the EffB4 baseline on external datasets. Investigate training dataset choice, "
            "checkpoint quality, and whether FreqAug is helping generalization."
        )

    # --- Section 6: Statistical significance ---
    if stats_rows:
        h(2, "6. Statistical Significance (Paired Tests)")
        wilcoxon_rows = [r for r in stats_rows if r.get("test_name") == "wilcoxon"]
        if wilcoxon_rows:
            lines.append("| Dataset | Metric | p-value (Wilcoxon) | Significant? | Better Method |")
            lines.append("| --- | --- | --- | --- | --- |")
            for r in wilcoxon_rows:
                p_val = r.get("p_value")
                sig = "✅ Yes" if r.get("significant_0_05") else "No"
                p_str = f"{float(p_val):.4f}" if p_val is not None else "N/A"
                lines.append(
                    f"| {r['dataset_name']} | `{r['metric']}` | {p_str} | {sig} | {r['better_method']} |"
                )
            lines.append("")
        else:
            p("No Wilcoxon results available.")
    elif stats_rows is not None:
        h(2, "6. Statistical Significance")
        p("Per-image CSV files were not found for one or both methods — statistical tests skipped.")
    else:
        h(2, "6. Statistical Significance")
        p("Per-image CSVs not available or statistical testing was not requested.")

    # --- Section 7: Paper narrative ---
    h(2, "7. Recommended Paper Narrative")
    lines += [
        "_Fill in the exact numbers after your teammate runs evaluation._",
        "",
        "Suggested structure:",
        "",
        "**Generalization paragraph:**",
        "> We evaluate both the EffB4 encoder-decoder baseline and the full BGD-SF model on",
        "> [N] external polyp segmentation datasets not seen during training. Full BGD-SF achieves",
        "> consistently higher Dice (Δ = [X.XX] on average) and improved boundary adherence",
        "> (mask_boundary_f1 Δ = [X.XX]), demonstrating that the proposed BGD-CMSCA, BG-SAGF,",
        "> MBGH, and FreqAug components collectively improve external generalization rather than",
        "> merely overfitting to the training distribution.",
        "",
        "**Fairness note (Methods section):**",
        "> Boundary metrics for all compared models are derived from the final segmentation mask",
        "> using a consistent morphological pipeline (dilation − erosion), ensuring a fair",
        "> comparison regardless of whether a model includes a dedicated boundary prediction head.",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV / JSON writers
# ---------------------------------------------------------------------------

def _write_csv(path: str, fieldnames: List[str], rows: List[Dict]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def _write_stats_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = [
        "dataset_name", "method_a", "method_b", "metric", "n_pairs",
        "mean_full", "mean_baseline", "test_name", "statistic", "p_value",
        "effect_size", "significant_0_05", "better_method", "notes",
    ]
    _write_csv(path, fieldnames, rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic analysis: EffB4 baseline vs Full BGD-SF on external datasets. "
            "Reads best-checkpoint evaluation outputs only. Never falls back to training metrics."
        )
    )
    parser.add_argument(
        "--baseline-exp",
        default=DEFAULT_BASELINE_EXP,
        help=f"Baseline experiment name (default: {DEFAULT_BASELINE_EXP})",
    )
    parser.add_argument(
        "--full-exp",
        default=DEFAULT_FULL_EXP,
        help=f"Full model experiment name (default: {DEFAULT_FULL_EXP})",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Explicit dataset names to include. "
            "If omitted, auto-discovered from evaluation directories."
        ),
    )
    parser.add_argument(
        "--baseline-label",
        default="EffB4 baseline",
        help="Display label for the baseline method.",
    )
    parser.add_argument(
        "--full-label",
        default="BGD-SF (Full)",
        help="Display label for the full model.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/tables",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip statistical tests even if per-image CSVs are available.",
    )
    args = parser.parse_args()

    baseline_exp = args.baseline_exp
    full_exp = args.full_exp
    baseline_label = args.baseline_label
    full_label = args.full_label
    output_dir = args.output_dir

    # --- Resolve dataset list ---
    if args.datasets:
        datasets = list(args.datasets)
    else:
        discovered_b = set(_discover_datasets(baseline_exp))
        discovered_f = set(_discover_datasets(full_exp))
        datasets = sorted(discovered_b | discovered_f)
        if not datasets:
            datasets = list(KNOWN_EXTERNAL_DATASETS)
            print(
                "WARNING: Could not auto-discover datasets from evaluation directories. "
                f"Using default list: {datasets}. "
                "Pass --datasets explicitly if these are wrong."
            )
        else:
            print(f"Auto-discovered datasets: {datasets}")

    # --- Load summaries ---
    baseline_data: Dict[str, Optional[Dict[str, float]]] = {}
    full_data: Dict[str, Optional[Dict[str, float]]] = {}
    missing_baseline: List[str] = []
    missing_full: List[str] = []

    for ds in datasets:
        bm = _load_dataset_summary(baseline_exp, ds)
        fm = _load_dataset_summary(full_exp, ds)

        if bm is None:
            print(
                f"WARNING: Evaluation output missing for '{baseline_exp}' on dataset '{ds}'. "
                "Run evaluation using best.pth. Metrics will be empty — no fallback."
            )
            missing_baseline.append(ds)
        if fm is None:
            print(
                f"WARNING: Evaluation output missing for '{full_exp}' on dataset '{ds}'. "
                "Run evaluation using best.pth. Metrics will be empty — no fallback."
            )
            missing_full.append(ds)

        baseline_data[ds] = bm
        full_data[ds] = fm

    # --- Build tables ---
    full_rows = _build_full_rows(
        datasets, baseline_data, full_data, baseline_label, full_label
    )
    gap_rows = _build_gap_rows(
        datasets, baseline_data, full_data, baseline_label, full_label
    )
    paper_rows = _build_paper_rows(datasets, baseline_data, full_data)

    # --- Statistical tests ---
    all_stats: List[Dict[str, object]] = []
    stats_attempted = False
    if not args.no_stats:
        for ds in datasets:
            b_csv = _load_per_image_csv(baseline_exp, ds)
            f_csv = _load_per_image_csv(full_exp, ds)
            if b_csv is None or f_csv is None:
                if b_csv is None:
                    print(
                        f"WARNING: Per-image CSV not found for '{baseline_exp}' / '{ds}' "
                        "— skipping statistical tests for this dataset."
                    )
                if f_csv is None:
                    print(
                        f"WARNING: Per-image CSV not found for '{full_exp}' / '{ds}' "
                        "— skipping statistical tests for this dataset."
                    )
                continue
            stats_attempted = True
            ds_stats = _run_paired_stats(
                b_csv, f_csv, ds, baseline_label, full_label
            )
            all_stats.extend(ds_stats)

    # --- Write outputs ---
    full_table_path = os.path.join(output_dir, "external_baseline_vs_full.csv")
    full_fieldnames = ["dataset_name", "method_name"] + REPORT_METRICS
    _write_csv(full_table_path, full_fieldnames, full_rows)

    gap_path = os.path.join(output_dir, "external_baseline_vs_full_gap.csv")
    _write_csv(
        gap_path,
        ["dataset_name", "metric", "effb4_baseline", "full_bgdsf", "delta", "better_method"],
        gap_rows,
    )

    paper_path = os.path.join(output_dir, "external_baseline_vs_full_paper.csv")
    paper_fieldnames = ["Dataset"]
    for col_prefix, _ in PAPER_METRICS:
        paper_fieldnames += [f"EffB4 {col_prefix}", f"BGD-SF {col_prefix}", f"Δ {col_prefix}"]
    _write_csv(paper_path, paper_fieldnames, paper_rows)

    if all_stats:
        stats_path = os.path.join(output_dir, "external_baseline_vs_full_stats.csv")
        _write_stats_csv(stats_path, all_stats)

    report_path = os.path.join(output_dir, "external_baseline_vs_full_report.md")
    report = _generate_report(
        datasets,
        baseline_data,
        full_data,
        gap_rows,
        all_stats if (stats_attempted or all_stats) else None,
        baseline_label,
        full_label,
        missing_baseline,
        missing_full,
    )
    ensure_dir(os.path.dirname(report_path))
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"Saved: {report_path}")

    if missing_baseline or missing_full:
        print(
            "\nRun the following to evaluate missing experiments "
            "(replace <DATASET_ROOT> with actual paths):"
        )
        for ds in missing_baseline:
            print(
                f"  python -m src.evaluate --experiment-name {baseline_exp} "
                f"--checkpoint experiments/{baseline_exp}/checkpoints/best.pth "
                f"--dataset-root <PATH_TO_{ds}>"
            )
        for ds in missing_full:
            print(
                f"  python -m src.evaluate --experiment-name {full_exp} "
                f"--checkpoint experiments/{full_exp}/checkpoints/best.pth "
                f"--dataset-root <PATH_TO_{ds}>"
            )


if __name__ == "__main__":
    main()
