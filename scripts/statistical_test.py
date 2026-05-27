#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import stats


HIGHER_IS_BETTER = {
    "dice",
    "iou",
    "precision",
    "recall",
    "f_measure",
    "boundary_f1",
    "fps",
}

LOWER_IS_BETTER = {
    "mae",
    "hausdorff",
    "hd",
    "hd95",
    "asd",
    "flops_g",
    "params_m",
    "ms_per_image",
}


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return float(np.mean(values))


def _rank_biserial(diff: np.ndarray) -> float | None:
    nonzero = diff[diff != 0]
    n = nonzero.size
    if n == 0:
        return None
    ranks = stats.rankdata(np.abs(nonzero))
    w_pos = float(np.sum(ranks[nonzero > 0]))
    w_neg = float(np.sum(ranks[nonzero < 0]))
    total = n * (n + 1) / 2.0
    return (w_pos - w_neg) / total if total > 0 else None


def _cohens_dz(diff: np.ndarray) -> float | None:
    if diff.size < 2:
        return None
    std = float(np.std(diff, ddof=1))
    if std == 0:
        return None
    return float(np.mean(diff) / std)


def _aligned_pairs(
    rows: List[Dict[str, str]],
    method_a: str,
    method_b: str,
    metrics: List[str],
    dataset_name: str | None,
    image_col: str,
    method_col: str,
    dataset_col: str,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, str]]:
    data_a: Dict[str, Dict[str, float]] = {}
    data_b: Dict[str, Dict[str, float]] = {}
    meta: Dict[str, str] = {}

    for row in rows:
        if row.get(method_col) not in {method_a, method_b}:
            continue
        if dataset_name and row.get(dataset_col) != dataset_name:
            continue

        image_id = row.get(image_col, "")
        if not image_id:
            continue
        method = row.get(method_col, "")
        dataset = row.get(dataset_col, "")

        target = data_a if method == method_a else data_b
        metrics_dict = target.setdefault(image_id, {})
        for metric in metrics:
            value = _to_float(row.get(metric))
            if value is not None:
                metrics_dict[metric] = value
        if dataset:
            meta[image_id] = dataset

    return data_a, data_b, meta


def _compare_methods(
    rows: List[Dict[str, str]],
    method_a: str,
    method_b: str,
    metrics: List[str],
    dataset_name: str | None,
    image_col: str,
    method_col: str,
    dataset_col: str,
) -> List[Dict[str, object]]:
    data_a, data_b, meta = _aligned_pairs(
        rows,
        method_a,
        method_b,
        metrics,
        dataset_name,
        image_col,
        method_col,
        dataset_col,
    )

    common_ids = sorted(set(data_a.keys()) & set(data_b.keys()))
    missing_a = len(data_b) - len(common_ids)
    missing_b = len(data_a) - len(common_ids)

    if missing_a or missing_b:
        dataset_label = dataset_name or "all"
        print(
            "Warning: missing paired images for "
            f"{method_a} vs {method_b} on {dataset_label} "
            f"(missing_a={missing_a}, missing_b={missing_b})"
        )

    results: List[Dict[str, object]] = []
    for metric in metrics:
        values_a = []
        values_b = []
        for image_id in common_ids:
            if metric not in data_a.get(image_id, {}) or metric not in data_b.get(image_id, {}):
                continue
            values_a.append(data_a[image_id][metric])
            values_b.append(data_b[image_id][metric])

        n_pairs = len(values_a)
        if n_pairs == 0:
            note = "no common pairs"
            results.append(
                {
                    "dataset_name": dataset_name or "all",
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": metric,
                    "n_pairs": 0,
                    "mean_a": None,
                    "mean_b": None,
                    "mean_difference": None,
                    "test_name": "wilcoxon",
                    "statistic": None,
                    "p_value": None,
                    "effect_size": None,
                    "significant_0_05": False,
                    "better_method": "",
                    "notes": note,
                }
            )
            results.append(
                {
                    "dataset_name": dataset_name or "all",
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": metric,
                    "n_pairs": 0,
                    "mean_a": None,
                    "mean_b": None,
                    "mean_difference": None,
                    "test_name": "ttest_rel",
                    "statistic": None,
                    "p_value": None,
                    "effect_size": None,
                    "significant_0_05": False,
                    "better_method": "",
                    "notes": note,
                }
            )
            continue

        values_a_arr = np.array(values_a, dtype=float)
        values_b_arr = np.array(values_b, dtype=float)

        is_higher = metric.lower() in HIGHER_IS_BETTER
        diff = values_a_arr - values_b_arr if is_higher else values_b_arr - values_a_arr
        mean_a = float(np.mean(values_a_arr))
        mean_b = float(np.mean(values_b_arr))
        mean_diff = float(np.mean(diff))

        better_method = ""
        if mean_diff > 0:
            better_method = method_a
        elif mean_diff < 0:
            better_method = method_b
        else:
            better_method = "tie"

        note_parts = []
        if missing_a or missing_b:
            note_parts.append(f"missing_pairs_a:{missing_a}")
            note_parts.append(f"missing_pairs_b:{missing_b}")
        notes = ";".join(note_parts) if note_parts else ""

        # Wilcoxon signed-rank test
        wilcoxon_stat = None
        wilcoxon_p = None
        effect = None
        try:
            wilcoxon_res = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
            wilcoxon_stat = float(wilcoxon_res.statistic)
            wilcoxon_p = float(wilcoxon_res.pvalue)
            effect = _rank_biserial(diff)
            if effect is None:
                effect = _cohens_dz(diff)
        except Exception as exc:
            notes = (notes + ";" if notes else "") + f"wilcoxon_failed:{exc}"

        results.append(
            {
                "dataset_name": dataset_name or "all",
                "method_a": method_a,
                "method_b": method_b,
                "metric": metric,
                "n_pairs": n_pairs,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "mean_difference": mean_diff,
                "test_name": "wilcoxon",
                "statistic": wilcoxon_stat,
                "p_value": wilcoxon_p,
                "effect_size": effect,
                "significant_0_05": bool(wilcoxon_p is not None and wilcoxon_p < 0.05),
                "better_method": better_method,
                "notes": notes,
            }
        )

        # Paired t-test
        t_stat = None
        t_p = None
        t_effect = None
        try:
            t_res = stats.ttest_rel(values_a_arr, values_b_arr, nan_policy="omit")
            t_stat = float(t_res.statistic)
            t_p = float(t_res.pvalue)
            t_effect = _cohens_dz(diff)
        except Exception as exc:
            notes_t = (notes + ";" if notes else "") + f"ttest_failed:{exc}"
        else:
            notes_t = notes

        results.append(
            {
                "dataset_name": dataset_name or "all",
                "method_a": method_a,
                "method_b": method_b,
                "metric": metric,
                "n_pairs": n_pairs,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "mean_difference": mean_diff,
                "test_name": "ttest_rel",
                "statistic": t_stat,
                "p_value": t_p,
                "effect_size": t_effect,
                "significant_0_05": bool(t_p is not None and t_p < 0.05),
                "better_method": better_method,
                "notes": notes_t,
            }
        )

    return results


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "dataset_name",
        "method_a",
        "method_b",
        "metric",
        "n_pairs",
        "mean_a",
        "mean_b",
        "mean_difference",
        "test_name",
        "statistic",
        "p_value",
        "effect_size",
        "significant_0_05",
        "better_method",
        "notes",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"results": rows}, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired statistical tests per image.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--method-a", default=None)
    parser.add_argument("--method-b", default=None)
    parser.add_argument("--compare-all-to", default=None)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--image-id-column", default="image_name")
    parser.add_argument("--method-column", default="method_name")
    parser.add_argument("--dataset-column", default="dataset_name")
    parser.add_argument(
        "--output-csv",
        default="outputs/tables/statistical_tests.csv",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/tables/statistical_tests.json",
    )
    args = parser.parse_args()

    rows = _read_csv(args.input_csv)

    if args.compare_all_to:
        method_a = args.compare_all_to
        methods = sorted({row.get(args.method_column, "") for row in rows if row.get(args.method_column)})
        method_pairs = [(method_a, method) for method in methods if method and method != method_a]
    else:
        if not args.method_a or not args.method_b:
            raise ValueError("Provide --method-a and --method-b, or use --compare-all-to.")
        method_pairs = [(args.method_a, args.method_b)]

    dataset_names: List[str | None]
    if args.dataset_name:
        dataset_names = [args.dataset_name]
    else:
        dataset_names = sorted(
            {row.get(args.dataset_column, "") for row in rows if row.get(args.dataset_column)}
        )
        if not dataset_names:
            dataset_names = [None]

    results: List[Dict[str, object]] = []
    for dataset_name in dataset_names:
        for method_a, method_b in method_pairs:
            results.extend(
                _compare_methods(
                    rows,
                    method_a,
                    method_b,
                    args.metrics,
                    dataset_name,
                    args.image_id_column,
                    args.method_column,
                    args.dataset_column,
                )
            )

    _write_csv(args.output_csv, results)
    _write_json(args.output_json, results)
    print(f"Saved CSV to {args.output_csv}")
    print(f"Saved JSON to {args.output_json}")


if __name__ == "__main__":
    main()
