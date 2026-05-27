#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


SPECIAL_ROWS = ["Params(M)", "FLOPs(G)", "FPS"]
LOWER_IS_BETTER = {
    "mae",
    "hd",
    "hd95",
    "hausdorff",
    "hausdorff distance",
    "asd",
    "assd",
    "params(m)",
    "flops(g)",
}


@dataclass(frozen=True)
class Record:
    method: str
    dataset: str
    metric: str
    value: str


def _read_csv(path: str) -> List[Record]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"method_name", "dataset_name", "metric_name", "value"}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = required.difference(set(reader.fieldnames or []))
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        records: List[Record] = []
        for row in reader:
            method = (row.get("method_name") or "").strip()
            dataset = (row.get("dataset_name") or "").strip()
            metric = (row.get("metric_name") or "").strip()
            value = (row.get("value") or "").strip()
            if not method or not metric:
                raise ValueError("Each row must include method_name and metric_name.")
            records.append(Record(method=method, dataset=dataset, metric=metric, value=value))

    if not records:
        raise ValueError("CSV file has no data rows.")
    return records


def _try_float(value: str) -> float | None:
    if value is None:
        return None
    cleaned = value.strip().replace(",", "").replace("%", "")
    if not cleaned:
        return None
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _order_methods(methods: Iterable[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for method in methods:
        if method not in seen:
            ordered.append(method)
            seen.add(method)

    special = [name for name in SPECIAL_ROWS if name in seen]
    regular = [name for name in ordered if name not in SPECIAL_ROWS]
    return regular + special


def _infer_group_keys(methods: List[str]) -> List[str]:
    if not methods:
        return []
    separators = ["|", " - ", " / ", "/"]
    best_sep = None
    best_score = 0

    for sep in separators:
        prefixes: List[str] = []
        for name in methods:
            if sep in name:
                prefix = name.split(sep, 1)[0].strip()
                prefixes.append(prefix)
            else:
                prefixes.append("")

        counts: Dict[str, int] = {}
        for prefix in prefixes:
            if prefix:
                counts[prefix] = counts.get(prefix, 0) + 1
        score = sum(count for count in counts.values() if count >= 2)
        if score > best_score:
            best_score = score
            best_sep = sep

    if best_sep is None:
        return ["all"] * len(methods)

    group_keys: List[str] = []
    for name in methods:
        if best_sep in name:
            prefix = name.split(best_sep, 1)[0].strip()
            group_keys.append(prefix or "all")
        else:
            group_keys.append("all")
    return group_keys


def _column_key_order(records: List[Record]) -> List[Tuple[str, str]]:
    order: List[Tuple[str, str]] = []
    seen = set()
    for record in records:
        key = (record.dataset, record.metric)
        if key not in seen:
            order.append(key)
            seen.add(key)
    return order


def _format_column_label(dataset: str, metric: str) -> str:
    if not dataset or dataset == metric:
        return metric
    return f"{dataset} {metric}".strip()


def _best_second(
    values: List[float],
    higher_is_better: bool = True,
) -> Tuple[float | None, float | None]:
    if not values:
        return None, None
    unique = sorted(set(values), reverse=higher_is_better)
    best = unique[0] if unique else None
    second = unique[1] if len(unique) > 1 else None
    return best, second


def _is_lower_better(metric: str) -> bool:
    return metric.strip().lower() in LOWER_IS_BETTER


def _build_table_data(records: List[Record]) -> Tuple[List[str], List[List[str]], List[Tuple[str, str]], Dict[str, Dict[Tuple[str, str], float]]]:
    method_order = _order_methods(record.method for record in records)
    column_keys = _column_key_order(records)

    values: Dict[str, Dict[Tuple[str, str], str]] = {method: {} for method in method_order}
    numeric_values: Dict[str, Dict[Tuple[str, str], float]] = {method: {} for method in method_order}

    for record in records:
        key = (record.dataset, record.metric)
        if key in values.get(record.method, {}):
            raise ValueError(
                f"Duplicate entry for method '{record.method}', dataset '{record.dataset}', metric '{record.metric}'."
            )
        values.setdefault(record.method, {})[key] = record.value
        number = _try_float(record.value)
        if number is not None:
            numeric_values.setdefault(record.method, {})[key] = number

    header = ["Method"] + [_format_column_label(*key) for key in column_keys]
    rows: List[List[str]] = []
    for method in method_order:
        row = [method]
        for key in column_keys:
            row.append(values.get(method, {}).get(key, ""))
        rows.append(row)

    return header, rows, column_keys, numeric_values


def _draw_group_separators(
    ax: plt.Axes,
    table,
    group_keys: List[str],
    num_cols: int,
) -> None:
    if not group_keys:
        return

    boundaries: List[int] = []
    current = group_keys[0]
    for idx, key in enumerate(group_keys):
        if key != current:
            boundaries.append(idx - 1)
            current = key
    boundaries.append(len(group_keys) - 1)

    ax.figure.canvas.draw()
    for row_idx in boundaries:
        table_row = row_idx + 1
        left_bbox = table[(table_row, 0)].get_bbox()
        right_bbox = table[(table_row, num_cols - 1)].get_bbox()
        ax.hlines(
            left_bbox.y0,
            left_bbox.x0,
            right_bbox.x1,
            colors="#000000",
            linewidth=0.8,
            transform=ax.transAxes,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 8 quantitative comparison table.")
    parser.add_argument("--csv", required=True, help="Path to comparison CSV.")
    parser.add_argument("--output", default="paper_figures/fig_table.png")
    args = parser.parse_args()

    records = _read_csv(args.csv)
    header, rows, column_keys, numeric_values = _build_table_data(records)

    methods = [row[0] for row in rows]
    method_groups = _infer_group_keys([m for m in methods if m not in SPECIAL_ROWS])
    if any(m in SPECIAL_ROWS for m in methods):
        summary_count = sum(1 for m in methods if m in SPECIAL_ROWS)
        method_groups.extend(["summary"] * summary_count)

    plt.rcParams.update({
        "font.size": 8.5,
        "font.family": "DejaVu Sans",
    })

    num_rows = len(rows) + 1
    num_cols = len(header)
    fig_width = max(6.0, 1.05 * num_cols)
    fig_height = max(2.5, 0.35 * num_rows)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_axis_off()

    table = ax.table(
        cellText=rows,
        colLabels=header,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.auto_set_column_width(col=list(range(num_cols)))

    for row_idx in range(1, len(rows) + 1):
        table[(row_idx, 0)].get_text().set_ha("left")

    ours_color = "#dbeafe"
    best_color = "#d9f2d9"
    second_color = "#fff4c2"

    for row_idx, method in enumerate(methods, start=1):
        if "ours" in method.lower():
            for col_idx in range(num_cols):
                table[(row_idx, col_idx)].set_facecolor(ours_color)

    best_second_by_col: Dict[Tuple[str, str], Tuple[float | None, float | None]] = {}
    for key in column_keys:
        values: List[float] = []
        for method in methods:
            number = numeric_values.get(method, {}).get(key)
            if number is not None:
                values.append(number)
        higher_is_better = not _is_lower_better(key[1])
        best_second_by_col[key] = _best_second(values, higher_is_better=higher_is_better)

    tol = 1e-9
    for col_idx, key in enumerate(column_keys, start=1):
        best, second = best_second_by_col.get(key, (None, None))
        for row_idx, method in enumerate(methods, start=1):
            number = numeric_values.get(method, {}).get(key)
            if number is None:
                continue
            if best is not None and abs(number - best) <= tol:
                cell = table[(row_idx, col_idx)]
                cell.set_facecolor(best_color)
                cell.get_text().set_fontweight("bold")
            elif second is not None and abs(number - second) <= tol:
                cell = table[(row_idx, col_idx)]
                cell.set_facecolor(second_color)
                cell.get_text().set_fontweight("bold")

    _draw_group_separators(ax, table, method_groups, num_cols)

    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
