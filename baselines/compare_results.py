import csv
import os
import sys
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import ensure_dir, load_json


EXPERIMENTS = [
    ("unet_baseline", "Plain U-Net"),
    ("effb4_unet", "EffB4 U-Net"),
    ("effb4_unet_msca", "EffB4 U-Net + MSCA"),
    ("effb4_unet_msca_csaf", "EffB4 U-Net + MSCA + CSAF"),
    ("polysegnet_full", "PolySegNet + Boundary"),
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
        "fbeta": float(metrics.get("val_fbeta", metrics.get("val_f1", 0.0))),
    }


def compare_results() -> None:
    rows: List[List[str]] = []
    for exp_name, label in EXPERIMENTS:
        results_path = os.path.join("experiments", exp_name, "results.json")
        if not os.path.isfile(results_path):
            print(f"Warning: missing results for {exp_name}")
            continue
        metrics = _extract_metrics(load_json(results_path))
        rows.append(
            [
                label,
                f"{metrics['dice']:.4f}",
                f"{metrics['iou']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['mae']:.4f}",
                f"{metrics['fbeta']:.4f}",
            ]
        )

    output_dir = os.path.join("outputs", "tables")
    ensure_dir(output_dir)

    csv_path = os.path.join(output_dir, "comparison_results.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Model", "Dice", "IoU", "Precision", "Recall", "MAE", "F-measure"])
        writer.writerows(rows)

    md_path = os.path.join(output_dir, "comparison_results.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("| Model | Dice | IoU | Precision | Recall | MAE | F-measure |\n")
        handle.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            handle.write("| " + " | ".join(row) + " |\n")

    print(f"Saved comparison tables to {output_dir}")


if __name__ == "__main__":
    compare_results()
