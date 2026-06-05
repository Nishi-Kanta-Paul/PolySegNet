#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from src.dataset import IMAGE_EXTS, PolypDataset, build_transforms
from src.evaluate import generate_boundary_target
from src.evaluate import _boundary_f1_score, _hausdorff_distance, _surface_metrics
from src.utils import ensure_dir, save_json


def _list_images(folder: str) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, name)
        for name in sorted(os.listdir(folder))
        if name.lower().endswith(IMAGE_EXTS)
    ]


def _index_predictions(pred_dir: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    paths = _list_images(pred_dir)
    name_map: Dict[str, str] = {}
    stem_map: Dict[str, str] = {}
    for path in paths:
        name = os.path.basename(path)
        name_map[name] = path
        stem = os.path.splitext(name)[0]
        if stem not in stem_map:
            stem_map[stem] = path
    return name_map, stem_map


def _match_prediction(
    image_path: str,
    pred_dir: str,
    pred_name_map: Dict[str, str],
    pred_stem_map: Dict[str, str],
    pred_suffix: str | None,
) -> str:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    if pred_suffix:
        candidate = os.path.join(pred_dir, f"{stem}{pred_suffix}")
        if os.path.isfile(candidate):
            return candidate
    base = os.path.basename(image_path)
    if base in pred_name_map:
        return pred_name_map[base]
    if stem in pred_stem_map:
        return pred_stem_map[stem]
    return ""


def _load_pred_mask(path: str, target_size: Tuple[int, int]) -> np.ndarray:
    pred = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if pred is None:
        raise FileNotFoundError(f"Prediction not found: {path}")
    pred = pred.astype(np.float32)
    if pred.max() > 1.0:
        pred = pred / 255.0
    if pred.shape[:2] != target_size:
        pred = cv2.resize(pred, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(pred, 0.0, 1.0)


def _compute_metrics(gt: np.ndarray, pred_prob: np.ndarray) -> Dict[str, float]:
    eps = 1e-6
    gt_bin = gt.astype(np.float32)
    pred_bin = (pred_prob >= 0.5).astype(np.float32)

    tp = float((pred_bin * gt_bin).sum())
    fp = float((pred_bin * (1.0 - gt_bin)).sum())
    fn = float(((1.0 - pred_bin) * gt_bin).sum())

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    beta = 1.0
    beta2 = beta * beta
    f_measure = (1.0 + beta2) * precision * recall / (beta2 * precision + recall + eps)
    mae = float(np.mean(np.abs(pred_prob - gt_bin)))

    gt_tensor = torch.from_numpy(gt_bin).unsqueeze(0).unsqueeze(0)
    pred_tensor = torch.from_numpy(pred_bin).unsqueeze(0).unsqueeze(0)
    boundary_target = generate_boundary_target(gt_tensor, kernel_size=3).squeeze().numpy() > 0.5
    boundary_pred = generate_boundary_target(pred_tensor, kernel_size=3).squeeze().numpy() > 0.5

    mask_boundary_f1 = _boundary_f1_score(boundary_pred, boundary_target)
    mask_hd = _hausdorff_distance(boundary_pred.astype(np.uint8), boundary_target.astype(np.uint8))
    mask_asd, mask_assd, mask_hd95 = _surface_metrics(
        boundary_pred.astype(np.uint8),
        boundary_target.astype(np.uint8),
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f_measure": float(f_measure),
        "mae": float(mae),
        "mask_boundary_f1": float(mask_boundary_f1),
        "mask_hd": float(mask_hd),
        "mask_hd95": float(mask_hd95),
        "mask_asd": float(mask_asd),
        "mask_assd": float(mask_assd),
    }


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline predictions.")
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--pred-suffix", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--split-dir", default=None)
    args = parser.parse_args()

    dataset = PolypDataset(
        dataset_root=args.dataset_root,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        split=args.split,
        transform=build_transforms("val", args.image_size),
        image_size=args.image_size,
        return_paths=True,
        split_dir=args.split_dir,
    )

    pred_name_map, pred_stem_map = _index_predictions(args.pred_dir)
    if not pred_name_map and not pred_stem_map:
        raise FileNotFoundError(f"No prediction images found in: {args.pred_dir}")

    per_image: List[Dict[str, float | str]] = []
    for idx in range(len(dataset)):
        _image, mask_tensor, meta = dataset[idx]
        image_path = meta.get("image_path", "")
        stem = os.path.splitext(os.path.basename(image_path))[0] if image_path else f"sample_{idx:05d}"

        pred_path = _match_prediction(
            image_path,
            args.pred_dir,
            pred_name_map,
            pred_stem_map,
            args.pred_suffix,
        )
        if not pred_path:
            raise FileNotFoundError(f"Prediction not found for image: {image_path}")

        gt = mask_tensor.squeeze(0).numpy()
        pred_prob = _load_pred_mask(pred_path, gt.shape)

        metrics = _compute_metrics(gt, pred_prob)
        metrics["image"] = stem
        metrics["image_path"] = image_path
        metrics["pred_path"] = pred_path
        per_image.append(metrics)

    summary = {
        "num_samples": len(per_image),
        "dice": _mean([item["dice"] for item in per_image]),
        "iou": _mean([item["iou"] for item in per_image]),
        "precision": _mean([item["precision"] for item in per_image]),
        "recall": _mean([item["recall"] for item in per_image]),
        "f_measure": _mean([item["f_measure"] for item in per_image]),
        "mae": _mean([item["mae"] for item in per_image]),
        "mask_boundary_f1": _mean([item["mask_boundary_f1"] for item in per_image]),
        "mask_hd": _mean([item["mask_hd"] for item in per_image]),
        "mask_hd95": _mean([item["mask_hd95"] for item in per_image]),
        "mask_asd": _mean([item["mask_asd"] for item in per_image]),
        "mask_assd": _mean([item["mask_assd"] for item in per_image]),
    }

    dataset_name = os.path.basename(os.path.normpath(args.dataset_root)) or "dataset"
    ensure_dir(args.output_dir)
    save_json(
        os.path.join(args.output_dir, "results.json"),
        {
            "method_name": args.method_name,
            "dataset_name": dataset_name,
            "pred_dir": args.pred_dir,
            "summary": summary,
            "per_image": per_image,
        },
    )

    csv_path = os.path.join(args.output_dir, "results.csv")
    fieldnames = [
        "image",
        "image_path",
        "pred_path",
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
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image)

    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
