#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import IMAGE_EXTS, build_transforms
from src.model import build_model
from src.utils import ensure_dir, get_device


@dataclass
class FailureCase:
    image_path: str
    image_rgb: np.ndarray
    gt_mask: np.ndarray
    pred_mask: np.ndarray
    error_map: np.ndarray
    dice: float
    reason: str
    descriptor: Tuple[str, str, str, str]


def _resolve_path(path: str, dataset_roots: List[str]) -> str:
    if not path:
        return path
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return path
    for root in dataset_roots:
        candidate = os.path.join(root, path)
        if os.path.exists(candidate):
            return candidate
    return path


def _list_images(folder: str) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, name)
        for name in sorted(os.listdir(folder))
        if name.lower().endswith(IMAGE_EXTS)
    ]


def _infer_mask_path(image_path: str) -> str:
    base_dir = os.path.dirname(image_path)
    for candidate in (
        base_dir.replace("images", "masks"),
        base_dir.replace("Images", "Masks"),
        base_dir.replace("image", "mask"),
    ):
        if candidate != base_dir:
            guess = os.path.join(candidate, os.path.basename(image_path))
            if os.path.isfile(guess):
                return guess
    return ""


def _index_masks(mask_dir: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    mask_paths = _list_images(mask_dir)
    name_map: Dict[str, str] = {}
    stem_map: Dict[str, str] = {}
    for path in mask_paths:
        name = os.path.basename(path)
        name_map[name] = path
        stem = os.path.splitext(name)[0]
        if stem not in stem_map:
            stem_map[stem] = path
    return name_map, stem_map


def _match_mask_path(
    image_path: str,
    mask_name_map: Dict[str, str],
    mask_stem_map: Dict[str, str],
) -> str:
    base = os.path.basename(image_path)
    if base in mask_name_map:
        return mask_name_map[base]
    stem = os.path.splitext(base)[0]
    if stem in mask_stem_map:
        return mask_stem_map[stem]
    fallback = _infer_mask_path(image_path)
    if fallback and os.path.isfile(fallback):
        return fallback
    return ""


def _load_image_mask(image_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask_bin = (mask_gray.astype(np.float32) / 255.0) >= 0.5
    return image_rgb, mask_bin.astype(np.uint8)


def _prepare_tensor(image_rgb: np.ndarray, transform) -> torch.Tensor:
    transformed = transform(image=image_rgb) if transform is not None else {"image": image_rgb}
    image = transformed["image"].astype(np.float32)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return tensor


def _build_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    defaults: Dict[str, object],
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    cfg_dict = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    cfg = SimpleNamespace(**cfg_dict)
    for key, value in defaults.items():
        if not hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.pretrained = False
    cfg.debug = False

    model = build_model(cfg).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _dice_score(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_bin = gt.astype(bool)
    pred_bin = pred.astype(bool)
    intersection = np.logical_and(gt_bin, pred_bin).sum()
    total = gt_bin.sum() + pred_bin.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def _error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt_bin = gt.astype(bool)
    pred_bin = pred.astype(bool)
    tp = pred_bin & gt_bin
    fp = pred_bin & ~gt_bin
    fn = ~pred_bin & gt_bin
    tn = ~pred_bin & ~gt_bin

    error = np.full((gt.shape[0], gt.shape[1], 3), 255, dtype=np.uint8)
    error[tp] = (0, 255, 0)
    error[fp] = (255, 0, 0)
    error[fn] = (0, 0, 255)
    error[tn] = (255, 255, 255)
    return error


def _failure_reason(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    small_ratio: float,
    contrast_thresh: float,
    blur_thresh: float,
) -> Tuple[str, Tuple[str, str, str, str]]:
    area_ratio = float(gt_mask.mean())
    if area_ratio < small_ratio:
        reason = "Small polyp"
    else:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        contrast = float(gray.std())
        if contrast < contrast_thresh:
            reason = "Low contrast"
        else:
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if lap_var < blur_thresh:
                reason = "Motion blur"
            else:
                reason = "Complex texture"

    size_bin = "small" if area_ratio < small_ratio else "large"
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    contrast = float(gray.std())
    contrast_bin = "low" if contrast < contrast_thresh else "high"
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_bin = "blur" if lap_var < blur_thresh else "sharp"
    descriptor = (reason, size_bin, contrast_bin, blur_bin)
    return reason, descriptor


def _select_diverse(cases: List[FailureCase], count: int) -> List[FailureCase]:
    if count <= 0:
        return []
    ordered = sorted(cases, key=lambda item: item.dice)
    selected: List[FailureCase] = []
    seen = set()

    for item in ordered:
        if item.descriptor not in seen:
            selected.append(item)
            seen.add(item.descriptor)
        if len(selected) >= count:
            return selected

    for item in ordered:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) >= count:
            break

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 7 failure case analysis.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-dir", required=True, help="Folder with test images.")
    parser.add_argument("--mask-dir", required=True, help="Folder with GT masks.")
    parser.add_argument("--datasets", nargs="+", default=[], help="Dataset roots.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--count", type=int, default=4, help="Number of failure cases.")
    parser.add_argument("--small-polyp-ratio", type=float, default=0.01)
    parser.add_argument("--contrast-thresh", type=float, default=25.0)
    parser.add_argument("--blur-thresh", type=float, default=80.0)
    parser.add_argument("--output", default="paper_figures/fig_failure.png")
    args = parser.parse_args()

    dataset_roots = [os.path.abspath(root) for root in args.datasets]
    image_dir = _resolve_path(args.image_dir, dataset_roots)
    mask_dir = _resolve_path(args.mask_dir, dataset_roots)

    image_paths = _list_images(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {image_dir}")

    mask_name_map, mask_stem_map = _index_masks(mask_dir)

    device = get_device(args.device)
    defaults = {
        "model_name": "bgdsf_polysegnet",
        "unified_channels": 128,
        "use_msca": True,
        "use_csaf": True,
        "use_dynamic_weighting": True,
        "use_boundary_guidance": True,
        "use_boundary_loss": True,
        "use_multilevel_boundary": True,
    }
    model = _build_model_from_checkpoint(args.checkpoint, device, defaults)
    transform = build_transforms("val", args.image_size)

    failures: List[FailureCase] = []
    with torch.no_grad():
        for image_path in image_paths:
            mask_path = _match_mask_path(image_path, mask_name_map, mask_stem_map)
            if not mask_path:
                raise FileNotFoundError(f"Mask not found for image: {image_path}")
            image_rgb, gt_mask = _load_image_mask(image_path, mask_path)
            h, w = gt_mask.shape

            tensor = _prepare_tensor(image_rgb, transform).to(device)
            outputs = model(tensor)
            mask_logits = outputs.get("mask_logits")
            if mask_logits is None:
                raise ValueError("mask_logits is required for failure analysis.")

            pred_prob = torch.sigmoid(mask_logits).squeeze(0).squeeze(0).cpu().numpy()
            pred_prob = cv2.resize(pred_prob, (w, h), interpolation=cv2.INTER_LINEAR)
            pred_mask = (pred_prob >= 0.5).astype(np.uint8)

            dice = _dice_score(gt_mask, pred_mask)
            error = _error_map(gt_mask, pred_mask)
            reason, descriptor = _failure_reason(
                image_rgb,
                gt_mask,
                args.small_polyp_ratio,
                args.contrast_thresh,
                args.blur_thresh,
            )

            failures.append(
                FailureCase(
                    image_path=image_path,
                    image_rgb=image_rgb,
                    gt_mask=gt_mask,
                    pred_mask=pred_mask,
                    error_map=error,
                    dice=dice,
                    reason=reason,
                    descriptor=descriptor,
                )
            )

    if not failures:
        raise ValueError("No samples were processed.")

    selected = _select_diverse(failures, args.count)
    rows = len(selected)
    cols = 4

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Input", "GT Mask", "Predicted Mask", "Error Map"]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=9)

    for row_idx, item in enumerate(selected):
        axes[row_idx, 0].imshow(item.image_rgb)
        axes[row_idx, 1].imshow(item.gt_mask, cmap="gray")
        axes[row_idx, 2].imshow(item.pred_mask, cmap="gray")
        axes[row_idx, 3].imshow(item.error_map)

        reason_text = f"{os.path.basename(item.image_path)}\n{item.reason} | Dice={item.dice:.3f}"
        axes[row_idx, 0].set_xlabel(reason_text, fontsize=8)

        for col_idx in range(cols):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    plt.tight_layout()
    output_path = os.path.abspath(args.output)
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
