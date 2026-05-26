#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import build_transforms
from src.model import build_model
from src.utils import ensure_dir, get_device


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


def _load_cases(args: argparse.Namespace) -> List[Dict[str, str]]:
    cases: List[Dict[str, str]] = []
    if args.cases_file:
        with open(args.cases_file, "r", encoding="utf-8") as handle:
            cases = json.load(handle)
    for entry in args.case:
        parts = [part.strip() for part in entry.split("|")]
        if len(parts) != 3:
            raise ValueError("Each --case must be 'label|image_path|mask_path'.")
        cases.append({"label": parts[0], "image": parts[1], "mask": parts[2]})
    if len(cases) != 5:
        raise ValueError("Provide exactly 5 cases via --cases-file or repeated --case.")
    return cases


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
    if image.max() > 1.0:
        image = image / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return tensor


def _predict_mask(model: torch.nn.Module, tensor: torch.Tensor) -> np.ndarray:
    outputs = model(tensor)
    mask_logits = outputs["mask_logits"]
    probs = torch.sigmoid(mask_logits).squeeze(0).squeeze(0).cpu().numpy()
    return probs


def _overlay_masks(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray | None = None,
    alpha: float = 0.5,
) -> np.ndarray:
    overlay = image_rgb.astype(np.float32) / 255.0
    green = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    if gt_mask is not None:
        mask = gt_mask.astype(bool)
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * green
    if pred_mask is not None:
        mask = pred_mask.astype(bool)
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * red

    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return overlay


def _boundary_crop_box(mask: np.ndarray, crop_size: int | None = None) -> Tuple[int, int, int, int]:
    h, w = mask.shape
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilation = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    erosion = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    boundary = (dilation - erosion) > 0

    ys, xs = np.where(boundary)
    if len(xs) == 0:
        cx, cy = w // 2, h // 2
    else:
        cx = int(np.median(xs))
        cy = int(np.median(ys))

    if crop_size is None:
        crop_size = int(0.2 * min(h, w))
        crop_size = max(crop_size, 32)
    half = crop_size // 2

    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(w, x0 + crop_size)
    y1 = min(h, y0 + crop_size)

    x0 = max(0, x1 - crop_size)
    y0 = max(0, y1 - crop_size)
    return x0, y0, x1 - x0, y1 - y0


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure 1 qualitative comparison (BGD-SF PolySegNet)."
    )
    parser.add_argument("--datasets", nargs="+", default=[], help="Dataset roots.")
    parser.add_argument("--cases-file", default=None, help="JSON list of 5 cases.")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case spec: label|image_path|mask_path (repeat 5 times).",
    )
    parser.add_argument("--prev-checkpoint", required=True)
    parser.add_argument("--ours-checkpoint", required=True)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="paper_figures/fig_qualitative.png")
    parser.add_argument("--crop-size", type=int, default=None)
    args = parser.parse_args()

    cases = _load_cases(args)

    dataset_roots = [os.path.abspath(root) for root in args.datasets]
    resolved_cases = []
    for case in cases:
        image_path = _resolve_path(case["image"], dataset_roots)
        mask_path = _resolve_path(case["mask"], dataset_roots)
        resolved_cases.append({"label": case["label"], "image": image_path, "mask": mask_path})

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

    prev_model = _build_model_from_checkpoint(args.prev_checkpoint, device, defaults)
    ours_model = _build_model_from_checkpoint(args.ours_checkpoint, device, defaults)

    image_size = args.image_size or 352
    transform = build_transforms("val", image_size)

    fig, axes = plt.subplots(5, 4, figsize=(12, 14))
    col_titles = ["Input", "GT Mask", "Previous PolySegNet", "BGD-SF PolySegNet (Ours)"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10)

    for row_idx, case in enumerate(resolved_cases):
        image_rgb, gt_mask = _load_image_mask(case["image"], case["mask"])
        original_h, original_w = gt_mask.shape

        tensor = _prepare_tensor(image_rgb, transform).to(device)
        with torch.no_grad():
            prev_prob = _predict_mask(prev_model, tensor)
            ours_prob = _predict_mask(ours_model, tensor)

        prev_prob = cv2.resize(prev_prob, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        ours_prob = cv2.resize(ours_prob, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        prev_mask = (prev_prob >= 0.5).astype(np.uint8)
        ours_mask = (ours_prob >= 0.5).astype(np.uint8)

        overlay_gt = _overlay_masks(image_rgb, gt_mask, None, alpha=0.5)
        overlay_prev = _overlay_masks(image_rgb, gt_mask, prev_mask, alpha=0.5)
        overlay_ours = _overlay_masks(image_rgb, gt_mask, ours_mask, alpha=0.5)

        x0, y0, bw, bh = _boundary_crop_box(gt_mask, args.crop_size)

        row_axes = axes[row_idx]
        row_axes[0].imshow(image_rgb)
        row_axes[1].imshow(overlay_gt)
        row_axes[2].imshow(overlay_prev)
        row_axes[3].imshow(overlay_ours)

        for ax in row_axes:
            ax.add_patch(
                plt.Rectangle(
                    (x0, y0),
                    bw,
                    bh,
                    linewidth=1.5,
                    edgecolor="red",
                    facecolor="none",
                )
            )
            ax.set_xticks([])
            ax.set_yticks([])

        row_axes[0].set_ylabel(case["label"], fontsize=9, rotation=0, labelpad=40)

    plt.tight_layout()
    output_path = os.path.abspath(args.output)
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
