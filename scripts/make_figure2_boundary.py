#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import IMAGE_EXTS, build_transforms
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
    return image_path


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


def _predict_boundary(model: torch.nn.Module, tensor: torch.Tensor) -> np.ndarray:
    outputs = model(tensor)
    boundary_logits = outputs.get("boundary_logits")
    if boundary_logits is None:
        raise ValueError("boundary_logits is required for boundary visualization.")
    probs = torch.sigmoid(boundary_logits).squeeze(0).squeeze(0).cpu().numpy()
    return probs


def _extract_gt_boundary(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    boundary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    return (boundary > 0).astype(np.uint8)


def _error_overlay(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    overlay = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)
    overlay[tp] = (0, 255, 0)
    overlay[fp] = (255, 0, 0)
    overlay[fn] = (0, 0, 255)
    return overlay


def _build_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    defaults: dict,
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
    parser = argparse.ArgumentParser(description="Figure 2 boundary visualization.")
    parser.add_argument("--datasets", nargs="+", default=[], help="Dataset roots.")
    parser.add_argument("--image-dir", required=True, help="Folder with input images.")
    parser.add_argument("--mask-dir", required=True, help="Folder with GT masks.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--output", default="paper_figures/fig_boundary.png")
    parser.add_argument("--indices", default="0,1,2,3")
    args = parser.parse_args()

    dataset_roots = [os.path.abspath(root) for root in args.datasets]
    image_dir = _resolve_path(args.image_dir, dataset_roots)
    mask_dir = _resolve_path(args.mask_dir, dataset_roots)

    image_paths = _list_images(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {image_dir}")

    indices = [int(idx) for idx in args.indices.split(",") if idx.strip()]
    if len(indices) != 4:
        raise ValueError("Provide exactly 4 comma-separated indices via --indices.")

    selected = [image_paths[i] for i in indices]
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

    fig, axes = plt.subplots(4, 5, figsize=(14, 12))
    col_titles = ["Input", "GT Mask", "GT Boundary", "Predicted Y_b", "Error Overlay"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10)

    for row_idx, image_path in enumerate(selected):
        mask_path = os.path.join(mask_dir, os.path.basename(image_path))
        if not os.path.isfile(mask_path):
            mask_path = _infer_mask_path(image_path)
        image_rgb, gt_mask = _load_image_mask(image_path, mask_path)
        h, w = gt_mask.shape

        tensor = _prepare_tensor(image_rgb, transform).to(device)
        with torch.no_grad():
            pred_prob = _predict_boundary(model, tensor)
        pred_prob = cv2.resize(pred_prob, (w, h), interpolation=cv2.INTER_LINEAR)
        pred_bin = (pred_prob >= 0.5).astype(np.uint8)

        gt_boundary = _extract_gt_boundary(gt_mask)
        error = _error_overlay(gt_boundary, pred_bin)

        row_axes = axes[row_idx]
        row_axes[0].imshow(image_rgb)
        row_axes[1].imshow(gt_mask, cmap="gray")
        row_axes[2].imshow(gt_boundary, cmap="gray")
        row_axes[3].imshow(pred_bin, cmap="gray")
        row_axes[4].imshow(error)

        for ax in row_axes:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    output_path = os.path.abspath(args.output)
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
