#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.dataset import build_transforms
from src.model import build_model
from src.train import _apply_frequency_aug
from src.utils import ensure_dir, get_device

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


def _load_image(image_path: str) -> np.ndarray:
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _prepare_tensor(image_rgb: np.ndarray, transform) -> torch.Tensor:
    transformed = transform(image=image_rgb) if transform is not None else {"image": image_rgb}
    image = transformed["image"].astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return tensor


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    array = array.transpose(1, 2, 0)
    array = array * IMAGENET_STD + IMAGENET_MEAN
    array = np.clip(array, 0.0, 1.0)
    return (array * 255.0).astype(np.uint8)


def _predict_mask(model: torch.nn.Module, tensor: torch.Tensor) -> np.ndarray:
    outputs = model(tensor)
    mask_logits = outputs["mask_logits"]
    probs = torch.sigmoid(mask_logits).squeeze(0).squeeze(0).cpu().numpy()
    return probs


def _overlay_prediction(image_rgb: np.ndarray, prob: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    mask = (prob >= 0.5).astype(np.uint8)
    overlay = image_rgb.astype(np.float32) / 255.0
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    overlay[mask == 1] = (1.0 - alpha) * overlay[mask == 1] + alpha * red
    return (np.clip(overlay, 0.0, 1.0) * 255.0).astype(np.uint8)


def _fft_amplitude(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    amplitude = np.log1p(np.abs(spectrum))
    amplitude = amplitude - amplitude.min()
    if amplitude.max() > 0:
        amplitude = amplitude / amplitude.max()
    return amplitude


def _spectrum_heatmap(amplitude: np.ndarray) -> np.ndarray:
    heat = cv2.applyColorMap((amplitude * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


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
    parser = argparse.ArgumentParser(description="Figure 6 frequency augmentation effect.")
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", default=[])
    parser.add_argument("--checkpoint-without", required=True)
    parser.add_argument("--checkpoint-with", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--output", default="paper_figures/fig_freqaug.png")
    args = parser.parse_args()

    dataset_roots = [os.path.abspath(root) for root in args.datasets]
    image_paths = [_resolve_path(path, dataset_roots) for path in args.images]

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

    model_without = _build_model_from_checkpoint(args.checkpoint_without, device, defaults)
    model_with = _build_model_from_checkpoint(args.checkpoint_with, device, defaults)

    transform = build_transforms("val", args.image_size)

    cols = len(image_paths)
    fig, axes = plt.subplots(4, cols, figsize=(cols * 3.0, 10))
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    row_titles = [
        "Original",
        "FreqAug",
        "Pred (no FreqAug)",
        "Pred (with FreqAug)",
    ]
    for i, title in enumerate(row_titles):
        axes[i, 0].set_ylabel(title, fontsize=10, rotation=90)

    with torch.no_grad():
        for col_idx, image_path in enumerate(image_paths):
            image_rgb = _load_image(image_path)
            tensor = _prepare_tensor(image_rgb, transform).to(device)

            aug_tensor = _apply_frequency_aug(tensor)
            aug_rgb = _denormalize(aug_tensor)

            prob_without = _predict_mask(model_without, tensor)
            prob_with = _predict_mask(model_with, tensor)

            prob_without = cv2.resize(
                prob_without, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR
            )
            prob_with = cv2.resize(
                prob_with, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR
            )

            overlay_without = _overlay_prediction(image_rgb, prob_without)
            overlay_with = _overlay_prediction(image_rgb, prob_with)

            ax0 = axes[0, col_idx]
            ax1 = axes[1, col_idx]
            ax2 = axes[2, col_idx]
            ax3 = axes[3, col_idx]

            ax0.imshow(image_rgb)
            ax1.imshow(aug_rgb)
            ax2.imshow(overlay_without)
            ax3.imshow(overlay_with)

            for ax in (ax0, ax1, ax2, ax3):
                ax.set_xticks([])
                ax.set_yticks([])

            amp_before = _fft_amplitude(image_rgb)
            amp_after = _fft_amplitude(aug_rgb)

            inset_before = inset_axes(ax0, width="32%", height="32%", loc="upper right")
            inset_before.imshow(_spectrum_heatmap(amp_before))
            inset_before.set_xticks([])
            inset_before.set_yticks([])

            inset_after = inset_axes(ax1, width="32%", height="32%", loc="upper right")
            inset_after.imshow(_spectrum_heatmap(amp_after))
            inset_after.set_xticks([])
            inset_after.set_yticks([])

    plt.tight_layout()
    output_path = os.path.abspath(args.output)
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
