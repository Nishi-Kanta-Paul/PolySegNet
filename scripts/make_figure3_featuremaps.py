#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def _normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    min_val = float(arr.min())
    max_val = float(arr.max())
    if max_val - min_val < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_val) / (max_val - min_val)


def _tensor_to_heatmap(
    tensor: torch.Tensor,
    target_size: int = 352,
    apply_sigmoid: bool = False,
) -> np.ndarray:
    if apply_sigmoid:
        tensor = torch.sigmoid(tensor)
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim == 3:
        tensor = tensor.mean(dim=0)
    array = tensor.detach().cpu().numpy()
    array = _normalize_map(array)
    array = cv2.resize(array, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap((array * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


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
    parser = argparse.ArgumentParser(description="Figure 3 feature map visualization.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--output", default="paper_figures/fig_featuremaps.png")
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
    model = _build_model_from_checkpoint(args.checkpoint, device, defaults)

    hooks = []
    activations: Dict[str, torch.Tensor] = {}

    def _hook(name: str):
        def handler(_module, _inputs, output):
            activations[name] = output.detach()
        return handler

    hooks.append(model.bottleneck.register_forward_hook(_hook("b")))
    if len(model.projections) >= 5:
        hooks.append(model.projections[-1].register_forward_hook(_hook("e5")))

    for idx, block in enumerate(model.decoder_blocks):
        stage = 4 - idx
        if hasattr(block, "gate_out"):
            hooks.append(block.gate_out.register_forward_hook(_hook(f"gate_d{stage}")))

    if model.mbgh is not None:
        hooks.append(model.mbgh.boundary_out.register_forward_hook(_hook("boundary_logits")))

    transform = build_transforms("val", args.image_size)

    rows = len(image_paths)
    cols = 9
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "Input",
        "E5",
        "B",
        "Gate@D4",
        "Gate@D3",
        "Gate@D2",
        "Gate@D1",
        "Y_b",
        "Y_hat",
    ]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=9)

    with torch.no_grad():
        for row_idx, image_path in enumerate(image_paths):
            activations.clear()
            image_rgb = _load_image(image_path)
            tensor = _prepare_tensor(image_rgb, transform).to(device)
            outputs = model(tensor)

            y_b = outputs.get("boundary_logits")
            y_hat = outputs.get("mask_logits")
            if y_b is None or y_hat is None:
                raise ValueError("boundary_logits and mask_logits are required.")

            display_maps = [
                image_rgb,
                _tensor_to_heatmap(activations.get("e5", y_b), args.image_size),
                _tensor_to_heatmap(activations.get("b", y_b), args.image_size),
                _tensor_to_heatmap(activations.get("gate_d4", y_b), args.image_size, True),
                _tensor_to_heatmap(activations.get("gate_d3", y_b), args.image_size, True),
                _tensor_to_heatmap(activations.get("gate_d2", y_b), args.image_size, True),
                _tensor_to_heatmap(activations.get("gate_d1", y_b), args.image_size, True),
                _tensor_to_heatmap(activations.get("boundary_logits", y_b), args.image_size, True),
                _tensor_to_heatmap(y_hat, args.image_size, True),
            ]

            for col_idx in range(cols):
                axes[row_idx, col_idx].imshow(display_maps[col_idx])
                axes[row_idx, col_idx].set_xticks([])
                axes[row_idx, col_idx].set_yticks([])

    for hook in hooks:
        hook.remove()

    plt.tight_layout()
    output_path = os.path.abspath(args.output)
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
