import json
import os
import random
from typing import Any, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preferred: str) -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(path: str, data: Any, indent: int = 2) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _to_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 0.5).astype(np.uint8)


def overlay_segmentation(
    image: np.ndarray,
    mask: np.ndarray,
    prediction: Optional[np.ndarray] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    base = _to_uint8_image(image)
    overlay = base.astype(np.float32)

    mask_bin = _to_binary_mask(mask)
    if mask_bin.any():
        overlay[mask_bin == 1] = (
            (1 - alpha) * overlay[mask_bin == 1]
            + alpha * np.array([255, 0, 0], dtype=np.float32)
        )

    if prediction is not None:
        pred_bin = _to_binary_mask(prediction)
        if pred_bin.any():
            overlay[pred_bin == 1] = (
                (1 - alpha) * overlay[pred_bin == 1]
                + alpha * np.array([0, 0, 255], dtype=np.float32)
            )

    return overlay.astype(np.uint8)
