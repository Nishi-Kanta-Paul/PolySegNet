from __future__ import annotations

import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from .config import Config
    from .dataset import IMAGE_EXTS, build_transforms
    from .model import build_model
    from .utils import ensure_dir, get_device, set_seed
except ImportError:
    ROOT = os.path.dirname(os.path.abspath(__file__))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from config import Config
    from dataset import IMAGE_EXTS, build_transforms
    from model import build_model
    from utils import ensure_dir, get_device, set_seed


def _resolve_path(path: str, dataset_root: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return path
    if dataset_root:
        candidate = os.path.join(dataset_root, path)
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


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    if not checkpoint_path:
        raise FileNotFoundError("Checkpoint path is required for explainability.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)


def _prepare_image(
    image_path: str,
    transform,
) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_size = (image_rgb.shape[1], image_rgb.shape[0])

    transformed = transform(image=image_rgb) if transform is not None else {"image": image_rgb}
    image = transformed["image"].astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return tensor, image_rgb, original_size


def _normalize_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    heatmap = F.relu(heatmap)
    min_val = heatmap.min()
    max_val = heatmap.max()
    if (max_val - min_val) < 1e-6:
        return torch.zeros_like(heatmap)
    return (heatmap - min_val) / (max_val - min_val)


def run_explainability(cfg: Config) -> None:
    """
    Simple feature-based visualization (not full Grad-CAM).
    Uses the last encoder feature map averaged across channels.
    """
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    checkpoint_path = cfg.checkpoint or cfg.best_checkpoint
    if not checkpoint_path:
        raise FileNotFoundError("Provide --checkpoint or ensure best.pth exists.")

    image_path = _resolve_path(cfg.image, cfg.dataset_root)
    image_dir = _resolve_path(cfg.image_dir, cfg.dataset_root)

    image_list: List[str] = []
    if image_path:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image_list = [image_path]
    elif image_dir:
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        image_list = _list_images(image_dir)
        if not image_list:
            raise FileNotFoundError(f"No images found in: {image_dir}")
    else:
        raise ValueError("Provide --image or --image_dir for explainability.")

    output_dir = os.path.join("experiments", cfg.experiment_name, "explainability")
    ensure_dir(output_dir)

    model = build_model(cfg).to(device)
    model.eval()
    _load_checkpoint(model, checkpoint_path, device)

    transform = build_transforms("val", cfg.image_size)

    with torch.no_grad():
        for image_path in image_list:
            tensor, original_rgb, original_size = _prepare_image(image_path, transform)
            tensor = tensor.to(device)

            features = model.encoder(tensor)
            heatmap = torch.mean(features[-1], dim=1, keepdim=True)
            heatmap = _normalize_heatmap(heatmap)
            heatmap = F.interpolate(
                heatmap,
                size=(original_size[1], original_size[0]),
                mode="bilinear",
                align_corners=False,
            )
            heatmap_np = heatmap.squeeze().cpu().numpy()

            heatmap_uint8 = np.clip(heatmap_np * 255.0, 0, 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(
                cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR),
                0.6,
                heatmap_color,
                0.4,
                0,
            )

            stem = os.path.splitext(os.path.basename(image_path))[0]
            cv2.imwrite(os.path.join(output_dir, f"{stem}_heatmap.png"), heatmap_color)
            cv2.imwrite(os.path.join(output_dir, f"{stem}_overlay.png"), overlay)
