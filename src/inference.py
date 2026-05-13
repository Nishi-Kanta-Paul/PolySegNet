from __future__ import annotations

import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
import torch

try:
    from .config import Config
    from .dataset import IMAGE_EXTS, build_transforms
    from .model import build_model
    from .utils import ensure_dir, get_device, overlay_segmentation, set_seed
except ImportError:
    ROOT = os.path.dirname(os.path.abspath(__file__))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from config import Config
    from dataset import IMAGE_EXTS, build_transforms
    from model import build_model
    from utils import ensure_dir, get_device, overlay_segmentation, set_seed


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
        raise FileNotFoundError("Checkpoint path is required for inference.")
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


def _save_outputs(
    output_dir: str,
    stem: str,
    original_rgb: np.ndarray,
    prob_map: np.ndarray,
    mask_bin: np.ndarray,
) -> None:
    ensure_dir(output_dir)

    prob_uint8 = np.clip(prob_map * 255.0, 0, 255).astype(np.uint8)
    mask_uint8 = (mask_bin * 255).astype(np.uint8)
    overlay = overlay_segmentation(original_rgb, mask_bin)

    cv2.imwrite(
        os.path.join(output_dir, f"{stem}_prob.png"),
        prob_uint8,
    )
    cv2.imwrite(
        os.path.join(output_dir, f"{stem}_mask.png"),
        mask_uint8,
    )
    cv2.imwrite(
        os.path.join(output_dir, f"{stem}_overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )


def run_inference(cfg: Config) -> None:
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
        raise ValueError("Provide --image or --image_dir for inference.")

    output_dir = os.path.join("experiments", cfg.experiment_name, "inference")
    ensure_dir(output_dir)

    model = build_model(cfg).to(device)
    model.eval()
    _load_checkpoint(model, checkpoint_path, device)

    transform = build_transforms("val", cfg.image_size)

    with torch.no_grad():
        for image_path in image_list:
            tensor, original_rgb, original_size = _prepare_image(image_path, transform)
            tensor = tensor.to(device)
            logits = model(tensor)
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()

            prob_resized = cv2.resize(
                probs,
                original_size,
                interpolation=cv2.INTER_LINEAR,
            )
            mask_bin = (prob_resized >= 0.5).astype(np.uint8)

            stem = os.path.splitext(os.path.basename(image_path))[0]
            _save_outputs(output_dir, stem, original_rgb, prob_resized, mask_bin)
