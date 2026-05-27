from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

try:
    from .config import Config
    from .dataset import PolypDataset, build_transforms
    from .model import build_model
    from .utils import ensure_dir, get_device, overlay_segmentation, save_json, set_seed
except ImportError:
    ROOT = os.path.dirname(os.path.abspath(__file__))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from config import Config
    from dataset import PolypDataset, build_transforms
    from model import build_model
    from utils import ensure_dir, get_device, overlay_segmentation, save_json, set_seed

try:
    from skimage.metrics import hausdorff_distance as _skimage_hausdorff
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

try:
    from scipy.spatial.distance import directed_hausdorff as _directed_hausdorff
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _ensure_shape(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    return tensor


def generate_boundary_target(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    mask: [B, 1, H, W] binary float tensor (values 0.0 or 1.0)
    Returns: boundary_target [B, 1, H, W] binary float tensor
    """
    mask = _ensure_shape(mask).float()
    pad = kernel_size // 2
    with torch.no_grad():
        dilation = F.max_pool2d(mask, kernel_size, stride=1, padding=pad)
        erosion = -F.max_pool2d(-mask, kernel_size, stride=1, padding=pad)
        boundary = (dilation - erosion).clamp(0.0, 1.0)
    return boundary


class SoftDiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = _ensure_shape(logits)
        targets = _ensure_shape(targets).float()
        probs = torch.sigmoid(logits)

        dims = tuple(range(1, probs.ndim))
        intersection = torch.sum(probs * targets, dim=dims)
        denominator = torch.sum(probs + targets, dim=dims)
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        return 1.0 - dice.mean()


def _get_config_value(config: object | None, name: str, default: float | bool) -> float | bool:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _get_weight(config: object | None, name: str, default: float) -> float:
    if config is None:
        return default
    if isinstance(config, dict):
        if name in config:
            return float(config[name])
        loss_weights = config.get("loss_weights")
    else:
        if hasattr(config, name):
            return float(getattr(config, name))
        loss_weights = getattr(config, "loss_weights", None)
    if isinstance(loss_weights, dict) and name in loss_weights:
        return float(loss_weights[name])
    return float(default)


class CompositeLoss(nn.Module):
    def __init__(self, config: object | None) -> None:
        super().__init__()
        self.bce_weight = _get_weight(config, "bce_weight", _get_weight(config, "bce", 1.0))
        self.dice_weight = _get_weight(config, "dice_weight", _get_weight(config, "dice", 1.0))
        self.boundary_weight = _get_weight(
            config,
            "boundary_weight",
            _get_weight(config, "boundary", 0.5),
        )
        self.aux_boundary_weight = _get_weight(config, "aux_boundary_weight", 0.3)
        self.boundary_kernel_size = int(_get_config_value(config, "boundary_kernel_size", 3))
        self.use_boundary_loss = bool(_get_config_value(config, "use_boundary_loss", True))
        self.use_multilevel_boundary = bool(
            _get_config_value(config, "use_multilevel_boundary", True)
        )

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(eps=1e-6)

    def forward(
        self,
        outputs: dict[str, object] | torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        masks = _ensure_shape(masks).float()

        if isinstance(outputs, dict):
            mask_logits = outputs.get("mask_logits")
            boundary_logits = outputs.get("boundary_logits")
            aux_boundary_logits = outputs.get("aux_boundary_logits")
            if aux_boundary_logits is None:
                aux_boundary_logits = []
        else:
            mask_logits = outputs
            boundary_logits = None
            aux_boundary_logits = []

        if mask_logits is None:
            raise ValueError("mask_logits is required for loss computation.")

        bce_loss = self.bce(mask_logits, masks) if self.bce_weight else torch.tensor(
            0.0, device=mask_logits.device
        )
        dice_loss = self.dice(mask_logits, masks) if self.dice_weight else torch.tensor(
            0.0, device=mask_logits.device
        )

        boundary_loss = torch.tensor(0.0, device=mask_logits.device)
        aux_boundary_loss = torch.tensor(0.0, device=mask_logits.device)

        if self.use_boundary_loss:
            boundary_target = generate_boundary_target(
                masks.detach(), kernel_size=self.boundary_kernel_size
            )

            if boundary_logits is not None and self.boundary_weight:
                boundary_loss = self.bce(boundary_logits, boundary_target) + self.dice(
                    boundary_logits, boundary_target
                )

            if self.use_multilevel_boundary and aux_boundary_logits:
                per_level: List[torch.Tensor] = []
                for aux_logits in aux_boundary_logits:
                    aux_resized = F.interpolate(
                        aux_logits,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    per_level.append(
                        self.bce(aux_resized, boundary_target)
                        + self.dice(aux_resized, boundary_target)
                    )
                if per_level and self.aux_boundary_weight:
                    aux_boundary_loss = torch.stack(per_level).mean()

        total_loss = (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
            + self.boundary_weight * boundary_loss
            + self.aux_boundary_weight * aux_boundary_loss
        )

        loss_dict = {
            "total_loss": float(total_loss.detach().item()),
            "seg_bce_loss": float(bce_loss.detach().item()),
            "seg_dice_loss": float(dice_loss.detach().item()),
            "boundary_loss": float(boundary_loss.detach().item()),
            "aux_boundary_loss": float(aux_boundary_loss.detach().item()),
        }
        return total_loss, loss_dict


def build_criterion(config: Config) -> CompositeLoss:
    return CompositeLoss(config)


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


def calculate_metrics(
    outputs_or_logits: dict[str, object] | torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
    beta: float = 1.0,
) -> Dict[str, float]:
    if isinstance(outputs_or_logits, dict):
        logits = outputs_or_logits.get("mask_logits")
        if logits is None:
            raise ValueError("mask_logits is required for metric computation.")
    else:
        logits = outputs_or_logits

    logits = _ensure_shape(logits)
    masks = _ensure_shape(masks).float()
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    dims = tuple(range(1, preds.ndim))
    tp = torch.sum(preds * masks, dim=dims)
    fp = torch.sum(preds * (1.0 - masks), dim=dims)
    fn = torch.sum((1.0 - preds) * masks, dim=dims)

    eps = 1e-6
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    beta2 = beta * beta
    f_measure = (1.0 + beta2) * precision * recall / (beta2 * precision + recall + eps)
    mae = torch.mean(torch.abs(probs - masks), dim=dims)

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "mae": mae.mean().item(),
        "f_measure": f_measure.mean().item(),
    }


def _normalize_for_visuals(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if image.max() > 1.0 or image.min() < 0.0:
        min_val = float(image.min())
        max_val = float(image.max())
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image)
    return image


def _boundary_f1_score(
    pred_boundary: np.ndarray,
    target_boundary: np.ndarray,
    eps: float = 1e-6,
) -> float:
    pred = pred_boundary.astype(bool)
    target = target_boundary.astype(bool)
    pred_sum = pred.sum()
    target_sum = target.sum()
    if pred_sum == 0 and target_sum == 0:
        return 1.0
    if pred_sum == 0 or target_sum == 0:
        return 0.0
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, np.logical_not(target)).sum()
    fn = np.logical_and(np.logical_not(pred), target).sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return float((2.0 * precision * recall) / (precision + recall + eps))


def _hausdorff_distance(pred_boundary: np.ndarray, target_boundary: np.ndarray) -> float:
    pred = pred_boundary.astype(np.uint8)
    target = target_boundary.astype(np.uint8)
    pred_points = np.column_stack(np.nonzero(pred))
    target_points = np.column_stack(np.nonzero(target))

    if pred_points.size == 0 and target_points.size == 0:
        return 0.0
    if pred_points.size == 0 or target_points.size == 0:
        height, width = pred.shape
        return float(np.hypot(height, width))

    if _HAS_SKIMAGE:
        return float(_skimage_hausdorff(pred, target))
    if _HAS_SCIPY:
        forward = _directed_hausdorff(pred_points, target_points)[0]
        backward = _directed_hausdorff(target_points, pred_points)[0]
        return float(max(forward, backward))

    inv_target = (1 - target).astype(np.uint8)
    inv_pred = (1 - pred).astype(np.uint8)
    dist_to_target = cv2.distanceTransform(inv_target, cv2.DIST_L2, 3)
    dist_to_pred = cv2.distanceTransform(inv_pred, cv2.DIST_L2, 3)
    forward = float(dist_to_target[pred == 1].max())
    backward = float(dist_to_pred[target == 1].max())
    return max(forward, backward)


def _load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    if not checkpoint_path:
        raise FileNotFoundError("Checkpoint path is required for evaluation.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)


def _collect_dataset_roots(cfg: Config) -> List[str]:
    roots: List[str] = []
    raw_roots = getattr(cfg, "dataset_roots", None)
    if isinstance(raw_roots, (list, tuple)) and raw_roots:
        roots = [str(root) for root in raw_roots if str(root)]
    else:
        root = getattr(cfg, "dataset_root", "") or ""
        if isinstance(root, str) and "," in root:
            roots = [part.strip() for part in root.split(",") if part.strip()]
        elif root:
            roots = [root]
    return roots


def _dataset_name_from_root(root: str, index: int) -> str:
    name = os.path.basename(os.path.normpath(root))
    return name or f"dataset_{index:02d}"


def _mean_metric(rows: List[Dict[str, object]], key: str) -> float:
    values = [row[key] for row in rows if key in row and np.isfinite(row[key])]
    return float(np.mean(values)) if values else 0.0


def _evaluate_dataset(
    model: nn.Module,
    device: torch.device,
    cfg: Config,
    dataset_root: str,
    output_dir: str,
    transform,
) -> tuple[Dict[str, float], List[Dict[str, object]]]:
    prob_dir = os.path.join(output_dir, "prob_maps")
    mask_dir = os.path.join(output_dir, "masks")
    boundary_dir = os.path.join(output_dir, "boundaries")
    overlay_dir = os.path.join(output_dir, "overlays")
    for folder in (output_dir, prob_dir, mask_dir, boundary_dir, overlay_dir):
        ensure_dir(folder)

    dataset = PolypDataset(
        dataset_root=dataset_root,
        image_dir=cfg.image_dir,
        mask_dir=cfg.mask_dir,
        split="test",
        transform=transform,
        debug=cfg.debug,
        length=cfg.debug_samples,
        image_size=cfg.image_size,
        seed=cfg.seed,
        return_paths=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    max_steps = cfg.debug_max_steps if cfg.debug else None
    threshold = 0.5
    beta = 1.0
    eps = 1e-6
    boundary_kernel = int(getattr(cfg, "boundary_kernel_size", 3))

    per_image: List[Dict[str, object]] = []
    sample_index = 0

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if max_steps is not None and step >= max_steps:
                break

            if len(batch) == 3:
                images, masks, meta = batch
            else:
                images, masks = batch
                meta = {}

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            if isinstance(outputs, dict):
                mask_logits = outputs.get("mask_logits")
                boundary_logits = outputs.get("boundary_logits")
            else:
                mask_logits = outputs
                boundary_logits = None

            if mask_logits is None:
                raise ValueError("mask_logits is required for evaluation.")

            probs = torch.sigmoid(mask_logits)
            preds = (probs >= threshold).float()

            boundary_target = generate_boundary_target(masks, kernel_size=boundary_kernel)
            if boundary_logits is not None:
                boundary_prob = torch.sigmoid(boundary_logits)
                boundary_pred = (boundary_prob >= threshold).float()
            else:
                boundary_prob = generate_boundary_target(preds, kernel_size=boundary_kernel)
                boundary_pred = (boundary_prob >= threshold).float()

            dims = tuple(range(1, preds.ndim))
            tp = torch.sum(preds * masks, dim=dims)
            fp = torch.sum(preds * (1.0 - masks), dim=dims)
            fn = torch.sum((1.0 - preds) * masks, dim=dims)

            dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
            iou = (tp + eps) / (tp + fp + fn + eps)
            precision = (tp + eps) / (tp + fp + eps)
            recall = (tp + eps) / (tp + fn + eps)
            beta2 = beta * beta
            f_measure = (1.0 + beta2) * precision * recall / (beta2 * precision + recall + eps)
            mae = torch.mean(torch.abs(probs - masks), dim=dims)

            batch_size = images.size(0)
            image_paths = list(meta.get("image_path", [])) if isinstance(meta, dict) else []
            mask_paths = list(meta.get("mask_path", [])) if isinstance(meta, dict) else []

            for i in range(batch_size):
                image_path = image_paths[i] if i < len(image_paths) else ""
                mask_path = mask_paths[i] if i < len(mask_paths) else ""
                stem = (
                    os.path.splitext(os.path.basename(image_path))[0]
                    if image_path
                    else f"sample_{sample_index:05d}"
                )
                sample_index += 1

                if image_path and os.path.isfile(image_path):
                    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    if image_bgr is None:
                        raise FileNotFoundError(f"Image not found: {image_path}")
                    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                else:
                    image_np = images[i].detach().cpu().permute(1, 2, 0).numpy()
                    original_rgb = _normalize_for_visuals(image_np)

                original_size = (original_rgb.shape[1], original_rgb.shape[0])

                if mask_path and os.path.isfile(mask_path):
                    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask_gray is None:
                        raise FileNotFoundError(f"Mask not found: {mask_path}")
                    gt_mask = (mask_gray.astype(np.float32) / 255.0) >= 0.5
                else:
                    gt_mask = masks[i].detach().cpu().squeeze().numpy() >= 0.5

                prob_small = probs[i].detach().cpu().squeeze().numpy()
                mask_small = preds[i].detach().cpu().squeeze().numpy().astype(np.uint8)
                boundary_small = boundary_prob[i].detach().cpu().squeeze().numpy()

                prob_resized = cv2.resize(prob_small, original_size, interpolation=cv2.INTER_LINEAR)
                mask_resized = cv2.resize(mask_small, original_size, interpolation=cv2.INTER_NEAREST)
                boundary_resized = cv2.resize(boundary_small, original_size, interpolation=cv2.INTER_LINEAR)

                overlay = overlay_segmentation(original_rgb, gt_mask, mask_resized)

                cv2.imwrite(
                    os.path.join(prob_dir, f"{stem}_prob.png"),
                    np.clip(prob_resized * 255.0, 0, 255).astype(np.uint8),
                )
                cv2.imwrite(
                    os.path.join(mask_dir, f"{stem}_mask.png"),
                    (mask_resized * 255).astype(np.uint8),
                )
                cv2.imwrite(
                    os.path.join(boundary_dir, f"{stem}_boundary.png"),
                    np.clip(boundary_resized * 255.0, 0, 255).astype(np.uint8),
                )
                cv2.imwrite(
                    os.path.join(overlay_dir, f"{stem}_overlay.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                )

                boundary_pred_np = boundary_pred[i].detach().cpu().squeeze().numpy() > 0.5
                boundary_target_np = boundary_target[i].detach().cpu().squeeze().numpy() > 0.5
                boundary_f1 = _boundary_f1_score(boundary_pred_np, boundary_target_np)
                hausdorff = _hausdorff_distance(
                    boundary_pred_np.astype(np.uint8),
                    boundary_target_np.astype(np.uint8),
                )

                per_image.append(
                    {
                        "image": stem,
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "dice": float(dice[i].item()),
                        "iou": float(iou[i].item()),
                        "precision": float(precision[i].item()),
                        "recall": float(recall[i].item()),
                        "mae": float(mae[i].item()),
                        "f_measure": float(f_measure[i].item()),
                        "boundary_f1": float(boundary_f1),
                        "hausdorff": float(hausdorff),
                    }
                )

    summary = {
        "num_samples": len(per_image),
        "dice": _mean_metric(per_image, "dice"),
        "iou": _mean_metric(per_image, "iou"),
        "precision": _mean_metric(per_image, "precision"),
        "recall": _mean_metric(per_image, "recall"),
        "mae": _mean_metric(per_image, "mae"),
        "f_measure": _mean_metric(per_image, "f_measure"),
        "boundary_f1": _mean_metric(per_image, "boundary_f1"),
        "hausdorff": _mean_metric(per_image, "hausdorff"),
    }

    results_path = os.path.join(output_dir, "results.json")
    save_json(results_path, {"summary": summary, "per_image": per_image})

    csv_path = os.path.join(output_dir, "results.csv")
    fieldnames = [
        "image",
        "image_path",
        "mask_path",
        "dice",
        "iou",
        "precision",
        "recall",
        "mae",
        "f_measure",
        "boundary_f1",
        "hausdorff",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image)

    return summary, per_image


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Optional[Config] = None,
) -> tuple[float, Dict[str, float], Dict[str, float]]:
    model.eval()
    max_steps = None
    if config is not None and getattr(config, "debug", False):
        max_steps = getattr(config, "debug_max_steps", None)
    metric_meters = {
        "dice": AverageMeter(),
        "iou": AverageMeter(),
        "precision": AverageMeter(),
        "recall": AverageMeter(),
        "mae": AverageMeter(),
        "f_measure": AverageMeter(),
    }
    loss_meters = {
        "total_loss": AverageMeter(),
        "seg_bce_loss": AverageMeter(),
        "seg_dice_loss": AverageMeter(),
        "boundary_loss": AverageMeter(),
        "aux_boundary_loss": AverageMeter(),
    }

    with torch.no_grad():
        for step, (images, masks) in enumerate(dataloader):
            if max_steps is not None and step >= max_steps:
                break
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            total_loss, loss_dict = criterion(outputs, masks)

            metrics = calculate_metrics(outputs, masks)
            batch_size = images.size(0)
            loss_meters["total_loss"].update(total_loss.item(), batch_size)
            for key in loss_meters:
                if key == "total_loss":
                    continue
                loss_meters[key].update(loss_dict.get(key, 0.0), batch_size)
            for key, value in metrics.items():
                metric_meters[key].update(value, batch_size)

    avg_loss = loss_meters["total_loss"].avg
    avg_metrics = {key: meter.avg for key, meter in metric_meters.items()}
    avg_loss_dict = {key: meter.avg for key, meter in loss_meters.items()}
    return avg_loss, avg_metrics, avg_loss_dict


def evaluate(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    checkpoint_path = cfg.checkpoint or cfg.best_checkpoint or cfg.last_checkpoint
    if not checkpoint_path:
        raise FileNotFoundError("Provide --checkpoint or ensure best.pth exists.")

    base_dir = os.path.join("experiments", cfg.experiment_name)
    evaluation_root = os.path.join(base_dir, "evaluation")
    ensure_dir(evaluation_root)

    model = build_model(cfg).to(device)
    model.eval()
    _load_checkpoint(model, checkpoint_path, device)

    dataset_roots = _collect_dataset_roots(cfg)
    if not dataset_roots:
        raise ValueError("Provide --dataset-root or --dataset-roots for evaluation.")

    transform = build_transforms("val", cfg.image_size)

    per_dataset_summary: Dict[str, Dict[str, float]] = {}
    overall_per_image: List[Dict[str, object]] = []

    for index, dataset_root in enumerate(dataset_roots):
        dataset_name = _dataset_name_from_root(dataset_root, index)
        output_dir = (
            evaluation_root
            if len(dataset_roots) == 1
            else os.path.join(evaluation_root, dataset_name)
        )
        summary, per_image = _evaluate_dataset(
            model,
            device,
            cfg,
            dataset_root,
            output_dir,
            transform,
        )
        per_dataset_summary[dataset_name] = summary
        overall_per_image.extend(per_image)

    if len(dataset_roots) > 1:
        overall_summary = {
            "num_samples": len(overall_per_image),
            "dice": _mean_metric(overall_per_image, "dice"),
            "iou": _mean_metric(overall_per_image, "iou"),
            "precision": _mean_metric(overall_per_image, "precision"),
            "recall": _mean_metric(overall_per_image, "recall"),
            "mae": _mean_metric(overall_per_image, "mae"),
            "f_measure": _mean_metric(overall_per_image, "f_measure"),
            "boundary_f1": _mean_metric(overall_per_image, "boundary_f1"),
            "hausdorff": _mean_metric(overall_per_image, "hausdorff"),
        }
        summary_path = os.path.join(evaluation_root, "summary.json")
        save_json(
            summary_path,
            {"overall": overall_summary, "datasets": per_dataset_summary},
        )


if __name__ == "__main__":
    from types import SimpleNamespace

    config = SimpleNamespace(
        bce_weight=1.0,
        dice_weight=1.0,
        boundary_weight=0.5,
        aux_boundary_weight=0.3,
        boundary_kernel_size=3,
        use_boundary_loss=True,
        use_multilevel_boundary=True,
    )
    b, h, w = 2, 352, 352
    outputs = {
        "mask_logits": torch.randn(b, 1, h, w),
        "boundary_logits": torch.randn(b, 1, h, w),
        "aux_boundary_logits": [
            torch.randn(b, 1, h // 2, w // 2),
            torch.randn(b, 1, h // 4, w // 4),
            torch.randn(b, 1, h // 8, w // 8),
        ],
    }
    masks = (torch.rand(b, 1, h, w) > 0.5).float()
    criterion = CompositeLoss(config)
    total_loss, loss_dict = criterion(outputs, masks)
    print("Loss dict:", loss_dict)
    assert not torch.isnan(total_loss), "NaN in total loss"
    for key, value in loss_dict.items():
        assert not torch.isnan(torch.tensor(float(value))), f"NaN in {key}"
    metrics = calculate_metrics(outputs, masks)
    print("Metrics:", metrics)
    print("All checks passed.")
