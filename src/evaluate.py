from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

try:
    from .config import Config
except ImportError:
    ROOT = os.path.dirname(os.path.abspath(__file__))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from config import Config


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
        self.multi_boundary_weight = _get_weight(config, "multi_boundary_weight", 0.2)
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
        multi_boundary_loss = torch.tensor(0.0, device=mask_logits.device)

        if self.use_boundary_loss:
            boundary_target = generate_boundary_target(
                masks.detach(), kernel_size=self.boundary_kernel_size
            )

            if self.boundary_weight:
                boundary_loss = self.dice(mask_logits, boundary_target)

            if boundary_logits is not None and self.aux_boundary_weight:
                aux_boundary_loss = self.bce(boundary_logits, boundary_target) + self.dice(
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
                if per_level and self.multi_boundary_weight:
                    multi_boundary_loss = torch.stack(per_level).mean()

        total_loss = (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
            + self.boundary_weight * boundary_loss
            + self.aux_boundary_weight * aux_boundary_loss
            + self.multi_boundary_weight * multi_boundary_loss
        )

        loss_dict = {
            "total_loss": float(total_loss.detach().item()),
            "bce_loss": float(bce_loss.detach().item()),
            "dice_loss": float(dice_loss.detach().item()),
            "boundary_loss": float(boundary_loss.detach().item()),
            "aux_boundary_loss": float(aux_boundary_loss.detach().item()),
            "multi_boundary_loss": float(multi_boundary_loss.detach().item()),
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
        "bce_loss": AverageMeter(),
        "dice_loss": AverageMeter(),
        "boundary_loss": AverageMeter(),
        "aux_boundary_loss": AverageMeter(),
        "multi_boundary_loss": AverageMeter(),
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
    raise NotImplementedError("Evaluation pipeline not implemented yet.")


if __name__ == "__main__":
    from types import SimpleNamespace

    config = SimpleNamespace(
        bce_weight=1.0,
        dice_weight=1.0,
        boundary_weight=0.5,
        aux_boundary_weight=0.3,
        multi_boundary_weight=0.2,
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
