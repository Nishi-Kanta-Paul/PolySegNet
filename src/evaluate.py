from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import Config


def _ensure_shape(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    return tensor


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


def _sobel_edges(tensor: torch.Tensor) -> torch.Tensor:
    tensor = _ensure_shape(tensor)
    gx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=tensor.device,
        dtype=tensor.dtype,
    ).view(1, 1, 3, 3)
    gy = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=tensor.device,
        dtype=tensor.dtype,
    ).view(1, 1, 3, 3)

    grad_x = F.conv2d(tensor, gx, padding=1)
    grad_y = F.conv2d(tensor, gy, padding=1)
    magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
    return magnitude


class BoundaryLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = _ensure_shape(logits)
        targets = _ensure_shape(targets).float()
        probs = torch.sigmoid(logits)

        pred_edges = _sobel_edges(probs)
        target_edges = _sobel_edges(targets)
        return torch.mean(torch.abs(pred_edges - target_edges))


class CompositeLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        boundary_weight: float = 1.0,
        use_boundary_loss: bool = False,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.use_boundary_loss = use_boundary_loss
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()
        self.boundary = BoundaryLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = _ensure_shape(targets).float()
        loss = 0.0
        if self.bce_weight:
            loss = loss + self.bce_weight * self.bce(logits, targets)
        if self.dice_weight:
            loss = loss + self.dice_weight * self.dice(logits, targets)
        if self.use_boundary_loss and self.boundary_weight:
            loss = loss + self.boundary_weight * self.boundary(logits, targets)
        return loss


def build_criterion(config: Config) -> CompositeLoss:
    weights = getattr(config, "loss_weights", {})
    return CompositeLoss(
        bce_weight=float(weights.get("bce", 1.0)),
        dice_weight=float(weights.get("dice", 1.0)),
        boundary_weight=float(weights.get("boundary", 1.0)),
        use_boundary_loss=bool(getattr(config, "use_boundary_loss", False)),
    )


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
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
    beta: float = 1.0,
) -> Dict[str, float]:
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
    fbeta = (1.0 + beta2) * precision * recall / (beta2 * precision + recall + eps)
    mae = torch.mean(torch.abs(probs - masks), dim=dims)

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "fbeta": fbeta.mean().item(),
        "mae": mae.mean().item(),
    }


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Optional[Config] = None,
) -> Dict[str, float]:
    model.eval()
    meters = {
        "loss": AverageMeter(),
        "dice": AverageMeter(),
        "iou": AverageMeter(),
        "precision": AverageMeter(),
        "recall": AverageMeter(),
        "fbeta": AverageMeter(),
        "mae": AverageMeter(),
    }

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)

            metrics = calculate_metrics(logits, masks)
            batch_size = images.size(0)
            meters["loss"].update(loss.item(), batch_size)
            for key, value in metrics.items():
                meters[key].update(value, batch_size)

    return {key: meter.avg for key, meter in meters.items()}


def evaluate(cfg: Config) -> None:
    raise NotImplementedError("Evaluation pipeline not implemented yet.")
