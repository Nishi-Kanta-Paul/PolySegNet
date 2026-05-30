from __future__ import annotations

import os
import sys
from dataclasses import asdict
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from .config import Config, build_arg_parser, config_from_args
    from .dataset import PolypDataset, build_transforms
    from .evaluate import AverageMeter, build_criterion, evaluate_model
    from .model import build_model
    from .utils import ensure_dir, get_device, overlay_segmentation, save_json, set_seed
except ImportError:
    ROOT = os.path.dirname(os.path.abspath(__file__))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    from config import Config, build_arg_parser, config_from_args
    from dataset import PolypDataset, build_transforms
    from evaluate import AverageMeter, build_criterion, evaluate_model
    from model import build_model
    from utils import ensure_dir, get_device, overlay_segmentation, save_json, set_seed


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


def _save_visuals(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: str,
    epoch: int,
    max_samples: int = 3,
) -> None:
    ensure_dir(output_dir)
    model.eval()
    saved = 0
    with torch.no_grad():
        for images, masks in dataloader:
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
                raise ValueError("mask_logits is required for visualization.")
            probs = torch.sigmoid(mask_logits)
            for idx in range(images.size(0)):
                image_np = images[idx].detach().cpu().permute(1, 2, 0).numpy()
                mask_np = masks[idx].detach().cpu().squeeze(0).numpy()
                pred_np = probs[idx].detach().cpu().squeeze(0).numpy()
                image_np = _normalize_for_visuals(image_np)
                overlay = overlay_segmentation(image_np, mask_np, pred_np)
                output_path = os.path.join(
                    output_dir, f"epoch_{epoch:03d}_sample_{saved}.png"
                )
                cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                if boundary_logits is not None:
                    boundary_dir = os.path.join(output_dir, "boundary")
                    ensure_dir(boundary_dir)
                    boundary_np = (
                        torch.sigmoid(boundary_logits[idx])
                        .detach()
                        .cpu()
                        .squeeze(0)
                        .numpy()
                    )
                    boundary_np = np.clip(boundary_np * 255.0, 0, 255).astype(np.uint8)
                    boundary_heatmap = cv2.applyColorMap(boundary_np, cv2.COLORMAP_JET)
                    boundary_path = os.path.join(
                        boundary_dir, f"epoch_{epoch:03d}_sample_{saved}.png"
                    )
                    cv2.imwrite(boundary_path, boundary_heatmap)
                saved += 1
                if saved >= max_samples:
                    return


def _save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    epoch: int,
    best_dice: float,
    cfg: Config,
) -> None:
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_dice": best_dice,
        "config": asdict(cfg),
        "use_dynamic_weighting": getattr(cfg, "use_dynamic_weighting", True),
        "use_boundary_guidance": getattr(cfg, "use_boundary_guidance", True),
        "use_multilevel_boundary": getattr(cfg, "use_multilevel_boundary", True),
        "use_mbgh": getattr(cfg, "use_mbgh", True),
        "use_freq_aug": getattr(cfg, "use_freq_aug", False),
    }
    torch.save(state, path)


def _load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    device: torch.device,
) -> Tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", {}))
    optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", {}))
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint.get("scheduler_state_dict"))
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_dice = float(checkpoint.get("best_dice", 0.0))
    return start_epoch, best_dice


def _build_dataloaders(cfg: Config, device: torch.device) -> Tuple[DataLoader, DataLoader]:
    train_transform = build_transforms(
        "train",
        cfg.image_size,
        use_freq_aug=getattr(cfg, "use_freq_aug", False),
    )
    val_transform = build_transforms("val", cfg.image_size)

    train_dataset = PolypDataset(
        dataset_root=cfg.dataset_root,
        image_dir=cfg.image_dir,
        mask_dir=cfg.mask_dir,
        split="train",
        transform=train_transform,
        debug=cfg.debug,
        length=cfg.debug_samples,
        image_size=cfg.image_size,
        seed=cfg.seed,
    )
    val_dataset = PolypDataset(
        dataset_root=cfg.dataset_root,
        image_dir=cfg.image_dir,
        mask_dir=cfg.mask_dir,
        split="val",
        transform=val_transform,
        debug=cfg.debug,
        length=cfg.debug_samples,
        image_size=cfg.image_size,
        seed=cfg.seed,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    base_dir = os.path.join("experiments", cfg.experiment_name)
    log_dir = os.path.join(base_dir, "logs")
    visuals_dir = os.path.join(log_dir, "visuals")
    ensure_dir(log_dir)
    ensure_dir(visuals_dir)

    checkpoint_dir = cfg.checkpoint_dir or os.path.join(base_dir, "checkpoints")
    ensure_dir(checkpoint_dir)
    if not cfg.best_checkpoint:
        cfg.best_checkpoint = os.path.join(checkpoint_dir, "best.pth")
    if not cfg.last_checkpoint:
        cfg.last_checkpoint = os.path.join(checkpoint_dir, "last.pth")

    train_loader, val_loader = _build_dataloaders(cfg, device)
    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.epochs, 1),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    start_epoch = 0
    best_dice = 0.0
    best_epoch = 0
    if cfg.resume_checkpoint:
        if not os.path.isfile(cfg.resume_checkpoint):
            raise FileNotFoundError(
                f"Resume checkpoint not found: {cfg.resume_checkpoint}"
            )
        start_epoch, best_dice = _load_checkpoint(
            cfg.resume_checkpoint,
            model,
            optimizer,
            scheduler,
            device,
        )

    log_path = os.path.join(log_dir, "train_log.json")
    log_entries: list[Dict[str, float]] = []

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        train_loss_meter = AverageMeter()

        for step, (images, masks) in enumerate(train_loader):
            if cfg.debug and step >= cfg.debug_max_steps:
                break
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(images)
                loss, loss_dict = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_meter.update(loss.item(), images.size(0))

        val_loss, val_metrics, val_loss_dict = evaluate_model(
            model, val_loader, criterion, device, cfg
        )
        if scheduler is not None:
            scheduler.step()

        val_dice = val_metrics.get("dice", 0.0)
        lr = optimizer.param_groups[0]["lr"]

        log_entry = {
            "epoch": epoch + 1 ,
            "train_loss": train_loss_meter.avg,
            "val_loss": val_loss,
            "total_loss": val_loss_dict.get("total_loss", val_loss),
            "seg_bce_loss": val_loss_dict.get("seg_bce_loss", 0.0),
            "seg_dice_loss": val_loss_dict.get("seg_dice_loss", 0.0),
            "boundary_loss": val_loss_dict.get("boundary_loss", 0.0),
            "aux_boundary_loss": val_loss_dict.get("aux_boundary_loss", 0.0),
            "val_dice": val_dice,
            "val_iou": val_metrics.get("iou", 0.0),
            "val_precision": val_metrics.get("precision", 0.0),
            "val_recall": val_metrics.get("recall", 0.0),
            "val_mae": val_metrics.get("mae", 0.0),
            "val_fbeta": val_metrics.get("f_measure", 0.0),
            "learning_rate": lr,
        }
        log_entries.append(log_entry)
        save_json(log_path, log_entries)

        _save_visuals(model, val_loader, device, visuals_dir, epoch + 1)

        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch + 1
            _save_checkpoint(
                cfg.best_checkpoint,
                model,
                optimizer,
                scheduler,
                epoch,
                best_dice,
                cfg,
            )

        _save_checkpoint(
            cfg.last_checkpoint,
            model,
            optimizer,
            scheduler,
            epoch,
            best_dice,
            cfg,
        )

        print(
            "Epoch {}/{} | train_loss {:.4f} | total_loss {:.4f} | seg_bce {:.4f} | seg_dice {:.4f} | "
            "boundary {:.4f} | aux {:.4f} | val_dice {:.4f} | val_iou {:.4f} | lr {:.6f}".format(
                epoch + 1,
                cfg.epochs,
                train_loss_meter.avg,
                val_loss_dict.get("total_loss", val_loss),
                val_loss_dict.get("seg_bce_loss", 0.0),
                val_loss_dict.get("seg_dice_loss", 0.0),
                val_loss_dict.get("boundary_loss", 0.0),
                val_loss_dict.get("aux_boundary_loss", 0.0),
                val_dice,
                val_metrics.get("iou", 0.0),
                lr,
            )
        )

    results_path = os.path.join(base_dir, "results.json")
    results = {
        "best_dice": best_dice,
        "best_val_dice": best_dice,
        "best_epoch": best_epoch,
        "best_checkpoint_path": cfg.best_checkpoint,
        "final_epoch": cfg.epochs,
        "final_metrics": log_entries[-1] if log_entries else {},
    }
    save_json(results_path, results)


def main() -> None:
    parser = build_arg_parser()
    parser.description = "PolySegNet training"
    args = parser.parse_args()
    cfg = config_from_args(args)
    train(cfg)


if __name__ == "__main__":
    main()
