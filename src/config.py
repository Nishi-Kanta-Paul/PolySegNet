from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import yaml


DEFAULT_LOSS_WEIGHTS: Dict[str, float] = {
    "bce": 1.0,
    "dice": 1.0,
    "boundary": 1.0,
}


@dataclass
class Config:
    experiment_name: str = "polysegnet"
    dataset_root: str = "data"
    image_dir: str = "images"
    mask_dir: str = "masks"
    image_size: int = 352
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    seed: int = 42
    device: str = "cuda"
    model_name: str = "polysegnet"
    pretrained: bool = True
    use_msca: bool = True
    use_csaf: bool = True
    use_boundary_loss: bool = False
    loss_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LOSS_WEIGHTS))
    checkpoint_dir: str = ""
    resume_checkpoint: str = ""
    best_checkpoint: str = ""
    last_checkpoint: str = ""
    debug: bool = False
    debug_samples: int = 8
    debug_max_steps: int = 2

    def finalize(self) -> None:
        if not self.checkpoint_dir:
            self.checkpoint_dir = os.path.join(
                "experiments", self.experiment_name, "checkpoints"
            )

        for key, value in DEFAULT_LOSS_WEIGHTS.items():
            self.loss_weights.setdefault(key, value)

        if self.debug:
            apply_debug_overrides(self)


def apply_debug_overrides(cfg: Config) -> None:
    cfg.debug = True
    cfg.batch_size = min(cfg.batch_size, 2) if cfg.batch_size else 2
    cfg.num_workers = 0
    cfg.epochs = 1
    cfg.debug_max_steps = min(cfg.debug_max_steps, 2) if cfg.debug_max_steps else 2
    cfg.debug_samples = min(cfg.debug_samples, 16) if cfg.debug_samples else 8


def load_config(path: Optional[str], finalize: bool = True) -> Config:
    cfg = Config()
    if not path:
        if finalize:
            cfg.finalize()
        return cfg

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if isinstance(raw, dict):
        for key, value in raw.items():
            if key == "loss_weights" and isinstance(value, dict):
                cfg.loss_weights.update(value)
            elif hasattr(cfg, key):
                setattr(cfg, key, value)

    if finalize:
        cfg.finalize()
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PolySegNet configuration")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--mask-dir", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-msca",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-csaf",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-boundary-loss",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--loss-bce", type=float, default=None)
    parser.add_argument("--loss-dice", type=float, default=None)
    parser.add_argument("--loss-boundary", type=float, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--best-checkpoint", default=None)
    parser.add_argument("--last-checkpoint", default=None)
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--debug-samples", type=int, default=None)
    parser.add_argument("--debug-max-steps", type=int, default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = load_config(getattr(args, "config", None), finalize=False)

    for field_name in (
        "experiment_name",
        "dataset_root",
        "image_dir",
        "mask_dir",
        "image_size",
        "batch_size",
        "num_workers",
        "epochs",
        "learning_rate",
        "weight_decay",
        "seed",
        "device",
        "model_name",
        "pretrained",
        "use_msca",
        "use_csaf",
        "use_boundary_loss",
        "checkpoint_dir",
        "resume_checkpoint",
        "best_checkpoint",
        "last_checkpoint",
        "debug",
        "debug_samples",
        "debug_max_steps",
    ):
        value = getattr(args, field_name, None)
        if value is not None:
            setattr(cfg, field_name, value)

    if getattr(args, "loss_bce", None) is not None:
        cfg.loss_weights["bce"] = args.loss_bce
    if getattr(args, "loss_dice", None) is not None:
        cfg.loss_weights["dice"] = args.loss_dice
    if getattr(args, "loss_boundary", None) is not None:
        cfg.loss_weights["boundary"] = args.loss_boundary

    cfg.finalize()
    return cfg


def parse_args(argv: Optional[list[str]] = None) -> Config:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return config_from_args(args)
