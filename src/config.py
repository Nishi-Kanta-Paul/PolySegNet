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
    dataset_roots: list[str] = field(default_factory=list)
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
    model_name: str = "bgdsf_polysegnet"
    bgdsf_encoder_name: str = "tf_efficientnet_b4"
    pretrained: bool = True
    smp_encoder_name: str = "resnet34"
    smp_encoder_weights: str = "imagenet"
    smp_in_channels: int = 3
    smp_classes: int = 1
    unetpp_encoder_name: str = "resnet34"
    unetpp_encoder_weights: str = "imagenet"
    unetpp_in_channels: int = 3
    unetpp_classes: int = 1
    use_msca: bool = True
    use_csaf: bool = True
    use_mbgh: bool = True
    use_boundary_loss: bool = True
    use_dynamic_weighting: bool = True
    use_boundary_guidance: bool = True
    use_multilevel_boundary: bool = True
    use_freq_aug: bool = False
    boundary_kernel_size: int = 3
    aux_boundary_weight: float = 0.3
    unified_channels: int = 128
    loss_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LOSS_WEIGHTS))
    checkpoint_dir: str = ""
    resume_checkpoint: str = ""
    best_checkpoint: str = ""
    last_checkpoint: str = ""
    checkpoint: str = ""
    image: str = ""
    debug: bool = False
    debug_samples: int = 8
    debug_max_steps: int = 2

    def finalize(self) -> None:
        if not self.checkpoint_dir:
            self.checkpoint_dir = os.path.join(
                "experiments", self.experiment_name, "checkpoints"
            )

        if not self.best_checkpoint:
            self.best_checkpoint = os.path.join(self.checkpoint_dir, "best.pth")
        if not self.last_checkpoint:
            self.last_checkpoint = os.path.join(self.checkpoint_dir, "latest.pth")

        for key, value in DEFAULT_LOSS_WEIGHTS.items():
            self.loss_weights.setdefault(key, value)

        if self.debug:
            apply_debug_overrides(self)


def apply_debug_overrides(cfg: Config) -> None:
    cfg.debug = True
    cfg.batch_size = min(cfg.batch_size, 2) if cfg.batch_size else 2
    cfg.num_workers = 0
    cfg.epochs = 2
    cfg.debug_max_steps = min(cfg.debug_max_steps, 2) if cfg.debug_max_steps else 2
    cfg.debug_samples = min(cfg.debug_samples, 16) if cfg.debug_samples else 8
    cfg.experiment_name = "debug"
    cfg.checkpoint_dir = os.path.join("experiments", cfg.experiment_name, "checkpoints")
    cfg.best_checkpoint = os.path.join(cfg.checkpoint_dir, "best.pth")
    cfg.last_checkpoint = os.path.join(cfg.checkpoint_dir, "latest.pth")
    cfg.use_boundary_loss = True
    cfg.use_multilevel_boundary = True


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
    parser.add_argument("--dataset-roots", nargs="+", default=None)
    parser.add_argument("--image-dir", "--image_dir", default=None)
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
    parser.add_argument("--bgdsf-encoder-name", default=None)
    parser.add_argument("--smp-encoder-name", default=None)
    parser.add_argument("--smp-encoder-weights", default=None)
    parser.add_argument("--smp-in-channels", type=int, default=None)
    parser.add_argument("--smp-classes", type=int, default=None)
    parser.add_argument("--unetpp-encoder-name", default=None)
    parser.add_argument("--unetpp-encoder-weights", default=None)
    parser.add_argument("--unetpp-in-channels", type=int, default=None)
    parser.add_argument("--unetpp-classes", type=int, default=None)
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
        "--use-mbgh",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-boundary-loss",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-dynamic-weighting",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-boundary-guidance",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-multilevel-boundary",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-freq-aug",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--loss-bce", type=float, default=None)
    parser.add_argument("--loss-dice", type=float, default=None)
    parser.add_argument("--loss-boundary", type=float, default=None)
    parser.add_argument("--boundary-kernel-size", type=int, default=None)
    parser.add_argument("--aux-boundary-weight", type=float, default=None)
    parser.add_argument("--unified-channels", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--best-checkpoint", default=None)
    parser.add_argument("--last-checkpoint", default=None)
    parser.add_argument("--image", default=None)
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
        "dataset_roots",
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
        "bgdsf_encoder_name",
        "smp_encoder_name",
        "smp_encoder_weights",
        "smp_in_channels",
        "smp_classes",
        "unetpp_encoder_name",
        "unetpp_encoder_weights",
        "unetpp_in_channels",
        "unetpp_classes",
        "pretrained",
        "use_msca",
        "use_csaf",
        "use_mbgh",
        "use_boundary_loss",
        "use_dynamic_weighting",
        "use_boundary_guidance",
        "use_multilevel_boundary",
        "use_freq_aug",
        "boundary_kernel_size",
        "aux_boundary_weight",
        "unified_channels",
        "checkpoint_dir",
        "checkpoint",
        "resume_checkpoint",
        "best_checkpoint",
        "last_checkpoint",
        "image",
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
