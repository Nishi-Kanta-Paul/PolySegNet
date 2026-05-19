import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import build_arg_parser, config_from_args
from src.train import train


VARIANTS = {
    "effb4_unet": {
        "experiment": "effb4_unet",
        "use_msca": False,
        "use_csaf": False,
        "use_boundary_loss": False,
        "model_name": "effb4_unet",
    },
    "effb4_unet_msca": {
        "experiment": "effb4_unet_msca",
        "use_msca": True,
        "use_csaf": False,
        "use_boundary_loss": False,
        "model_name": "effb4_unet",
    },
    "effb4_unet_msca_csaf": {
        "experiment": "effb4_unet_msca_csaf",
        "use_msca": True,
        "use_csaf": True,
        "use_boundary_loss": False,
        "model_name": "effb4_unet",
    },
    "polysegnet_full": {
        "experiment": "polysegnet_full",
        "use_msca": True,
        "use_csaf": True,
        "use_boundary_loss": True,
        "model_name": "polysegnet",
    },
}


def main() -> None:
    parser = build_arg_parser()
    parser.description = "EfficientNet-B4 U-Net ablations"
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS.keys()),
        default="effb4_unet",
    )
    args = parser.parse_args()

    cfg = config_from_args(args)
    variant = VARIANTS[args.variant]

    if args.experiment_name is None:
        cfg.experiment_name = variant["experiment"]
    cfg.use_msca = variant["use_msca"]
    cfg.use_csaf = variant["use_csaf"]
    cfg.use_boundary_loss = variant["use_boundary_loss"]
    cfg.model_name = variant["model_name"]

    train(cfg)


if __name__ == "__main__":
    main()
