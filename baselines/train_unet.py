import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import build_arg_parser, config_from_args
from src.train import train


def main() -> None:
    parser = build_arg_parser()
    parser.description = "U-Net baseline training"
    args = parser.parse_args()

    cfg = config_from_args(args)
    if args.experiment_name is None:
        cfg.experiment_name = "unet_baseline"
    cfg.model_name = "unet"
    cfg.use_msca = False
    cfg.use_csaf = False
    cfg.use_boundary_loss = False

    train(cfg)


if __name__ == "__main__":
    main()
