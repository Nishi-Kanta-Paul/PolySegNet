import argparse
import os
import sys

import cv2
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import build_arg_parser, config_from_args
from src.dataset import PolypDataset, build_transforms
from src.utils import ensure_dir, overlay_segmentation, set_seed


def main() -> None:
    parser = build_arg_parser()
    parser.description = "PolySegNet dataset check"
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--num-samples", type=int, default=3)
    args = parser.parse_args()

    cfg = config_from_args(args)
    set_seed(cfg.seed)

    transform = build_transforms(args.split, cfg.image_size)
    dataset = PolypDataset(
        dataset_root=cfg.dataset_root,
        image_dir=cfg.image_dir,
        mask_dir=cfg.mask_dir,
        split=args.split,
        transform=transform,
        debug=cfg.debug,
        length=cfg.debug_samples,
        image_size=cfg.image_size,
        seed=cfg.seed,
    )

    print(f"Dataset size: {len(dataset)}")
    if len(dataset) == 0:
        return

    image, mask = dataset[0]
    print(f"Image shape: {tuple(image.shape)}")
    print(f"Mask shape: {tuple(mask.shape)}")

    values = set()
    check_count = min(len(dataset), 25)
    for idx in range(check_count):
        _, mask = dataset[idx]
        values.update(torch.unique(mask).cpu().numpy().tolist())

    is_binary = all(value in (0.0, 1.0) for value in values)
    print(f"Mask binary: {is_binary}, values: {sorted(values)}")

    output_dir = os.path.join("experiments", "debug_dataset", "logs")
    ensure_dir(output_dir)

    sample_count = min(len(dataset), args.num_samples)
    for idx in range(sample_count):
        image, mask = dataset[idx]
        image_np = image.permute(1, 2, 0).numpy()
        mask_np = mask.squeeze(0).numpy()
        overlay = overlay_segmentation(image_np, mask_np)
        output_path = os.path.join(output_dir, f"{args.split}_sample_{idx}.png")
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"Saved {sample_count} overlays to {output_dir}")


if __name__ == "__main__":
    main()
