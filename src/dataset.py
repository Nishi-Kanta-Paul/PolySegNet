import os
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _list_images(root: str) -> List[str]:
    if not root or not os.path.isdir(root):
        return []
    return [
        os.path.join(root, name)
        for name in sorted(os.listdir(root))
        if name.lower().endswith(IMAGE_EXTS)
    ]


class PolypDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: Optional[Callable] = None,
        debug: bool = False,
        length: int = 8,
        image_size: int = 352,
    ) -> None:
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.debug = debug
        self.length = length
        self.image_size = image_size
        self.image_paths = [] if debug else _list_images(images_dir)

    def __len__(self) -> int:
        return self.length if self.debug else len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.debug:
            image = np.random.randint(
                0, 256, (self.image_size, self.image_size, 3), dtype=np.uint8
            )
            mask = np.random.randint(
                0, 2, (self.image_size, self.image_size), dtype=np.uint8
            )
        else:
            image_path = self.image_paths[idx]
            mask_path = os.path.join(self.masks_dir, os.path.basename(image_path))

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32) / 255.0

        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1)
        return image_tensor, mask_tensor
