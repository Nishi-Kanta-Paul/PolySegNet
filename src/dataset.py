import os
import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
DEFAULT_SPLIT_RATIOS: Tuple[float, float, float] = (0.8, 0.1, 0.1)


def _resolve_dir(base: str, subdir: str) -> str:
    if not subdir:
        return base
    if os.path.isabs(subdir):
        return subdir
    if base:
        return os.path.join(base, subdir)
    return subdir


def _list_images(root: str) -> List[str]:
    if not root or not os.path.isdir(root):
        return []
    return [
        os.path.join(root, name)
        for name in sorted(os.listdir(root))
        if name.lower().endswith(IMAGE_EXTS)
    ]


def _read_split_file(path: str) -> List[str]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _write_split_file(path: str, names: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(names))


def _create_splits(
    names: List[str],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    ordered = sorted(names)
    rng.shuffle(ordered)

    total = len(ordered)
    train_count = int(total * ratios[0])
    val_count = int(total * ratios[1])
    test_count = total - train_count - val_count

    train_names = ordered[:train_count]
    val_names = ordered[train_count : train_count + val_count]
    test_names = ordered[train_count + val_count : train_count + val_count + test_count]

    return {"train": train_names, "val": val_names, "test": test_names}


def _load_or_create_split_names(
    names: List[str],
    dataset_root: str,
    split: str,
    ratios: Tuple[float, float, float],
    seed: int,
    split_dir: Optional[str] = None,
) -> List[str]:
    split_dir = split_dir or dataset_root or "."
    train_path = os.path.join(split_dir, "train.txt")
    val_path = os.path.join(split_dir, "val.txt")
    test_path = os.path.join(split_dir, "test.txt")

    if os.path.isfile(train_path) and os.path.isfile(val_path) and os.path.isfile(test_path):
        split_files = {"train": train_path, "val": val_path, "test": test_path}
        return _read_split_file(split_files[split])

    splits = _create_splits(names, ratios, seed)
    _write_split_file(train_path, splits["train"])
    _write_split_file(val_path, splits["val"])
    _write_split_file(test_path, splits["test"])
    return splits[split]


def _frequency_augment_np(image: np.ndarray, **kwargs) -> np.ndarray:
    img = image.astype(np.float32)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    fft = np.fft.fft2(img, axes=(0, 1))
    amplitude = np.abs(fft)
    phase = np.angle(fft)

    alpha = np.random.uniform(0.05, 0.15)
    noise = np.random.uniform(-1.0, 1.0, size=amplitude.shape)
    amplitude = amplitude * (1.0 + alpha * noise)

    perturbed = amplitude * np.exp(1j * phase)
    reconstructed = np.fft.ifft2(perturbed, axes=(0, 1)).real

    min_val = img.min(axis=(0, 1), keepdims=True)
    max_val = img.max(axis=(0, 1), keepdims=True)
    reconstructed = np.maximum(np.minimum(reconstructed, max_val), min_val)
    return reconstructed


def build_transforms(
    split: str,
    image_size: int,
    vertical_flip: bool = False,
    use_freq_aug: bool = False,
) -> A.Compose:
    resize = A.Resize(
        height=image_size,
        width=image_size,
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
    )
    normalize = A.Normalize()

    if split == "train":
        transforms = [
            resize,
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(p=0.5),
        ]
        if use_freq_aug:
            transforms.append(A.Lambda(image=_frequency_augment_np, p=1.0))
        transforms.append(normalize)
        if vertical_flip:
            transforms.insert(2, A.VerticalFlip(p=0.5))
    else:
        transforms = [resize, normalize]

    return A.Compose(transforms)


class PolypDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        image_dir: str,
        mask_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        debug: bool = False,
        length: int = 8,
        image_size: int = 352,
        seed: int = 42,
        split_ratios: Tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
        split_dir: Optional[str] = None,
        return_paths: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split '{split}'.")

        self.split = split
        self.transform = transform
        self.debug = debug
        self.length = length
        self.image_size = image_size
        self.samples: List[Tuple[str, str]] = []
        self.return_paths = return_paths

        if self.debug:
            return

        images_dir = _resolve_dir(dataset_root, image_dir)
        masks_dir = _resolve_dir(dataset_root, mask_dir)

        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not os.path.isdir(masks_dir):
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

        image_paths = _list_images(images_dir)
        if not image_paths:
            raise FileNotFoundError(f"No images found in: {images_dir}")

        image_names = [os.path.basename(path) for path in image_paths]
        split_names = _load_or_create_split_names(
            image_names,
            dataset_root,
            split,
            split_ratios,
            seed,
            split_dir,
        )

        path_map = {os.path.basename(path): path for path in image_paths}
        missing_images = [name for name in split_names if name not in path_map]
        if missing_images:
            raise FileNotFoundError(
                "Split file references missing images, for example: "
                f"{missing_images[:3]}"
            )

        mask_paths = _list_images(masks_dir)
        mask_name_map = {os.path.basename(path): path for path in mask_paths}
        mask_stem_map: Dict[str, str] = {}
        for path in mask_paths:
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem not in mask_stem_map:
                mask_stem_map[stem] = path

        missing_masks: List[str] = []
        for name in split_names:
            if name in mask_name_map:
                continue
            stem = os.path.splitext(name)[0]
            if stem not in mask_stem_map:
                missing_masks.append(name)
        if missing_masks:
            raise FileNotFoundError(
                "Masks missing for images, for example: "
                f"{missing_masks[:3]}"
            )

        self.samples = []
        for name in split_names:
            image_path = path_map[name]
            mask_path = mask_name_map.get(name)
            if mask_path is None:
                stem = os.path.splitext(name)[0]
                mask_path = mask_stem_map[stem]
            self.samples.append((image_path, mask_path))

    def __len__(self) -> int:
        return self.length if self.debug else len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, str]]]:
        if self.debug:
            image = np.random.randint(
                0, 256, (self.image_size, self.image_size, 3), dtype=np.uint8
            )
            mask = np.random.randint(
                0, 2, (self.image_size, self.image_size), dtype=np.uint8
            )
            image_path = f"debug_image_{idx}.png"
            mask_path = f"debug_mask_{idx}.png"
        else:
            image_path, mask_path = self.samples[idx]

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
        else:
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )
            mask = cv2.resize(
                mask,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )

        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)

        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1)
        if self.return_paths:
            return image_tensor, mask_tensor, {"image_path": image_path, "mask_path": mask_path}
        return image_tensor, mask_tensor
