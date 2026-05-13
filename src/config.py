from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class DataConfig:
    dataset: str = "kvasir-seg"
    images_dir: str = ""
    masks_dir: str = ""
    image_size: int = 352
    num_workers: int = 4
    debug: bool = False


@dataclass
class TrainConfig:
    batch_size: int = 8
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-5
    device: str = "cuda"
    seed: int = 42


@dataclass
class ModelConfig:
    encoder_name: str = "tf_efficientnet_b4"
    pretrained: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def load_config(path: Optional[str]) -> Config:
    if not path:
        return Config()

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    cfg = Config()
    for section in ("data", "train", "model"):
        section_values = raw.get(section, {})
        if isinstance(section_values, dict):
            section_obj = getattr(cfg, section)
            for key, value in section_values.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)

    return cfg
