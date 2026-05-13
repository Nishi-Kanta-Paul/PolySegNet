import torch
from torch import nn


class PolySegNet(nn.Module):
    def __init__(self, encoder_name: str = "tf_efficientnet_b4", pretrained: bool = True):
        super().__init__()
        self.encoder_name = encoder_name
        self.pretrained = pretrained
        self.placeholder = nn.Identity()
        # TODO: implement EfficientNet-B4 encoder + U-Net decoder + MSCA + CSAF.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("PolySegNet forward pass not implemented yet.")
