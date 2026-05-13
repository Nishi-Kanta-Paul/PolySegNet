from __future__ import annotations

from typing import Iterable, List, Tuple

import timm
import torch
import torch.nn.functional as F
from torch import nn

try:
    from .utils import count_parameters
except ImportError:
    from utils import count_parameters


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MSCA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        dilations: Iterable[int] = (1, 3, 5, 7),
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.branches = nn.ModuleList(
            [
                ConvBNReLU(in_channels, out_channels, kernel_size=3, dilation=d)
                for d in dilations
            ]
        )
        self.fuse = ConvBNReLU(out_channels * len(self.branches), out_channels, kernel_size=1)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [branch(x) for branch in self.branches]
        fused = torch.cat(features, dim=1)
        fused = self.fuse(fused)
        return fused + self.residual(x)


class CSAF(nn.Module):
    def __init__(
        self,
        skip_channels: int,
        decoder_channels: int,
        out_channels: int,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.skip_proj = ConvBNReLU(skip_channels, out_channels, kernel_size=1)
        self.dec_proj = ConvBNReLU(decoder_channels, out_channels, kernel_size=1)

        hidden = max(out_channels // reduction, 4)
        self.channel_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, skip: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
        skip = self.skip_proj(skip)
        decoder = self.dec_proj(decoder)
        fused = skip + decoder

        channel_gate = self.channel_mlp(fused)
        fused = fused * channel_gate

        avg_map = torch.mean(fused, dim=1, keepdim=True)
        max_map, _ = torch.max(fused, dim=1, keepdim=True)
        spatial_gate = self.spatial_conv(torch.cat([avg_map, max_map], dim=1))
        fused = fused * spatial_gate
        return fused


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_csaf: bool,
    ) -> None:
        super().__init__()
        self.use_csaf = use_csaf
        if use_csaf:
            self.fuse = CSAF(skip_channels, in_channels, out_channels)
            self.refine = nn.Sequential(
                ConvBNReLU(out_channels, out_channels, kernel_size=3),
                ConvBNReLU(out_channels, out_channels, kernel_size=3),
            )
        else:
            self.skip_proj = ConvBNReLU(skip_channels, out_channels, kernel_size=1)
            self.dec_proj = ConvBNReLU(in_channels, out_channels, kernel_size=1)
            self.refine = nn.Sequential(
                ConvBNReLU(out_channels * 2, out_channels, kernel_size=3),
                ConvBNReLU(out_channels, out_channels, kernel_size=3),
            )

    def forward(self, decoder: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        decoder = F.interpolate(
            decoder,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if self.use_csaf:
            fused = self.fuse(skip, decoder)
        else:
            skip = self.skip_proj(skip)
            decoder = self.dec_proj(decoder)
            fused = torch.cat([skip, decoder], dim=1)
        return self.refine(fused)


class PolySegNet(nn.Module):
    def __init__(
        self,
        encoder_name: str = "tf_efficientnet_b4",
        pretrained: bool = True,
        use_msca: bool = True,
        use_csaf: bool = True,
    ) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.pretrained = pretrained
        self.use_msca = use_msca
        self.use_csaf = use_csaf

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
        )
        encoder_channels: List[int] = list(self.encoder.feature_info.channels())
        if len(encoder_channels) < 2:
            raise ValueError("Encoder must provide at least two feature maps.")

        bottleneck_channels = encoder_channels[-1]
        self.msca = MSCA(bottleneck_channels) if use_msca else nn.Identity()

        decoder_blocks: List[DecoderBlock] = []
        in_channels = bottleneck_channels
        for skip_channels in reversed(encoder_channels[:-1]):
            out_channels = skip_channels
            decoder_blocks.append(
                DecoderBlock(
                    in_channels=in_channels,
                    skip_channels=skip_channels,
                    out_channels=out_channels,
                    use_csaf=use_csaf,
                )
            )
            in_channels = out_channels

        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.head = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        bottleneck = self.msca(features[-1])

        decoder = bottleneck
        for block, skip in zip(self.decoder_blocks, reversed(features[:-1])):
            decoder = block(decoder, skip)

        logits = self.head(decoder)
        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits


def build_model(config) -> PolySegNet:
    return PolySegNet(
        encoder_name="tf_efficientnet_b4",
        pretrained=getattr(config, "pretrained", True),
        use_msca=getattr(config, "use_msca", True),
        use_csaf=getattr(config, "use_csaf", True),
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolySegNet().to(device)
    dummy = torch.randn(2, 3, 352, 352, device=device)
    with torch.no_grad():
        output = model(dummy)
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Parameters: {count_parameters(model)}")
