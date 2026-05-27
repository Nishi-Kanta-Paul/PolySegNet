from __future__ import annotations

from typing import Iterable, List

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


class OriginalMSCA(nn.Module):
    def __init__(
        self,
        channels: int,
        dilations: Iterable[int] = (1, 3, 5, 7),
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [ConvBNReLU(channels, channels, kernel_size=3, dilation=d) for d in dilations]
        )
        self.fuse = ConvBNReLU(channels * len(dilations), channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [branch(x) for branch in self.branches]
        fused = self.fuse(torch.cat(features, dim=1))
        return fused + x


class BGDCMSCA(nn.Module):
    def __init__(
        self,
        channels: int,
        use_dynamic_weighting: bool = True,
    ) -> None:
        super().__init__()
        self.use_dynamic_weighting = use_dynamic_weighting

        self.reduce = ConvBNReLU(channels, channels, kernel_size=1)
        self.d1 = ConvBNReLU(channels, channels, kernel_size=3, dilation=1)
        self.d2 = ConvBNReLU(channels, channels, kernel_size=3, dilation=2)
        self.d3 = ConvBNReLU(channels, channels, kernel_size=3, dilation=3)
        self.d4 = ConvBNReLU(channels, channels, kernel_size=3, dilation=5)

        self.boundary_conv = ConvBNReLU(channels, channels, kernel_size=3)
        self.boundary_out = nn.Conv2d(channels, 1, kernel_size=1)

        if use_dynamic_weighting:
            hidden = max(channels // 4, 4)
            self.scale_mlp = nn.Sequential(
                nn.Linear(channels, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 1),
            )

        fuse_in = channels * (6 if use_dynamic_weighting else 5)
        self.fuse = ConvBNReLU(fuse_in, channels, kernel_size=1)
        self.residual = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0 = self.reduce(x)
        f1 = self.d1(f0)
        f2 = self.d2(f1)
        f3 = self.d3(f2)
        f4 = self.d4(f3)

        pb = torch.sigmoid(self.boundary_out(self.boundary_conv(f0)))

        if self.use_dynamic_weighting:
            mod = 1.0 + pb
            s1 = F.adaptive_avg_pool2d(f1 * mod, 1)
            s2 = F.adaptive_avg_pool2d(f2 * mod, 1)
            s3 = F.adaptive_avg_pool2d(f3 * mod, 1)
            s4 = F.adaptive_avg_pool2d(f4 * mod, 1)

            s = torch.stack([s1, s2, s3, s4], dim=1).squeeze(-1).squeeze(-1)
            s_flat = s.reshape(-1, s.shape[-1])
            weights = self.scale_mlp(s_flat)
            weights = weights.view(s.shape[0], 4, 1)
            weights = torch.softmax(weights, dim=1)
            weights = weights.view(s.shape[0], 4, 1, 1, 1)

            f_dyn = (
                weights[:, 0] * f1
                + weights[:, 1] * f2
                + weights[:, 2] * f3
                + weights[:, 3] * f4
            )
            fused = torch.cat([f0, f_dyn, f1, f2, f3, f4], dim=1)
        else:
            fused = torch.cat([f0, f1, f2, f3, f4], dim=1)

        fused = self.fuse(fused)
        return fused + self.residual(x)


class BGSAGF(nn.Module):
    def __init__(
        self,
        channels: int,
        use_boundary_guidance: bool = True,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.use_boundary_guidance = use_boundary_guidance

        self.up_conv = ConvBNReLU(channels, channels, kernel_size=3)

        self.boundary_conv = ConvBNReLU(channels * 2, channels, kernel_size=3)
        self.boundary_out = nn.Conv2d(channels, 1, kernel_size=1)

        gate_in = channels * 2 + (1 if use_boundary_guidance else 0)
        self.gate_conv = ConvBNReLU(gate_in, channels, kernel_size=3)
        self.gate_out = nn.Conv2d(channels, channels, kernel_size=1)

        hidden = max(channels // reduction, 4)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        self.fuse = ConvBNReLU(channels * 2, channels, kernel_size=3)

    def forward(self, skip: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
        decoder = F.interpolate(
            decoder,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoder = self.up_conv(decoder)

        if self.use_boundary_guidance:
            join = torch.cat([skip, decoder], dim=1)
            boundary = torch.sigmoid(self.boundary_out(self.boundary_conv(join)))
            gate_in = torch.cat([skip, decoder, boundary], dim=1)
            gate = torch.sigmoid(self.gate_out(self.gate_conv(gate_in)))
            selected = skip * gate * (1.0 + boundary)
        else:
            gate_in = torch.cat([skip, decoder], dim=1)
            gate = torch.sigmoid(self.gate_out(self.gate_conv(gate_in)))
            selected = skip * gate

        avg_pool = F.adaptive_avg_pool2d(selected, 1)
        max_pool = F.adaptive_max_pool2d(selected, 1)
        channel_attn = torch.sigmoid(self.channel_mlp(avg_pool) + self.channel_mlp(max_pool))
        selected = selected * channel_attn

        avg_map = torch.mean(selected, dim=1, keepdim=True)
        max_map = torch.amax(selected, dim=1, keepdim=True)
        spatial_attn = torch.sigmoid(self.spatial_conv(torch.cat([avg_map, max_map], dim=1)))
        selected = selected * spatial_attn

        fused = torch.cat([decoder, selected], dim=1)
        return self.fuse(fused)


class SimpleFuseBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.up_conv = ConvBNReLU(channels, channels, kernel_size=3)
        self.fuse = ConvBNReLU(channels * 2, channels, kernel_size=3)

    def forward(self, skip: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
        decoder = F.interpolate(
            decoder,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoder = self.up_conv(decoder)
        return self.fuse(torch.cat([skip, decoder], dim=1))


class MBGH(nn.Module):
    def __init__(self, channels: int, use_multilevel_boundary: bool = True) -> None:
        super().__init__()
        self.use_multilevel_boundary = use_multilevel_boundary

        self.boundary_aux = ConvBNReLU(channels, channels, kernel_size=3)
        self.boundary_out = nn.Conv2d(channels, 1, kernel_size=1)
        self.mask_out = nn.Conv2d(channels, 1, kernel_size=1)

        self.aux_heads = nn.ModuleList()
        if use_multilevel_boundary:
            for _ in range(3):
                self.aux_heads.append(
                    nn.Sequential(
                        ConvBNReLU(channels, channels, kernel_size=3),
                        nn.Conv2d(channels, 1, kernel_size=1),
                    )
                )

    def forward(
        self,
        d1: torch.Tensor,
        d2: torch.Tensor | None = None,
        d3: torch.Tensor | None = None,
        d4: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        boundary_logits = self.boundary_out(self.boundary_aux(d1))
        boundary_prob = torch.sigmoid(boundary_logits)
        d1_bg = d1 + d1 * boundary_prob
        mask_logits = self.mask_out(d1_bg)

        aux_logits: list[torch.Tensor] = []
        if self.use_multilevel_boundary:
            features = [d2, d3, d4]
            for feat, head in zip(features, self.aux_heads):
                if feat is None:
                    continue
                feat = F.interpolate(
                    feat,
                    size=d1.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                aux_logits.append(head(feat))
        return boundary_logits, mask_logits, aux_logits


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, kernel_size=3),
            ConvBNReLU(out_channels, out_channels, kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.up4(bottleneck)
        dec4 = self.dec4(torch.cat([dec4, enc4], dim=1))
        dec3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))
        dec2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        return self.out_conv(dec1)


def _build_unetpp(config) -> nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "segmentation_models_pytorch is required for Unet++. "
            "Install with: pip install segmentation-models-pytorch"
        ) from exc

    encoder_name = getattr(config, "unetpp_encoder_name", "resnet34")
    encoder_weights = getattr(config, "unetpp_encoder_weights", "imagenet")
    in_channels = int(getattr(config, "unetpp_in_channels", 3))
    classes = int(getattr(config, "unetpp_classes", 1))

    return smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )


class BGDSFPolySegNet(nn.Module):
    def __init__(
        self,
        encoder_name: str = "tf_efficientnet_b4",
        pretrained: bool = True,
        unified_channels: int = 128,
        use_msca: bool = True,
        use_csaf: bool = True,
        use_mbgh: bool = True,
        use_dynamic_weighting: bool = True,
        use_boundary_guidance: bool = True,
        use_boundary_loss: bool = False,
        use_multilevel_boundary: bool = True,
        msca_type: str = "bgd_cmsca",
    ) -> None:
        super().__init__()
        self.use_msca = use_msca
        self.use_csaf = use_csaf
        self.use_mbgh = use_mbgh

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
        )
        encoder_channels: List[int] = list(self.encoder.feature_info.channels())
        if len(encoder_channels) < 5:
            raise ValueError("Encoder must provide at least five feature maps.")
        if len(encoder_channels) > 5:
            encoder_channels = encoder_channels[-5:]

        self.projections = nn.ModuleList(
            [ConvBNReLU(ch, unified_channels, kernel_size=1) for ch in encoder_channels]
        )

        if use_msca:
            if msca_type == "original_msca":
                self.bottleneck = OriginalMSCA(unified_channels)
            else:
                self.bottleneck = BGDCMSCA(
                    unified_channels,
                    use_dynamic_weighting=use_dynamic_weighting,
                )
        else:
            self.bottleneck = nn.Identity()

        decoder_blocks: List[nn.Module] = []
        for _ in range(4):
            if use_csaf:
                decoder_blocks.append(
                    BGSAGF(
                        unified_channels,
                        use_boundary_guidance=use_boundary_guidance,
                    )
                )
            else:
                decoder_blocks.append(SimpleFuseBlock(unified_channels))
        self.decoder_blocks = nn.ModuleList(decoder_blocks)

        self.final_conv = nn.Conv2d(unified_channels, 1, kernel_size=1)
        self.mbgh = MBGH(unified_channels, use_multilevel_boundary) if use_mbgh else None

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        features = self.encoder(x)
        if len(features) > 5:
            features = features[-5:]
        if len(features) < 5:
            raise ValueError("Encoder output does not contain five feature maps.")

        projected = [proj(feat) for proj, feat in zip(self.projections, features)]
        e1, e2, e3, e4, e5 = projected

        if self.use_msca:
            b = self.bottleneck(e5)
        else:
            b = e5

        d5 = b
        d4 = self.decoder_blocks[0](e4, d5)
        d3 = self.decoder_blocks[1](e3, d4)
        d2 = self.decoder_blocks[2](e2, d3)
        d1 = self.decoder_blocks[3](e1, d2)

        if self.use_mbgh and self.mbgh is not None:
            boundary_logits, mask_logits, aux_boundary_logits = self.mbgh(d1, d2, d3, d4)
        else:
            mask_logits = self.final_conv(d1)
            boundary_logits = None
            aux_boundary_logits = []

        target_size = x.shape[-2:]
        mask_logits = F.interpolate(
            mask_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        if boundary_logits is not None:
            boundary_logits = F.interpolate(
                boundary_logits,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        # Keep auxiliary boundary logits at their native decoder scales.

        return {
            "mask_logits": mask_logits,
            "boundary_logits": boundary_logits,
            "aux_boundary_logits": aux_boundary_logits,
        }


def build_model(config) -> nn.Module:
    model_name = getattr(config, "model_name", "polysegnet")
    pretrained = bool(getattr(config, "pretrained", True))
    if getattr(config, "debug", False):
        pretrained = False

    unified_channels = int(getattr(config, "unified_channels", 128))
    use_msca = bool(getattr(config, "use_msca", True))
    use_csaf = bool(getattr(config, "use_csaf", True))
    use_mbgh = bool(getattr(config, "use_mbgh", True))
    use_dynamic_weighting = bool(getattr(config, "use_dynamic_weighting", True))
    use_boundary_guidance = bool(getattr(config, "use_boundary_guidance", True))
    use_boundary_loss = bool(getattr(config, "use_boundary_loss", False))
    use_multilevel_boundary = bool(getattr(config, "use_multilevel_boundary", True))

    if model_name == "unet":
        return UNet()

    if model_name in {"unetpp", "unetplusplus", "unet++"}:
        return _build_unetpp(config)

    msca_type = "bgd_cmsca"
    if model_name == "original_msca":
        msca_type = "original_msca"

    if model_name in {
        "polysegnet",
        "effb4_unet",
        "effb4",
        "bgdsf_polysegnet",
        "bgdsf",
        "original_msca",
    }:
        return BGDSFPolySegNet(
            encoder_name="tf_efficientnet_b4",
            pretrained=pretrained,
            unified_channels=unified_channels,
            use_msca=use_msca,
            use_csaf=use_csaf,
            use_mbgh=use_mbgh,
            use_dynamic_weighting=use_dynamic_weighting,
            use_boundary_guidance=use_boundary_guidance,
            use_boundary_loss=use_boundary_loss,
            use_multilevel_boundary=use_multilevel_boundary,
            msca_type=msca_type,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


if __name__ == "__main__":
    from types import SimpleNamespace

    config = SimpleNamespace(
        model_name="bgdsf_polysegnet",
        pretrained=False,
        unified_channels=128,
        use_msca=True,
        use_csaf=True,
        use_mbgh=True,
        use_boundary_loss=True,
        use_dynamic_weighting=True,
        use_boundary_guidance=True,
        use_multilevel_boundary=True,
    )
    model = build_model(config)
    model.eval()
    x = torch.randn(2, 3, 352, 352)
    with torch.no_grad():
        outputs = model(x)
    print("mask_logits shape:      ", outputs["mask_logits"].shape)
    print("boundary_logits shape:  ", outputs["boundary_logits"].shape)
    print("aux_boundary count:     ", len(outputs["aux_boundary_logits"]))
    for i, ab in enumerate(outputs["aux_boundary_logits"]):
        print(f"  aux_boundary[{i}] shape:", ab.shape)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n:.2f}M")
    assert outputs["mask_logits"].shape == (2, 1, 352, 352)
    assert outputs["boundary_logits"].shape == (2, 1, 352, 352)
    print("All assertions passed.")
