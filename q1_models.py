"""
Q1 CycleGAN models (Generator + Discriminator).

Implements:
- ResNet generator (as in CycleGAN paper)
- 70x70 PatchGAN discriminator
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ResnetGeneratorConfig:
    in_channels: int = 3
    out_channels: int = 3
    ngf: int = 64
    n_blocks: int = 9  # use 9 for 256x256, 6 for 128x128


@dataclass(frozen=True)
class PatchDiscriminatorConfig:
    in_channels: int = 3
    ndf: int = 64
    n_layers: int = 3


def init_weights(net: nn.Module) -> None:
    """
    CycleGAN-style init: Normal(0, 0.02) for conv/linear; norm weights ~1.
    """

    def _init(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            "Conv" in classname or "Linear" in classname
        ):  # Conv2d/ConvTranspose2d/Linear
            nn.init.normal_(m.weight.data, 0.0, 0.02)  # type: ignore[attr-defined]
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias.data, 0.0)  # type: ignore[attr-defined]
        elif "InstanceNorm2d" in classname or "BatchNorm2d" in classname:
            if getattr(m, "weight", None) is not None:
                nn.init.normal_(m.weight.data, 1.0, 0.02)  # type: ignore[attr-defined]
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias.data, 0.0)  # type: ignore[attr-defined]

    net.apply(_init)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, cfg: ResnetGeneratorConfig) -> None:
        super().__init__()
        if cfg.n_blocks <= 0:
            raise ValueError("n_blocks must be > 0")

        model: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(cfg.in_channels, cfg.ngf, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(cfg.ngf),
            nn.ReLU(inplace=True),
        ]

        # Downsample
        in_f = cfg.ngf
        out_f = in_f * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_f, out_f, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_f),
                nn.ReLU(inplace=True),
            ]
            in_f = out_f
            out_f = min(out_f * 2, 512)

        # Residual blocks
        for _ in range(cfg.n_blocks):
            model += [ResidualBlock(in_f)]

        # Upsample
        out_f = in_f // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_f, out_f, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(out_f),
                nn.ReLU(inplace=True),
            ]
            in_f = out_f
            out_f = out_f // 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_f, cfg.out_channels, kernel_size=7, stride=1, padding=0),
            nn.Tanh(),
        ]

        self.net = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PatchDiscriminator(nn.Module):
    def __init__(self, cfg: PatchDiscriminatorConfig) -> None:
        super().__init__()

        kw = 4
        padw = 1

        sequence: list[nn.Module] = [
            nn.Conv2d(cfg.in_channels, cfg.ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, cfg.n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    cfg.ndf * nf_mult_prev,
                    cfg.ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=False,
                ),
                nn.InstanceNorm2d(cfg.ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** cfg.n_layers, 8)
        sequence += [
            nn.Conv2d(
                cfg.ndf * nf_mult_prev,
                cfg.ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=False,
            ),
            nn.InstanceNorm2d(cfg.ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Output 1-channel patch map (no sigmoid for LSGAN)
        sequence += [nn.Conv2d(cfg.ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.net = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

