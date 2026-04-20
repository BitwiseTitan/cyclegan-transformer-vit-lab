"""
Q3 custom Vision Transformer for CIFAR-10.

Implements a small ViT suitable for CIFAR-10 (32x32).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class Q3ViTConfig:
    img_size: int = 32
    patch_size: int = 4
    in_chans: int = 3
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    num_classes: int = 10


class _MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _EncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + self.drop(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class Q3ViT(nn.Module):
    def __init__(self, cfg: Q3ViTConfig = Q3ViTConfig()) -> None:
        super().__init__()
        if cfg.img_size % cfg.patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.cfg = cfg
        num_patches = (cfg.img_size // cfg.patch_size) ** 2
        patch_dim = cfg.in_chans * cfg.patch_size * cfg.patch_size

        self.patch_to_emb = nn.Linear(patch_dim, cfg.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, 1 + num_patches, cfg.embed_dim))
        self.pos_drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [_EncoderBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _img_to_patches(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, N, patch_dim)
        B, C, H, W = x.shape
        p = self.cfg.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, H/p, W/p, p, p, C)
        return x.view(B, (H // p) * (W // p), p * p * C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self._img_to_patches(x)
        x = self.patch_to_emb(patches)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_emb)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])

