"""
Q1 losses (adversarial, cycle-consistency, identity).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self, mode: str = "lsgan") -> None:
        super().__init__()
        self.mode = mode
        if mode not in {"lsgan", "vanilla"}:
            raise ValueError(f"Unsupported GANLoss mode: {mode}")
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if self.mode == "lsgan":
            target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
            return self.mse(pred, target)
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.bce(pred, target)

