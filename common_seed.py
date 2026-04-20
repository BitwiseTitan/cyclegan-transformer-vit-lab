"""
Reproducibility helpers (seed everything).

All tasks (Q1/Q2/Q3) should call `seed_everything()` at the start of training.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


@dataclass(frozen=True)
class SeedConfig:
    seed: int = 42
    deterministic: bool = False


def seed_everything(cfg: SeedConfig | int = 42) -> int:
    """
    Seed Python, NumPy, and Torch (if available).

    Returns the seed used.
    """
    seed = cfg.seed if isinstance(cfg, SeedConfig) else int(cfg)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if isinstance(cfg, SeedConfig) and cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return seed

