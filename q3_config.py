"""
Q3 configuration (CIFAR-10: CNN vs ViT).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common_utils import project_root


@dataclass(frozen=True)
class Q3Config:
    data_dir: Path = project_root() / "data" / "q3"
    artifacts_dir: Path = project_root() / "artifacts" / "q3"

    epochs: int = 20
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 5e-2
    num_workers: int = 2
    seed: int = 42

