"""
Q1 configuration (CycleGAN face ↔ sketch).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common_utils import project_root


@dataclass(frozen=True)
class Q1Config:
    # Data
    data_dir: Path = project_root() / "data" / "q1"
    image_size: int = 256
    batch_size: int = 1  # CycleGAN paper uses 1
    num_workers: int = 2

    # Training
    epochs: int = 50
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_cycle: float = 10.0
    lambda_id: float = 5.0

    # Outputs
    artifacts_dir: Path = project_root() / "artifacts" / "q1"
    run_name: str = "cyclegan_face_sketch"

