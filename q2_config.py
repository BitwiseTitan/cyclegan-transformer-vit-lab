"""
Q2 configuration (English → Urdu translation).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common_utils import project_root


@dataclass(frozen=True)
class Q2Config:
    data_dir: Path = project_root() / "data" / "q2"
    artifacts_dir: Path = project_root() / "artifacts" / "q2"

    # Baseline approach (Phase 2):
    # - Start with mBART-50 fine-tuning for strong results fast
    model_name: str = "facebook/mbart-large-50-many-to-many-mmt"

    max_source_len: int = 128
    max_target_len: int = 128
    train_batch_size: int = 8
    eval_batch_size: int = 8
    lr: float = 3e-5
    num_train_epochs: int = 3
    seed: int = 42

