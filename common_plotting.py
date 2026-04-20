"""
Shared plotting helpers (loss curves, confusion matrices).

Keep outputs under `artifacts/<task>/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt

from common_utils import ensure_dir


def save_curves(
    out_path: str | Path,
    curves: Mapping[str, Sequence[float]],
    title: str = "",
    xlabel: str = "epoch",
    ylabel: str = "value",
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    plt.figure(figsize=(7, 4))
    for name, ys in curves.items():
        plt.plot(list(range(1, len(ys) + 1)), list(ys), label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

