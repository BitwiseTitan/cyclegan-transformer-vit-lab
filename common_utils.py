"""
Shared utilities (filesystem, config IO, device selection).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """
    Return the project root directory.

    Current repo layout keeps `.py` files at the project root, so this is just
    the directory containing this file.
    """
    return Path(__file__).resolve().parent


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=to_jsonable)


def getenv_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def select_torch_device(prefer: str = "cuda") -> str:
    """
    Return 'cuda' if available, else 'mps' (Apple), else 'cpu'.
    """
    try:
        import torch

        if prefer == "cuda" and torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

