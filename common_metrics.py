"""
Shared evaluation metrics utilities.

Q2 uses BLEU (via sacrebleu/evaluate).
Q3 uses accuracy and optional per-class metrics.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def accuracy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    preds = logits.argmax(axis=-1)
    return float((preds == labels).mean())


def topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    topk = np.argsort(-logits, axis=-1)[:, :k]
    correct = (topk == labels.reshape(-1, 1)).any(axis=1)
    return float(correct.mean())


def mean(values: Iterable[float]) -> float:
    vs = list(values)
    return float(sum(vs) / max(1, len(vs)))

