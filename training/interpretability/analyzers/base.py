"""
interpretability/analyzers/base.py
==================================
Abstract base class shared by all interpretability analyzers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class BaseAnalyzer(ABC):
    """Common interface for all interpretability analyzers.

    Lifecycle:
        1. ``collect()`` — called once per eval batch to accumulate outputs
        2. ``analyze()`` — called once after all batches; returns aggregate stats
        3. ``visualize()`` — saves plots to *save_dir*
        4. ``explain_sample()`` — returns a one-line human-readable string
    """

    name: str = 'base'

    # Keys this analyzer needs from the prediction dict.
    required_keys: Tuple[str, ...] = ()

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear accumulated buffers."""
        self._buffers: Dict[str, list] = {k: [] for k in self.required_keys}
        self._labels: list = []

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _to_numpy(t) -> np.ndarray:
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().float().numpy()
        return np.asarray(t)

    def _stack(self, key: str) -> Optional[np.ndarray]:
        bufs = self._buffers.get(key, [])
        if not bufs:
            return None
        return np.concatenate(bufs, axis=0)

    @property
    def _all_labels(self) -> np.ndarray:
        return np.concatenate(self._labels, axis=0)

    # ── interface ──────────────────────────────────────────────────────────

    def collect(self, preds: dict, labels) -> None:
        """Accumulate one batch of predictions."""
        self._labels.append(self._to_numpy(labels))
        for k in self.required_keys:
            val = preds.get(k)
            if val is not None:
                self._buffers[k].append(self._to_numpy(val))

    @abstractmethod
    def analyze(self) -> dict:
        ...

    @abstractmethod
    def visualize(self, save_dir: str) -> None:
        ...

    @abstractmethod
    def explain_sample(self, idx: int) -> str:
        ...
