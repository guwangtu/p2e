from __future__ import annotations

from typing import Any, Dict

import torch

from .base import BaseMetric
from ..core.registry import REGISTRY


@REGISTRY.register_metric("fd_placeholder")
class FDPlaceholderMetric(BaseMetric):
    """Placeholder Frechet-like distance on first-order moments.

    Replace with real FD implementation after feature extractor is finalized.
    """

    def compute(
        self,
        ecg_true: torch.Tensor,
        ecg_pred: torch.Tensor,
        mask: torch.Tensor | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        if mask is not None:
            t = ecg_true[mask]
            p = ecg_pred[mask]
        else:
            t = ecg_true.reshape(-1)
            p = ecg_pred.reshape(-1)
        dist = torch.abs(t.mean() - p.mean())
        return {"fd_placeholder": float(dist.item())}
