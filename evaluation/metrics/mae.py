from __future__ import annotations

from typing import Any, Dict

import torch

from metrics.base import BaseMetric
from core.registry import REGISTRY


@REGISTRY.register_metric("mae")
class MAEMetric(BaseMetric):
    """Mean Absolute Error between predicted and true ECG."""

    def compute(
        self,
        ecg_true: torch.Tensor,
        ecg_pred: torch.Tensor,
        mask: torch.Tensor | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        diff = (ecg_pred - ecg_true).abs()
        if mask is not None:
            m = mask.to(diff.device)
            denom = m.sum().clamp_min(1)
            mae = diff[m].sum() / denom
        else:
            mae = diff.mean()
        return {"mae": float(mae.item())}
