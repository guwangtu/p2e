from __future__ import annotations

from typing import Any, Dict

import torch

from metrics.base import BaseMetric
from core.registry import REGISTRY


@REGISTRY.register_metric("rmse")
class RMSEMetric(BaseMetric):
    def compute(
        self,
        ecg_true: torch.Tensor,
        ecg_pred: torch.Tensor,
        mask: torch.Tensor | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        diff2 = (ecg_pred - ecg_true) ** 2
        if mask is not None:
            m = mask.to(diff2.device)
            denom = m.sum().clamp_min(1)
            mse = diff2[m].sum() / denom
        else:
            mse = diff2.mean()
        return {"rmse": float(torch.sqrt(mse).item())}
