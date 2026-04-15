from __future__ import annotations

from typing import Any, Dict

import torch

from metrics.base import BaseMetric
from core.registry import REGISTRY


@REGISTRY.register_metric("pearson")
class PearsonMetric(BaseMetric):
    """
    Pearson correlation coefficient between predicted and true ECG.

    Computes per-sample correlation and returns the mean.
    """

    def compute(
        self,
        ecg_true: torch.Tensor,
        ecg_pred: torch.Tensor,
        mask: torch.Tensor | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        # ecg_true, ecg_pred: (N, L)
        if ecg_true.ndim == 1:
            ecg_true = ecg_true.unsqueeze(0)
            ecg_pred = ecg_pred.unsqueeze(0)

        correlations = []
        for i in range(ecg_true.shape[0]):
            if mask is not None:
                m = mask[i].bool()
                y = ecg_true[i][m].float()
                p = ecg_pred[i][m].float()
            else:
                y = ecg_true[i].float()
                p = ecg_pred[i].float()

            if y.numel() < 2:
                continue

            y_centered = y - y.mean()
            p_centered = p - p.mean()

            num = (y_centered * p_centered).sum()
            den = torch.sqrt((y_centered ** 2).sum() * (p_centered ** 2).sum()).clamp_min(1e-8)
            correlations.append((num / den).item())

        if not correlations:
            return {"pearson": 0.0}

        mean_r = sum(correlations) / len(correlations)
        return {"pearson": float(mean_r)}
