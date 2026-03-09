from __future__ import annotations

from typing import Any, Dict

import torch

from .base import BaseTaskHead
from ..core.registry import REGISTRY


@REGISTRY.register_task("bp_estimation")
class BPEstimationTask(BaseTaskHead):
    """Downstream task interface example for blood pressure.

    Current behavior:
    - Reads labels `sbp` and `dbp` when provided by data adapter.
    - Uses simple signal summary as placeholder predictor.
    """

    def evaluate(
        self,
        ecg_true: torch.Tensor,
        ecg_pred: torch.Tensor,
        labels: Dict[str, torch.Tensor] | None = None,
        mask: torch.Tensor | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        if not labels or "sbp" not in labels or "dbp" not in labels:
            return {"bp_status": 0.0}

        if mask is not None:
            denom = mask.sum(dim=1).clamp_min(1)
            feat = (ecg_pred * mask).sum(dim=1) / denom
        else:
            feat = ecg_pred.mean(dim=1)

        sbp_pred = 120.0 + 10.0 * feat
        dbp_pred = 75.0 + 6.0 * feat
        sbp_mae = torch.mean(torch.abs(sbp_pred - labels["sbp"].to(feat.device)))
        dbp_mae = torch.mean(torch.abs(dbp_pred - labels["dbp"].to(feat.device)))
        return {"sbp_mae": float(sbp_mae.item()), "dbp_mae": float(dbp_mae.item())}
