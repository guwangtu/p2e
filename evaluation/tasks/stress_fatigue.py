from __future__ import annotations

from typing import Any, Dict

import torch

from tasks.base import BaseTaskHead
from core.registry import REGISTRY


@REGISTRY.register_task("stress_fatigue")
class StressFatigueTask(BaseTaskHead):
    """Downstream stress/fatigue classification task interface.

    Current behavior:
    - Reads integer labels `stress_level` if provided.
    - Computes a placeholder accuracy from signal-energy thresholding.
    """

    def evaluate(
        self,
        ecg_true: torch.Tensor,
        ecg_pred: torch.Tensor,
        labels: Dict[str, torch.Tensor] | None = None,
        mask: torch.Tensor | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        if not labels or "stress_level" not in labels:
            return {"stress_status": 0.0}

        if mask is not None:
            denom = mask.sum(dim=1).clamp_min(1)
            energy = ((ecg_pred**2) * mask).sum(dim=1) / denom
        else:
            energy = (ecg_pred**2).mean(dim=1)

        pred_cls = torch.bucketize(energy, boundaries=torch.tensor([0.3, 0.8], device=energy.device))
        true_cls = labels["stress_level"].to(energy.device)
        acc = (pred_cls == true_cls).float().mean()
        return {"stress_acc": float(acc.item())}
