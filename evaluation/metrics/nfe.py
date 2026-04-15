from __future__ import annotations

from typing import Any, Dict

import torch

from metrics.base import BaseMetric
from core.registry import REGISTRY


@REGISTRY.register_metric("nfe")
class NFEMetric(BaseMetric):
    """
    Number of Function Evaluations (NFE) for generative models.

    Reports the NFE used during inference, which reflects the computational
    cost of the ODE/SDE solver in flow matching / diffusion models.

    NFE is passed via meta["nfe"] from the model adapter, or via config as a
    fallback. For non-generative models (e.g., UNet1D), NFE = 1 (single forward pass).
    """

    def compute(
        self,
        ecg_true: torch.Tensor,
        ecg_pred: torch.Tensor,
        mask: torch.Tensor | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        meta = meta or {}

        # Priority: meta from model adapter > config > default
        nfe = meta.get("nfe", self.config.get("nfe", 1))

        return {"nfe": float(nfe)}
