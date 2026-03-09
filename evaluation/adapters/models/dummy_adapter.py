from __future__ import annotations

from typing import Any, Dict

import torch

from adapters.models.base import BaseModelAdapter
from core.registry import REGISTRY
from core.types import EvalBatch, EvalPrediction


@REGISTRY.register_model("dummy")
class DummyIdentityModelAdapter(BaseModelAdapter):
    """A baseline adapter for pipeline bring-up.

    It returns PPG directly as generated ECG.
    """

    def __init__(self, config: Dict[str, Any], runtime: Dict[str, Any]) -> None:
        super().__init__(config, runtime)
        self.device = runtime.get("device", "cpu")

    def setup(self) -> None:
        return

    def predict(self, batch: EvalBatch) -> EvalPrediction:
        ppg = batch.ppg.to(self.device)
        return EvalPrediction(ecg_pred=ppg.clone())
