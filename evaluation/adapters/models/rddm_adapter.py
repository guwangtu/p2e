from __future__ import annotations

from typing import Any, Dict

from adapters.models.base import BaseModelAdapter
from core.registry import REGISTRY
from core.types import EvalBatch, EvalPrediction


@REGISTRY.register_model("rddm")
class RDDMAdapter(BaseModelAdapter):
    """RDDM model adapter placeholder.

    Integrate `RDDM/diffusion.py` loading and sampling logic here.
    """

    def setup(self) -> None:
        raise NotImplementedError(
            "RDDMAdapter.setup is not implemented yet. Wire in load_pretrained_DPM here."
        )

    def predict(self, batch: EvalBatch) -> EvalPrediction:
        raise NotImplementedError(
            "RDDMAdapter.predict is not implemented yet. Wire in diffusion sampling here."
        )
