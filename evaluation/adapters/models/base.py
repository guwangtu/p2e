from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from core.types import EvalBatch, EvalPrediction


class BaseModelAdapter(ABC):
    def __init__(self, config: Dict[str, Any], runtime: Dict[str, Any]) -> None:
        self.config = config
        self.runtime = runtime

    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def predict(self, batch: EvalBatch) -> EvalPrediction:
        pass
