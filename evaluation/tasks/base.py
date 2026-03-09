from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


class BaseTaskHead(ABC):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def evaluate(
        self,
        ecg_true: torch.Tensor,
        ecg_pred: torch.Tensor,
        labels: Dict[str, torch.Tensor] | None = None,
        mask: torch.Tensor | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        pass
