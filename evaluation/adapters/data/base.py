from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable

from core.types import EvalBatch


class BaseDataAdapter(ABC):
    def __init__(self, config: Dict[str, Any], runtime: Dict[str, Any]) -> None:
        self.config = config
        self.runtime = runtime

    @abstractmethod
    def build_dataloader(self) -> Iterable[EvalBatch]:
        pass
