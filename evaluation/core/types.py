from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class EvalBatch:
    ppg: torch.Tensor
    ecg: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    labels: Optional[Dict[str, torch.Tensor]] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalPrediction:
    ecg_pred: torch.Tensor
    aux: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    metrics: Dict[str, float] = field(default_factory=dict)
    tasks: Dict[str, Dict[str, float]] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
