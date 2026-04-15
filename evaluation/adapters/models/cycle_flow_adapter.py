from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch

from adapters.models.base import BaseModelAdapter
from core.registry import REGISTRY
from core.types import EvalBatch, EvalPrediction


@REGISTRY.register_model("cycle_flow")
class CycleFlowAdapter(BaseModelAdapter):
    """Adapter for CycleFlow (LitCycleFlow) evaluation."""

    def __init__(self, config: Dict[str, Any], runtime: Dict[str, Any]) -> None:
        super().__init__(config, runtime)
        self.device = runtime.get("device", "cpu")
        self.model = None

    def _bootstrap_import_path(self) -> None:
        eval_dir = Path(__file__).resolve().parents[2]  # p2e/evaluation
        p2e_dir = eval_dir.parent  # p2e
        p2e_dir_str = str(p2e_dir)
        if p2e_dir_str not in sys.path:
            sys.path.insert(0, p2e_dir_str)

    def setup(self) -> None:
        self._bootstrap_import_path()
        from scripts.models.cycle_flow import LitCycleFlow

        ckpt_path = str(self.config.get("ckpt_path", "")).strip()
        if not ckpt_path:
            raise ValueError("CycleFlowAdapter requires 'ckpt_path' in config.")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        self.model = LitCycleFlow.load_from_checkpoint(ckpt_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, batch: EvalBatch) -> EvalPrediction:
        if self.model is None:
            raise RuntimeError("CycleFlowAdapter.setup must be called before predict.")

        ppg_cycles = batch.ppg.to(self.device).float()    # (B, W, L)
        rr = batch.labels.get("rr_intervals", None)

        if rr is None:
            raise ValueError("CycleFlowAdapter requires 'rr_intervals' in batch.labels")

        rr = rr.to(self.device).float()

        n_steps = int(self.config.get("n_sample_steps", 20))

        with torch.no_grad():
            # Generate per-sample
            results = []
            B = ppg_cycles.shape[0]
            for i in range(B):
                ecg_pred = self.model.generate(
                    ppg_cycles[i:i+1],
                    rr[i:i+1],
                    n_steps=n_steps,
                )
                results.append(ecg_pred)
            ecg_pred = torch.stack(results, dim=0)  # (B, total_L)

        return EvalPrediction(
            ecg_pred=ecg_pred.detach().cpu(),
            aux={"nfe": n_steps},
        )
