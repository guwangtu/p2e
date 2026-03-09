from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch

from adapters.models.base import BaseModelAdapter
from core.registry import REGISTRY
from core.types import EvalBatch, EvalPrediction


@REGISTRY.register_model("unet1d")
class UNet1DAdapter(BaseModelAdapter):
    """Adapter for `p2e/scripts/models/unet1d.py::UNet1D`."""

    def __init__(self, config: Dict[str, Any], runtime: Dict[str, Any]) -> None:
        super().__init__(config, runtime)
        self.device = runtime.get("device", "cpu")
        self.model: torch.nn.Module | None = None

    def _bootstrap_import_path(self) -> None:
        eval_dir = Path(__file__).resolve().parents[2]  # p2e/evaluation
        p2e_dir = eval_dir.parent  # p2e
        p2e_dir_str = str(p2e_dir)
        if p2e_dir_str not in sys.path:
            sys.path.insert(0, p2e_dir_str)

    def setup(self) -> None:
        self._bootstrap_import_path()
        from scripts.models.unet1d import UNet1D  # pylint: disable=import-outside-toplevel

        in_ch = int(self.config.get("in_ch", 1))
        out_ch = int(self.config.get("out_ch", 1))
        base_ch = int(self.config.get("base_ch", 32))
        depth = int(self.config.get("depth", 4))
        self.model = UNet1D(in_ch=in_ch, out_ch=out_ch, base_ch=base_ch, depth=depth).to(self.device)

        ckpt_path = str(self.config.get("ckpt_path", "")).strip()
        if ckpt_path:
            self._load_checkpoint(ckpt_path=ckpt_path)

        self.model.eval()

    def _load_checkpoint(self, ckpt_path: str) -> None:
        if self.model is None:
            raise RuntimeError("Model is not initialized before loading checkpoint.")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        strict = bool(self.config.get("strict", True))
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

        # Support Lightning checkpoints where net params are prefixed with `net.`
        if any(k.startswith("net.") for k in state_dict.keys()):
            state_dict = {
                k[len("net."):] if k.startswith("net.") else k: v for k, v in state_dict.items()
            }
        self.model.load_state_dict(state_dict, strict=strict)

    def predict(self, batch: EvalBatch) -> EvalPrediction:
        if self.model is None:
            raise RuntimeError("UNet1DAdapter.setup must be called before predict.")

        x = batch.ppg.to(self.device).float()
        if x.ndim == 2:
            x = x.unsqueeze(1)  # [B, 1, L]
        elif x.ndim != 3:
            raise ValueError(f"Expected batch.ppg shape [B,L] or [B,C,L], got {tuple(x.shape)}")

        with torch.no_grad():
            y = self.model(x)

        if y.ndim == 3 and y.shape[1] == 1:
            y = y[:, 0, :]  # [B, L]
        return EvalPrediction(ecg_pred=y.detach().cpu())
