from __future__ import annotations

from typing import Iterable

import torch

from adapters.data.base import BaseDataAdapter
from core.registry import REGISTRY
from core.types import EvalBatch


@REGISTRY.register_data("bidmc_mock")
class BIDMCMockAdapter(BaseDataAdapter):
    """Synthetic BIDMC-like dataloader for architecture testing.

    Generates variable-length batches with masks to validate the pipeline.
    """

    def build_dataloader(self) -> Iterable[EvalBatch]:
        batch_size = int(self.config.get("batch_size", 8))
        num_batches = int(self.config.get("num_batches", 4))
        min_len = int(self.config.get("min_len", 512))
        max_len = int(self.config.get("max_len", 1024))
        seed = int(self.runtime.get("seed", 42))

        g = torch.Generator().manual_seed(seed)

        for _ in range(num_batches):
            lengths = torch.randint(
                low=min_len, high=max_len + 1, size=(batch_size,), generator=g
            )
            max_batch_len = int(lengths.max().item())
            ppg = torch.zeros(batch_size, max_batch_len, dtype=torch.float32)
            ecg = torch.zeros(batch_size, max_batch_len, dtype=torch.float32)
            mask = torch.zeros(batch_size, max_batch_len, dtype=torch.bool)
            sbp = torch.zeros(batch_size, dtype=torch.float32)
            dbp = torch.zeros(batch_size, dtype=torch.float32)
            stress = torch.zeros(batch_size, dtype=torch.long)

            for i, L in enumerate(lengths.tolist()):
                t = torch.linspace(0, 4.0 * 3.14159265, L)
                p = torch.sin(t) + 0.03 * torch.randn(L, generator=g)
                e = torch.sin(t + 0.1) + 0.05 * torch.randn(L, generator=g)
                ppg[i, :L] = p
                ecg[i, :L] = e
                mask[i, :L] = True

                sbp[i] = 120.0 + 3.0 * torch.randn(1, generator=g)
                dbp[i] = 75.0 + 2.0 * torch.randn(1, generator=g)
                stress[i] = int(torch.randint(0, 3, (1,), generator=g).item())

            labels = {"sbp": sbp, "dbp": dbp, "stress_level": stress}
            yield EvalBatch(ppg=ppg, ecg=ecg, mask=mask, labels=labels, meta={"source": "mock"})
