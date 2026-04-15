from __future__ import annotations

from typing import Any, Dict

import torch
import numpy as np

from metrics.base import BaseMetric
from core.registry import REGISTRY


def _dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Dynamic Time Warping distance between two 1D sequences.
    Uses O(N*M) DP. For long sequences, a window constraint is applied.
    """
    n, m = len(x), len(y)

    # For very long sequences, use Sakoe-Chiba band to limit computation
    window = max(10, abs(n - m) + int(0.1 * max(n, m)))

    # Initialize cost matrix with infinity
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window) + 1
        for j in range(j_start, j_end):
            cost = abs(x[i - 1] - y[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1],  # match
            )

    return float(dtw_matrix[n, m])


@REGISTRY.register_metric("dtw")
class DTWMetric(BaseMetric):
    """
    Dynamic Time Warping distance between predicted and true ECG.

    Computes per-sample DTW and returns the mean.
    Optionally downsamples long signals for efficiency.
    """

    def compute(
        self,
        ecg_true: torch.Tensor,
        ecg_pred: torch.Tensor,
        mask: torch.Tensor | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        if ecg_true.ndim == 1:
            ecg_true = ecg_true.unsqueeze(0)
            ecg_pred = ecg_pred.unsqueeze(0)

        max_len = int(self.config.get("max_len", 2048))
        distances = []

        for i in range(ecg_true.shape[0]):
            if mask is not None:
                m = mask[i].bool()
                y = ecg_true[i][m].cpu().numpy()
                p = ecg_pred[i][m].cpu().numpy()
            else:
                y = ecg_true[i].cpu().numpy()
                p = ecg_pred[i].cpu().numpy()

            if len(y) < 2:
                continue

            # Downsample if too long (for computational efficiency)
            if len(y) > max_len:
                step = len(y) // max_len
                y = y[::step]
                p = p[::step]

            try:
                # Try using dtw-python or fastdtw if available
                from dtw import dtw as dtw_lib
                alignment = dtw_lib(y, p)
                distances.append(float(alignment.distance))
            except ImportError:
                try:
                    from fastdtw import fastdtw
                    dist, _ = fastdtw(y, p, radius=10)
                    distances.append(float(dist))
                except ImportError:
                    # Fallback to our own implementation
                    distances.append(_dtw_distance(y, p))

        if not distances:
            return {"dtw": 0.0}

        mean_dtw = sum(distances) / len(distances)
        return {"dtw": float(mean_dtw)}
