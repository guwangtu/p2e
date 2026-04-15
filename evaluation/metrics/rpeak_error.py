from __future__ import annotations

from typing import Any, Dict, List

import torch
import numpy as np
from scipy.signal import find_peaks

from metrics.base import BaseMetric
from core.registry import REGISTRY


def _detect_r_peaks(ecg: np.ndarray, fs: int, min_distance_sec: float = 0.4) -> np.ndarray:
    """Detect R-peaks using scipy find_peaks with adaptive threshold."""
    min_distance = int(min_distance_sec * fs)
    height_thr = np.std(ecg) * 0.5
    peaks, _ = find_peaks(ecg, distance=min_distance, height=height_thr)
    return peaks


def _match_peaks(
    peaks_true: np.ndarray,
    peaks_pred: np.ndarray,
    tolerance_samples: int,
) -> List[float]:
    """
    Match predicted R-peaks to true R-peaks using greedy nearest-neighbor.
    Returns list of signed errors (in samples) for matched pairs.
    """
    if len(peaks_true) == 0 or len(peaks_pred) == 0:
        return []

    used_pred = set()
    errors = []

    for tp in peaks_true:
        dists = np.abs(peaks_pred - tp)
        order = np.argsort(dists)
        for idx in order:
            if dists[idx] > tolerance_samples:
                break
            if idx not in used_pred:
                used_pred.add(idx)
                errors.append(float(peaks_pred[idx] - tp))
                break

    return errors


@REGISTRY.register_metric("rpeak_error")
class RPeakErrorMetric(BaseMetric):
    """
    R-peak detection error between predicted and true ECG.

    Reports:
    - rpeak_error_ms:  mean absolute R-peak timing error (milliseconds)
    - rpeak_precision: fraction of predicted peaks matched to a true peak
    - rpeak_recall:    fraction of true peaks matched by a predicted peak
    - rpeak_f1:        harmonic mean of precision and recall
    """

    def compute(
        self,
        ecg_true: torch.Tensor,
        ecg_pred: torch.Tensor,
        mask: torch.Tensor | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        fs = int(self.config.get("fs", 125))
        tolerance_ms = float(self.config.get("tolerance_ms", 150))  # matching window
        tolerance_samples = int(tolerance_ms / 1000.0 * fs)

        if ecg_true.ndim == 1:
            ecg_true = ecg_true.unsqueeze(0)
            ecg_pred = ecg_pred.unsqueeze(0)

        all_errors_ms: List[float] = []
        total_tp = 0  # true peaks matched
        total_true = 0
        total_pred = 0

        for i in range(ecg_true.shape[0]):
            if mask is not None:
                m = mask[i].bool()
                y = ecg_true[i][m].cpu().numpy()
                p = ecg_pred[i][m].cpu().numpy()
            else:
                y = ecg_true[i].cpu().numpy()
                p = ecg_pred[i].cpu().numpy()

            if len(y) < fs:
                continue

            peaks_true = _detect_r_peaks(y, fs)
            peaks_pred = _detect_r_peaks(p, fs)

            total_true += len(peaks_true)
            total_pred += len(peaks_pred)

            errors = _match_peaks(peaks_true, peaks_pred, tolerance_samples)
            total_tp += len(errors)

            # Convert errors from samples to ms
            for e in errors:
                all_errors_ms.append(abs(e) / fs * 1000.0)

        # Compute summary statistics
        if all_errors_ms:
            mean_err = sum(all_errors_ms) / len(all_errors_ms)
        else:
            mean_err = float("nan")

        precision = total_tp / max(total_pred, 1)
        recall = total_tp / max(total_true, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        return {
            "rpeak_error_ms": float(mean_err),
            "rpeak_precision": float(precision),
            "rpeak_recall": float(recall),
            "rpeak_f1": float(f1),
        }
