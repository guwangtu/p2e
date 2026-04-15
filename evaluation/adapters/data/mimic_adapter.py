from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch

from adapters.data.base import BaseDataAdapter
from core.registry import REGISTRY
from core.types import EvalBatch


def _load_subject_ids(split_path: str) -> List[str]:
    """Load subject IDs from split file (python list string or one-per-line)."""
    with open(split_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    if content[0] in ["[", "("]:
        ids = ast.literal_eval(content)
        return [str(x) for x in ids]
    return [ln.strip() for ln in content.splitlines() if ln.strip()]


@REGISTRY.register_data("mimic")
class MimicDataAdapter(BaseDataAdapter):
    """
    Load real MIMIC PPG/ECG data for evaluation.

    Reads from the v1 directory layout:
      data_dir/ppg/{pid}_ppg.npy  -> (num_segments, T)
      data_dir/ecg/{pid}_ecg.npy  -> (num_segments, T)

    Config keys:
      data_dir:       path to MIMIC-BP dataset root
      split:          split file name (default: test_subjects.txt)
      batch_size:     samples per batch (default: 8)
      max_samples:    cap total samples (default: 0 = all)
      num_segments_per_patient: segments to use per patient (default: 30)
      fs:             sampling rate (default: 125)
    """

    def build_dataloader(self) -> Iterable[EvalBatch]:
        data_dir = str(self.config.get("data_dir", ""))
        if not data_dir:
            raise ValueError("MimicDataAdapter requires 'data_dir' in config.")

        split_file = str(self.config.get("split", "test_subjects.txt"))
        split_path = split_file if os.path.isabs(split_file) else os.path.join(data_dir, split_file)

        batch_size = int(self.config.get("batch_size", 8))
        max_samples = int(self.config.get("max_samples", 0))
        num_segments = int(self.config.get("num_segments_per_patient", 30))
        fs = int(self.config.get("fs", 125))

        patient_ids = _load_subject_ids(split_path)

        # Collect all (pid, seg_idx) pairs
        samples = []
        for pid in patient_ids:
            ppg_path = os.path.join(data_dir, "ppg", f"{pid}_ppg.npy")
            ecg_path = os.path.join(data_dir, "ecg", f"{pid}_ecg.npy")
            if not os.path.exists(ppg_path) or not os.path.exists(ecg_path):
                continue
            for idx in range(num_segments):
                samples.append((pid, idx, ppg_path, ecg_path))

        if max_samples > 0:
            samples = samples[:max_samples]

        # Yield batches
        for batch_start in range(0, len(samples), batch_size):
            batch_items = samples[batch_start: batch_start + batch_size]
            ppg_list, ecg_list = [], []

            for pid, idx, ppg_path, ecg_path in batch_items:
                ppg_arr = np.load(ppg_path)
                ecg_arr = np.load(ecg_path)
                if idx >= ppg_arr.shape[0] or idx >= ecg_arr.shape[0]:
                    continue
                ppg_list.append(torch.from_numpy(ppg_arr[idx].astype(np.float32)))
                ecg_list.append(torch.from_numpy(ecg_arr[idx].astype(np.float32)))

            if not ppg_list:
                continue

            # Pad to same length within batch
            max_len = max(p.shape[-1] for p in ppg_list)
            B = len(ppg_list)
            ppg_batch = torch.zeros(B, max_len)
            ecg_batch = torch.zeros(B, max_len)
            mask_batch = torch.zeros(B, max_len, dtype=torch.bool)

            for i, (p, e) in enumerate(zip(ppg_list, ecg_list)):
                L = p.shape[-1]
                ppg_batch[i, :L] = p
                ecg_batch[i, :L] = e
                mask_batch[i, :L] = True

            yield EvalBatch(
                ppg=ppg_batch,
                ecg=ecg_batch,
                mask=mask_batch,
                labels={"fs": torch.tensor([fs] * B)},
                meta={"source": "mimic", "fs": fs},
            )


@REGISTRY.register_data("mimic_cycle")
class MimicCycleDataAdapter(BaseDataAdapter):
    """
    Load preprocessed MIMIC cycle data for CycleFlow evaluation.

    Reads from the v2 cycle directory layout:
      data_dir/cycles/{pid}_ppg_cycles.npy    -> (N_cycles, L)
      data_dir/cycles/{pid}_ecg_cycles.npy    -> (N_cycles, L)
      data_dir/cycles/{pid}_rr_intervals.npy  -> (N_cycles,)

    Each sample is a window of W consecutive cycles.

    Config keys:
      data_dir:       path to MIMIC-BP dataset root (with cycles/ subdir)
      split:          split file name (default: test_subjects.txt)
      batch_size:     windows per batch (default: 4)
      window_size:    cycles per window (default: 32)
      cycle_len:      resampled cycle length (default: 256)
      overlap_len:    overlap samples (default: 32)
      max_samples:    cap total windows (default: 0 = all)
      fs:             sampling rate (default: 125)
    """

    def build_dataloader(self) -> Iterable[EvalBatch]:
        data_dir = str(self.config.get("data_dir", ""))
        if not data_dir:
            raise ValueError("MimicCycleDataAdapter requires 'data_dir' in config.")

        cycle_dir = os.path.join(data_dir, "cycles")
        if not os.path.exists(cycle_dir):
            raise FileNotFoundError(
                f"cycles/ directory not found at {cycle_dir}. "
                f"Run preprocess_and_save() first."
            )

        split_file = str(self.config.get("split", "test_subjects.txt"))
        split_path = split_file if os.path.isabs(split_file) else os.path.join(data_dir, split_file)

        batch_size = int(self.config.get("batch_size", 4))
        W = int(self.config.get("window_size", 32))
        cycle_len = int(self.config.get("cycle_len", 256))
        Lo = int(self.config.get("overlap_len", 32))
        max_samples = int(self.config.get("max_samples", 0))
        fs = int(self.config.get("fs", 125))

        patient_ids = _load_subject_ids(split_path)

        # Build window index: (pid, start_cycle)
        windows: List[tuple] = []
        for pid in patient_ids:
            ppg_path = os.path.join(cycle_dir, f"{pid}_ppg_cycles.npy")
            if not os.path.exists(ppg_path):
                continue
            n_cycles = np.load(ppg_path, mmap_mode="r").shape[0]
            if n_cycles < W + 1:
                continue
            for start in range(1, n_cycles - W + 1, W):  # non-overlapping for eval
                windows.append((pid, start))

        if max_samples > 0:
            windows = windows[:max_samples]

        # Yield batches
        for batch_start in range(0, len(windows), batch_size):
            batch_items = windows[batch_start: batch_start + batch_size]
            B = len(batch_items)

            ppg_batch = torch.zeros(B, W, cycle_len)
            ecg_batch = torch.zeros(B, W, cycle_len)
            rr_batch = torch.zeros(B, W)
            overlap_batch = torch.zeros(B, W, Lo)

            for i, (pid, start) in enumerate(batch_items):
                ppg_all = np.load(os.path.join(cycle_dir, f"{pid}_ppg_cycles.npy"), mmap_mode="r")
                ecg_all = np.load(os.path.join(cycle_dir, f"{pid}_ecg_cycles.npy"), mmap_mode="r")
                rr_all = np.load(os.path.join(cycle_dir, f"{pid}_rr_intervals.npy"), mmap_mode="r")

                ppg_batch[i] = torch.from_numpy(np.array(ppg_all[start:start + W]))
                ecg_batch[i] = torch.from_numpy(np.array(ecg_all[start:start + W]))
                rr_batch[i] = torch.from_numpy(np.array(rr_all[start:start + W]))

                # Overlap: tail of previous cycle
                for t in range(W):
                    prev_idx = start + t - 1
                    prev_ecg = np.array(ecg_all[prev_idx])
                    overlap_batch[i, t] = torch.from_numpy(prev_ecg[-Lo:])

            yield EvalBatch(
                ppg=ppg_batch,           # (B, W, L) — cycle sequences
                ecg=ecg_batch,           # (B, W, L) — ground truth
                mask=None,               # all valid
                labels={
                    "rr_intervals": rr_batch,      # (B, W)
                    "ecg_overlap": overlap_batch,   # (B, W, Lo)
                    "fs": torch.tensor([fs] * B),
                },
                meta={"source": "mimic_cycle", "fs": fs, "window_size": W},
            )
