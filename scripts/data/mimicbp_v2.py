"""
MimicBP v2 DataModule — 逐心动周期的 PPG→ECG 数据集。

与 v1 的区别：
  - 数据以心动周期序列组织，而非固定长度片段
  - 每个样本是连续 W 个周期的滑动窗口
  - 包含 RR 间期、ECG 关键点标注、overlap 片段
  - 支持信号级预处理（滤波、质量筛选）

预处理后的目录约定：
  data_dir/
    cycles/
      {pid}_ppg_cycles.npy    # shape: (N_cycles, L)
      {pid}_ecg_cycles.npy    # shape: (N_cycles, L)
      {pid}_rr_intervals.npy  # shape: (N_cycles,)  单位秒
      {pid}_keypoints.npy     # shape: (N_cycles, 10) -> P/Q/R/S/T 位置+幅度
      {pid}_cycle_meta.npz    # 可选: sqi scores, original lengths 等
    train_subjects.txt
    val_subjects.txt
    test_subjects.txt

也支持从原始信号在线预处理（raw 模式）：
  data_dir/
    ppg/{pid}_ppg.npy         # shape: (num_segments, T_raw)
    ecg/{pid}_ecg.npy         # shape: (num_segments, T_raw)
"""

from __future__ import annotations

import ast
import os
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import lightning as L
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Signal processing utilities
# ---------------------------------------------------------------------------

def bandpass_filter(sig: np.ndarray, low: float, high: float, fs: int, order: int = 4) -> np.ndarray:
    """Butterworth bandpass filter."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, sig, axis=-1).astype(np.float32)


def notch_filter(sig: np.ndarray, freq: float, fs: int, Q: float = 30.0) -> np.ndarray:
    """Notch filter to remove powerline interference."""
    from scipy.signal import iirnotch, sosfiltfilt
    b, a = iirnotch(freq, Q, fs)
    from scipy.signal import sosfilt_zi, lfilter
    return lfilter(b, a, sig, axis=-1).astype(np.float32)


def detect_r_peaks(ecg: np.ndarray, fs: int) -> np.ndarray:
    """
    Detect R-peaks in a single-lead ECG signal.
    Returns array of R-peak sample indices.
    """
    try:
        import neurokit2 as nk
        _, info = nk.ecg_peaks(ecg, sampling_rate=fs)
        peaks = np.array(info["ECG_R_Peaks"], dtype=np.int64)
    except ImportError:
        # fallback: simple threshold-based detection via scipy
        from scipy.signal import find_peaks
        # 简单高通 + 平方 + 峰值检测
        min_distance = int(0.4 * fs)  # 最小 RR = 0.4s
        peaks, _ = find_peaks(ecg, distance=min_distance, height=np.std(ecg) * 0.5)
        peaks = np.array(peaks, dtype=np.int64)
    return peaks


def delineate_ecg(ecg_cycle: np.ndarray, fs: int) -> np.ndarray:
    """
    Detect P, Q, R, S, T keypoints within a single resampled ECG cycle.
    Returns: array of shape (10,) -> [p_pos, q_pos, r_pos, s_pos, t_pos,
                                       p_amp, q_amp, r_amp, s_amp, t_amp]
    Positions are normalized to [0, 1] within the cycle.
    If detection fails for a wave, returns NaN for that entry.
    """
    L = len(ecg_cycle)
    result = np.full(10, np.nan, dtype=np.float32)

    try:
        import neurokit2 as nk
        # Wrap cycle into a longer signal for neurokit (it needs context)
        padded = np.tile(ecg_cycle, 3)
        signals, info = nk.ecg_delineate(
            padded, sampling_rate=fs, method="dwt", show=False
        )
        # Take the middle cycle's annotations
        offset = L
        wave_names = ["P", "Q", "R", "S", "T"]  # note: neurokit uses ECG_{wave}_Peaks/Onsets
        for i, w in enumerate(["P", "Q", "R", "S", "T"]):
            key_peak = f"ECG_{w}_Peaks"
            if key_peak in info:
                peaks = [p for p in info[key_peak] if not np.isnan(p) and offset <= p < offset + L]
                if peaks:
                    pos = peaks[0] - offset
                    result[i] = pos / L  # normalized position
                    result[i + 5] = ecg_cycle[int(pos)]  # amplitude
    except Exception:
        # Fallback: simple heuristic — find R as max, Q as min before R, S as min after R
        r_idx = int(np.argmax(ecg_cycle))
        result[2] = r_idx / L
        result[7] = ecg_cycle[r_idx]

        # Q: minimum in [R-0.1s, R]
        q_start = max(0, r_idx - int(0.1 * fs))
        if q_start < r_idx:
            q_idx = q_start + int(np.argmin(ecg_cycle[q_start:r_idx]))
            result[1] = q_idx / L
            result[6] = ecg_cycle[q_idx]

        # S: minimum in [R, R+0.1s]
        s_end = min(L, r_idx + int(0.1 * fs))
        if r_idx < s_end:
            s_idx = r_idx + int(np.argmin(ecg_cycle[r_idx:s_end]))
            result[3] = s_idx / L
            result[8] = ecg_cycle[s_idx]

    return result


def detect_ppg_peaks(ppg: np.ndarray, fs: int, ptt_compensation_ms: float = 0.0) -> np.ndarray:
    """
    Detect systolic peaks in a PPG signal.
    PPG peaks correspond to ECG R-peaks with a fixed delay (pulse transit time).

    Args:
        ppg: PPG signal (T,)
        fs: sampling rate
        ptt_compensation_ms: shift peaks backward by this many ms to approximate
            ECG R-peak timing. Typical PTT is 200-300ms.

    Returns array of peak sample indices (compensated for PTT if requested).
    """
    try:
        import neurokit2 as nk
        _, info = nk.ppg_peaks(ppg, sampling_rate=fs)
        peaks = np.array(info["PPG_Peaks"], dtype=np.int64)
    except (ImportError, Exception):
        from scipy.signal import find_peaks
        min_distance = int(0.4 * fs)  # min RR = 0.4s (~150 bpm)
        peaks, _ = find_peaks(ppg, distance=min_distance, prominence=np.std(ppg) * 0.3)
        peaks = np.array(peaks, dtype=np.int64)

    # Compensate for pulse transit time: shift peaks backward
    if ptt_compensation_ms > 0 and len(peaks) > 0:
        shift = int(ptt_compensation_ms * fs / 1000)
        peaks = np.maximum(peaks - shift, 0)

    return peaks


def segment_ppg_cycles(
    ppg: np.ndarray,
    fs: int,
    cycle_len: int = 256,
    rr_min: float = 0.4,
    rr_max: float = 1.5,
    ptt_compensation_ms: float = 0.0,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Segment PPG signal into heartbeat cycles using PPG peak detection (no ECG needed).

    Args:
        ppg: PPG signal, shape (T,)
        fs: sampling rate
        cycle_len: resample each cycle to this length
        rr_min/rr_max: valid RR interval range in seconds
        ptt_compensation_ms: pulse transit time compensation in ms

    Returns:
        dict with keys: ppg_cycles, rr_intervals, original_lengths, cycle_starts
        or None if too few valid cycles.
    """
    from scipy.interpolate import CubicSpline

    peaks = detect_ppg_peaks(ppg, fs, ptt_compensation_ms=ptt_compensation_ms)
    if len(peaks) < 3:
        return None

    ppg_cycles = []
    rr_intervals = []
    original_lengths = []
    cycle_starts = []  # start sample index in original signal

    for i in range(1, len(peaks) - 1):
        rr_prev = (peaks[i] - peaks[i - 1]) / fs
        rr_next = (peaks[i + 1] - peaks[i]) / fs
        rr = rr_next

        if not (rr_min <= rr <= rr_max) or not (rr_min <= rr_prev <= rr_max):
            continue

        # Cycle boundaries: 30% before peak, 70% after peak
        rr_samples = peaks[i + 1] - peaks[i]
        start = peaks[i] - int(0.3 * rr_samples)
        end = peaks[i] + int(0.7 * rr_samples)

        if start < 0 or end > len(ppg):
            continue

        ppg_seg = ppg[start:end]
        orig_len = len(ppg_seg)

        if orig_len < 10:
            continue

        t_orig = np.linspace(0, 1, orig_len)
        t_new = np.linspace(0, 1, cycle_len)
        ppg_resampled = CubicSpline(t_orig, ppg_seg)(t_new).astype(np.float32)

        ppg_cycles.append(ppg_resampled)
        rr_intervals.append(rr)
        original_lengths.append(orig_len)
        cycle_starts.append(start)

    if len(ppg_cycles) < 3:
        return None

    return {
        "ppg_cycles": np.stack(ppg_cycles),
        "rr_intervals": np.array(rr_intervals, dtype=np.float32),
        "original_lengths": np.array(original_lengths, dtype=np.int32),
        "cycle_starts": np.array(cycle_starts, dtype=np.int32),
    }


def preprocess_ppg_for_inference(
    ppg: np.ndarray,
    fs: int = 125,
    cycle_len: int = 256,
    window_size: int = 32,
    overlap_len: int = 32,
    ppg_bandpass: Tuple[float, float] = (0.5, 8.0),
    powerline_freq: float = 50.0,
    ptt_compensation_ms: float = 250.0,
) -> Optional[Dict[str, Any]]:
    """
    Preprocess a raw PPG signal for inference with CycleFlow or CardioWorldModel.
    No ECG required — uses PPG peak detection for cycle segmentation.

    Args:
        ppg: raw PPG signal, shape (T,), e.g. (3750,) for 30s @ 125Hz
        fs: sampling rate in Hz
        cycle_len: resample each cycle to this length
        window_size: number of consecutive cycles per model input window
        overlap_len: overlap length for ECG decoder conditioning
        ppg_bandpass: bandpass filter range for PPG
        powerline_freq: notch filter frequency
        ptt_compensation_ms: pulse transit time compensation (ms).
            PPG peaks are delayed ~200-300ms from ECG R-peaks.
            Set to 0 to disable.

    Returns:
        dict ready for model.generate() / model.imagine() + reconstruction info:
        {
            "ppg_cycles":      Tensor (n_windows, W, cycle_len),
            "rr_intervals":    Tensor (n_windows, W),
            "n_cycles":        int,
            "n_windows":       int,
            "original_lengths": ndarray (N,) — original sample count per cycle
            "cycle_starts":    ndarray (N,) — start index in filtered signal
            "signal_length":   int — length of the original signal
        }
        or None if extraction fails.
    """
    import torch

    signal_length = len(ppg)

    # 1. Filter
    ppg = ppg.astype(np.float64)
    ppg = bandpass_filter(ppg, *ppg_bandpass, fs)
    ppg = notch_filter(ppg, powerline_freq, fs)

    # 2. Normalize
    ppg = (ppg - ppg.mean()) / (ppg.std() + 1e-8)

    # 3. Segment into cycles using PPG peaks with PTT compensation
    result = segment_ppg_cycles(ppg, fs, cycle_len,
                                ptt_compensation_ms=ptt_compensation_ms)
    if result is None:
        return None

    ppg_cycles = result["ppg_cycles"]           # (N, cycle_len)
    rr_intervals = result["rr_intervals"]       # (N,)
    original_lengths = result["original_lengths"]  # (N,)
    cycle_starts = result["cycle_starts"]        # (N,)
    n_cycles = len(ppg_cycles)

    if n_cycles < window_size:
        pad_n = window_size - n_cycles
        ppg_cycles = np.concatenate([
            ppg_cycles,
            np.tile(ppg_cycles[-1:], (pad_n, 1)),
        ], axis=0)
        rr_intervals = np.concatenate([
            rr_intervals,
            np.tile(rr_intervals[-1:], pad_n),
        ], axis=0)
        original_lengths = np.concatenate([
            original_lengths,
            np.tile(original_lengths[-1:], pad_n),
        ], axis=0)
        cycle_starts = np.concatenate([
            cycle_starts,
            np.tile(cycle_starts[-1:], pad_n),
        ], axis=0)

    # 4. Build windows
    n_windows = max(1, (len(ppg_cycles) - window_size) // window_size + 1)
    windows_ppg, windows_rr = [], []
    windows_orig_len, windows_starts = [], []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        if end > len(ppg_cycles):
            start = len(ppg_cycles) - window_size
            end = len(ppg_cycles)
        windows_ppg.append(ppg_cycles[start:end])
        windows_rr.append(rr_intervals[start:end])
        windows_orig_len.append(original_lengths[start:end])
        windows_starts.append(cycle_starts[start:end])

    return {
        "ppg_cycles": torch.from_numpy(np.stack(windows_ppg)).float(),
        "rr_intervals": torch.from_numpy(np.stack(windows_rr)).float(),
        "n_cycles": n_cycles,
        "n_windows": n_windows,
        "original_lengths": np.stack(windows_orig_len),   # (n_windows, W)
        "cycle_starts": np.stack(windows_starts),          # (n_windows, W)
        "signal_length": signal_length,
    }


def reconstruct_ecg_from_cycles(
    ecg_cycles: np.ndarray,
    original_lengths: np.ndarray,
    cycle_starts: np.ndarray,
    signal_length: int,
    cycle_len: int = 256,
    n_actual_cycles: int = None,
) -> np.ndarray:
    """
    Reconstruct a full-length ECG signal from generated cycles by placing
    each cycle back at its original position with proper resampling.

    This avoids the naive concatenation + interpolation approach that
    destroys temporal alignment.

    Args:
        ecg_cycles: (W*L,) or (W, L) — generated ECG cycles
        original_lengths: (W,) — original sample count for each cycle
        cycle_starts: (W,) — start index in original signal for each cycle
        signal_length: total length of the output signal
        cycle_len: resampled cycle length (256)
        n_actual_cycles: if padding was used, only use first n cycles

    Returns:
        ecg_out: (signal_length,) — reconstructed ECG signal
    """
    from scipy.interpolate import CubicSpline

    if ecg_cycles.ndim == 1:
        W = len(original_lengths)
        ecg_cycles = ecg_cycles[:W * cycle_len].reshape(W, cycle_len)

    W = len(ecg_cycles)
    if n_actual_cycles is not None:
        W = min(W, n_actual_cycles)

    ecg_out = np.zeros(signal_length, dtype=np.float32)
    weight = np.zeros(signal_length, dtype=np.float32)

    for i in range(W):
        orig_len = int(original_lengths[i])
        start = int(cycle_starts[i])
        end = start + orig_len

        if end > signal_length:
            end = signal_length
            orig_len = end - start

        if orig_len < 2:
            continue

        # Resample from cycle_len back to original length
        t_cycle = np.linspace(0, 1, cycle_len)
        t_orig = np.linspace(0, 1, orig_len)
        ecg_resampled = CubicSpline(t_cycle, ecg_cycles[i])(t_orig).astype(np.float32)

        # Overlap-add with triangular window for smooth blending
        win = np.ones(orig_len, dtype=np.float32)
        # Taper edges for smooth blending at overlapping boundaries
        taper = min(orig_len // 4, 10)
        if taper > 0:
            win[:taper] = np.linspace(0, 1, taper)
            win[-taper:] = np.linspace(1, 0, taper)

        ecg_out[start:end] += ecg_resampled * win
        weight[start:end] += win

    # Normalize by overlap count
    mask = weight > 0
    ecg_out[mask] /= weight[mask]

    return ecg_out


def segment_cycles(
    ppg: np.ndarray,
    ecg: np.ndarray,
    fs: int,
    cycle_len: int = 256,
    rr_min: float = 0.4,
    rr_max: float = 1.5,
    use_ppg_peaks: bool = False,
    ptt_compensation_ms: float = 250.0,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Segment synchronized PPG/ECG signals into heartbeat cycles.

    Args:
        ppg: PPG signal, shape (T,)
        ecg: ECG signal, shape (T,)
        fs: sampling rate
        cycle_len: resample each cycle to this length
        rr_min/rr_max: valid RR interval range in seconds
        use_ppg_peaks: if True, use PPG peaks (with PTT compensation) instead of ECG R-peaks.
            This ensures training and inference use the same segmentation strategy.
        ptt_compensation_ms: PTT compensation when using PPG peaks (default 250ms).

    Returns:
        dict with keys: ppg_cycles, ecg_cycles, rr_intervals, keypoints, original_lengths
        or None if too few valid cycles.
    """
    from scipy.interpolate import CubicSpline

    if use_ppg_peaks:
        r_peaks = detect_ppg_peaks(ppg, fs, ptt_compensation_ms=ptt_compensation_ms)
    else:
        r_peaks = detect_r_peaks(ecg, fs)
    if len(r_peaks) < 3:
        return None

    ppg_cycles = []
    ecg_cycles = []
    rr_intervals = []
    keypoints = []
    original_lengths = []

    for i in range(1, len(r_peaks) - 1):
        rr_prev = (r_peaks[i] - r_peaks[i - 1]) / fs
        rr_next = (r_peaks[i + 1] - r_peaks[i]) / fs
        rr = rr_next  # current cycle's RR interval

        # Quality check: valid RR range
        if not (rr_min <= rr <= rr_max) or not (rr_min <= rr_prev <= rr_max):
            continue

        # Cycle boundaries: 30% before R, 70% after R
        rr_samples = r_peaks[i + 1] - r_peaks[i]
        start = r_peaks[i] - int(0.3 * rr_samples)
        end = r_peaks[i] + int(0.7 * rr_samples)

        if start < 0 or end > len(ecg):
            continue

        ecg_seg = ecg[start:end]
        ppg_seg = ppg[start:end]
        orig_len = len(ecg_seg)

        if orig_len < 10:
            continue

        # Resample to fixed length via cubic spline
        t_orig = np.linspace(0, 1, orig_len)
        t_new = np.linspace(0, 1, cycle_len)

        ecg_resampled = CubicSpline(t_orig, ecg_seg)(t_new).astype(np.float32)
        ppg_resampled = CubicSpline(t_orig, ppg_seg)(t_new).astype(np.float32)

        # ECG keypoint detection on resampled cycle
        kp = delineate_ecg(ecg_resampled, fs)

        ppg_cycles.append(ppg_resampled)
        ecg_cycles.append(ecg_resampled)
        rr_intervals.append(rr)
        keypoints.append(kp)
        original_lengths.append(orig_len)

    if len(ppg_cycles) < 3:
        return None

    return {
        "ppg_cycles": np.stack(ppg_cycles),       # (N, L)
        "ecg_cycles": np.stack(ecg_cycles),        # (N, L)
        "rr_intervals": np.array(rr_intervals, dtype=np.float32),  # (N,)
        "keypoints": np.stack(keypoints),           # (N, 10)
        "original_lengths": np.array(original_lengths, dtype=np.int32),  # (N,)
    }


def preprocess_and_save(
    data_dir: str,
    output_dir: str,
    fs: int = 125,
    cycle_len: int = 256,
    ecg_bandpass: Tuple[float, float] = (0.5, 40.0),
    ppg_bandpass: Tuple[float, float] = (0.5, 8.0),
    powerline_freq: float = 50.0,
    files: Tuple[str, str] = ("ppg", "ecg"),
    use_ppg_peaks: bool = False,
    ptt_compensation_ms: float = 250.0,
):
    """
    Batch preprocess raw PPG/ECG npy files into cycle-segmented format.
    Reads from v1 directory layout, writes to v2 cycle layout.
    """
    out_path = Path(output_dir) / "cycles"
    out_path.mkdir(parents=True, exist_ok=True)

    ppg_dir = Path(data_dir) / files[0]
    ecg_dir = Path(data_dir) / files[1]

    # Discover all patient IDs from ppg directory
    ppg_files = sorted(ppg_dir.glob(f"*_{files[0]}.npy"))
    pids = [f.stem.replace(f"_{files[0]}", "") for f in ppg_files]

    stats = {"total": len(pids), "success": 0, "failed": 0, "total_cycles": 0}

    for pid in pids:
        ppg_path = ppg_dir / f"{pid}_{files[0]}.npy"
        ecg_path = ecg_dir / f"{pid}_{files[1]}.npy"

        if not ecg_path.exists():
            stats["failed"] += 1
            continue

        ppg_raw = np.load(str(ppg_path))  # (num_segments, T)
        ecg_raw = np.load(str(ecg_path))

        all_ppg_cycles = []
        all_ecg_cycles = []
        all_rr = []
        all_kp = []
        all_orig_len = []

        n_segments = min(ppg_raw.shape[0], ecg_raw.shape[0])
        for seg_idx in range(n_segments):
            ppg_seg = ppg_raw[seg_idx].astype(np.float64)
            ecg_seg = ecg_raw[seg_idx].astype(np.float64)

            # Filtering
            try:
                ecg_seg = bandpass_filter(ecg_seg, *ecg_bandpass, fs)
                ecg_seg = notch_filter(ecg_seg, powerline_freq, fs)
                ppg_seg = bandpass_filter(ppg_seg, *ppg_bandpass, fs)
            except Exception:
                continue

            # Normalize per segment
            ecg_seg = (ecg_seg - ecg_seg.mean()) / (ecg_seg.std() + 1e-8)
            ppg_seg = (ppg_seg - ppg_seg.mean()) / (ppg_seg.std() + 1e-8)

            result = segment_cycles(ppg_seg, ecg_seg, fs, cycle_len,
                                    use_ppg_peaks=use_ppg_peaks,
                                    ptt_compensation_ms=ptt_compensation_ms)
            if result is None:
                continue

            all_ppg_cycles.append(result["ppg_cycles"])
            all_ecg_cycles.append(result["ecg_cycles"])
            all_rr.append(result["rr_intervals"])
            all_kp.append(result["keypoints"])
            all_orig_len.append(result["original_lengths"])

        if not all_ppg_cycles:
            stats["failed"] += 1
            continue

        ppg_all = np.concatenate(all_ppg_cycles, axis=0)
        ecg_all = np.concatenate(all_ecg_cycles, axis=0)
        rr_all = np.concatenate(all_rr, axis=0)
        kp_all = np.concatenate(all_kp, axis=0)
        orig_all = np.concatenate(all_orig_len, axis=0)

        np.save(str(out_path / f"{pid}_ppg_cycles.npy"), ppg_all)
        np.save(str(out_path / f"{pid}_ecg_cycles.npy"), ecg_all)
        np.save(str(out_path / f"{pid}_rr_intervals.npy"), rr_all)
        np.save(str(out_path / f"{pid}_keypoints.npy"), kp_all)
        np.savez(
            str(out_path / f"{pid}_cycle_meta.npz"),
            original_lengths=orig_all,
            num_cycles=len(ppg_all),
        )

        stats["success"] += 1
        stats["total_cycles"] += len(ppg_all)
        print(f"  [{pid}] {len(ppg_all)} cycles extracted")

    print(f"\n[Preprocess] Done. {stats['success']}/{stats['total']} patients, "
          f"{stats['total_cycles']} total cycles, {stats['failed']} failed.")
    return stats


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _load_subject_ids(split_path: str) -> List[str]:
    """兼容 python list 字符串或每行一个 pid 两种格式。"""
    with open(split_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    if content[0] in ["[", "("]:
        ids = ast.literal_eval(content)
        if isinstance(ids, (list, tuple)):
            return [str(x) for x in ids]
        raise ValueError(f"split file parsed but not list/tuple: {type(ids)}")
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    return [str(x) for x in lines]


class MimicBPCycleDataset(Dataset):
    """
    逐心动周期的 PPG→ECG 数据集。

    每个样本是连续 W 个心动周期的滑动窗口，返回：
    {
        "ppg_cycles":    (W, L),     # PPG 周期序列
        "ecg_cycles":    (W, L),     # ECG 周期序列 (target)
        "rr_intervals":  (W,),       # RR 间期序列 (秒)
        "keypoints":     (W, 10),    # ECG 关键点 (P/Q/R/S/T pos+amp)
        "ecg_overlap":   (W, L_o),   # 上一周期尾部 overlap 片段
        "keypoint_mask": (W,),       # 关键点是否有效 (无 NaN)
        "meta": {...}
    }
    """

    def __init__(
        self,
        data_dir: str,
        split_file: str,
        window_size: int = 32,
        cycle_len: int = 256,
        overlap_len: int = 32,
        stride: int = 16,
        fs: int = 125,
        transform: Optional[Callable] = None,
    ):
        self.data_dir = data_dir
        self.cycle_dir = os.path.join(data_dir, "cycles")
        self.window_size = window_size
        self.cycle_len = cycle_len
        self.overlap_len = overlap_len
        self.stride = stride
        self.fs = fs
        self.transform = transform

        split_path = split_file if os.path.isabs(split_file) else os.path.join(data_dir, split_file)
        self.patient_ids = _load_subject_ids(split_path)

        # Build index: (pid, start_cycle_idx)
        self.samples: List[Tuple[str, int]] = []
        self._cycle_counts: Dict[str, int] = {}

        for pid in self.patient_ids:
            ppg_path = os.path.join(self.cycle_dir, f"{pid}_ppg_cycles.npy")
            if not os.path.exists(ppg_path):
                warnings.warn(f"Missing cycle data for patient {pid}, skipping.")
                continue

            # Only read shape, not full data (memory-mapped)
            ppg_arr = np.load(ppg_path, mmap_mode="r")
            n_cycles = ppg_arr.shape[0]
            self._cycle_counts[pid] = n_cycles

            if n_cycles < self.window_size + 1:
                # Need +1 because the first cycle needs a "previous" overlap
                continue

            # Sliding windows with stride, starting from index 1 (so index 0 can provide overlap)
            for start in range(1, n_cycles - self.window_size + 1, self.stride):
                self.samples.append((pid, start))

        print(
            f"[MimicBPCycleDataset] {len(self.samples)} windows "
            f"(W={window_size}, stride={stride}) "
            f"from {len(self._cycle_counts)} patients. split={split_path}"
        )

        # Cache for memory-mapped arrays (per-worker, set in worker_init or lazily)
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}

    def _get_patient_data(self, pid: str) -> Dict[str, np.ndarray]:
        """Lazy-load and cache patient cycle data (memory-mapped)."""
        if pid not in self._cache:
            cd = self.cycle_dir
            self._cache[pid] = {
                "ppg": np.load(os.path.join(cd, f"{pid}_ppg_cycles.npy"), mmap_mode="r"),
                "ecg": np.load(os.path.join(cd, f"{pid}_ecg_cycles.npy"), mmap_mode="r"),
                "rr": np.load(os.path.join(cd, f"{pid}_rr_intervals.npy"), mmap_mode="r"),
                "kp": np.load(os.path.join(cd, f"{pid}_keypoints.npy"), mmap_mode="r"),
            }
        return self._cache[pid]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pid, start = self.samples[idx]
        W = self.window_size
        Lo = self.overlap_len

        data = self._get_patient_data(pid)

        # Window: cycles [start, start+W)
        ppg_cycles = np.array(data["ppg"][start: start + W], dtype=np.float32)  # (W, L)
        ecg_cycles = np.array(data["ecg"][start: start + W], dtype=np.float32)  # (W, L)
        rr_intervals = np.array(data["rr"][start: start + W], dtype=np.float32)  # (W,)
        keypoints = np.array(data["kp"][start: start + W], dtype=np.float32)     # (W, 10)

        # Overlap: for each cycle t in window, take the tail Lo points of cycle t-1
        # cycle t-1 for the first cycle in window is at index start-1
        ecg_overlap = np.zeros((W, Lo), dtype=np.float32)
        for t in range(W):
            prev_idx = start + t - 1  # always >= 0 since start >= 1
            prev_ecg = np.array(data["ecg"][prev_idx], dtype=np.float32)
            ecg_overlap[t] = prev_ecg[-Lo:]

        # Keypoint validity mask: True if no NaN in keypoint vector
        keypoint_mask = ~np.any(np.isnan(keypoints), axis=-1)  # (W,)
        # Replace NaN with 0 for tensor compatibility
        keypoints = np.nan_to_num(keypoints, nan=0.0)

        sample = {
            "ppg_cycles": torch.from_numpy(ppg_cycles),       # (W, L)
            "ecg_cycles": torch.from_numpy(ecg_cycles),       # (W, L)
            "rr_intervals": torch.from_numpy(rr_intervals),   # (W,)
            "keypoints": torch.from_numpy(keypoints),          # (W, 10)
            "ecg_overlap": torch.from_numpy(ecg_overlap),     # (W, Lo)
            "keypoint_mask": torch.from_numpy(keypoint_mask),  # (W,)
            "meta": {"pid": pid, "start_cycle": start, "fs": self.fs},
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


class MimicBPCycleDataModule(L.LightningDataModule):
    """
    Lightning DataModule for cycle-based PPG→ECG.

    支持两种模式：
    1. preprocessed=True (默认): 读取已预处理的 cycles/ 目录
    2. preprocessed=False: 在 prepare_data() 中从原始 npy 在线预处理
    """

    def __init__(
        self,
        data_dir: str,
        train_split: str = "train_subjects.txt",
        val_split: str = "val_subjects.txt",
        test_split: str = "test_subjects.txt",
        # cycle params
        window_size: int = 32,
        cycle_len: int = 256,
        overlap_len: int = 32,
        stride: int = 16,
        fs: int = 125,
        # preprocessing
        preprocessed: bool = True,
        preprocess_output_dir: Optional[str] = None,
        ecg_bandpass: Tuple[float, float] = (0.5, 40.0),
        ppg_bandpass: Tuple[float, float] = (0.5, 8.0),
        powerline_freq: float = 50.0,
        files: Tuple[str, str] = ("ppg", "ecg"),
        # loader params
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
        seed_for_loader: int = 42,
        # optional
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["transform"])

        self.data_dir = str(data_dir)
        self.train_split = str(train_split)
        self.val_split = str(val_split)
        self.test_split = str(test_split)

        self.window_size = window_size
        self.cycle_len = cycle_len
        self.overlap_len = overlap_len
        self.stride = stride
        self.fs = fs

        self.preprocessed = preprocessed
        self.preprocess_output_dir = preprocess_output_dir or self.data_dir
        self.ecg_bandpass = ecg_bandpass
        self.ppg_bandpass = ppg_bandpass
        self.powerline_freq = powerline_freq
        self.files = files

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.seed_for_loader = seed_for_loader
        self.transform = transform

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def prepare_data(self):
        """Run preprocessing if needed (called once, single process)."""
        if not self.preprocessed:
            cycles_dir = Path(self.preprocess_output_dir) / "cycles"
            if not cycles_dir.exists() or len(list(cycles_dir.glob("*_ppg_cycles.npy"))) == 0:
                print("[MimicBPCycleDataModule] Running preprocessing...")
                preprocess_and_save(
                    data_dir=self.data_dir,
                    output_dir=self.preprocess_output_dir,
                    fs=self.fs,
                    cycle_len=self.cycle_len,
                    ecg_bandpass=self.ecg_bandpass,
                    ppg_bandpass=self.ppg_bandpass,
                    powerline_freq=self.powerline_freq,
                    files=self.files,
                )
            else:
                print("[MimicBPCycleDataModule] Preprocessed data already exists, skipping.")
        else:
            # Verify cycles directory exists
            cycles_dir = Path(self.data_dir) / "cycles"
            if not cycles_dir.exists():
                raise FileNotFoundError(
                    f"cycles/ directory not found at {cycles_dir}. "
                    f"Run preprocess_and_save() first or set preprocessed=False."
                )

    def _make_dataset(self, split_file: str, stride: Optional[int] = None) -> MimicBPCycleDataset:
        base_dir = self.preprocess_output_dir if not self.preprocessed else self.data_dir
        return MimicBPCycleDataset(
            data_dir=base_dir,
            split_file=split_file,
            window_size=self.window_size,
            cycle_len=self.cycle_len,
            overlap_len=self.overlap_len,
            stride=stride or self.stride,
            fs=self.fs,
            transform=self.transform,
        )

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            self.train_ds = self._make_dataset(self.train_split, stride=self.stride)
            # Validation uses non-overlapping windows
            self.val_ds = self._make_dataset(self.val_split, stride=self.window_size)
        if stage in (None, "test"):
            self.test_ds = self._make_dataset(self.test_split, stride=self.window_size)

    def _make_generator(self):
        g = torch.Generator()
        g.manual_seed(self.seed_for_loader)
        return g

    def train_dataloader(self):
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            worker_init_fn=_seed_worker,
            generator=self._make_generator(),
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            worker_init_fn=_seed_worker,
            generator=self._make_generator(),
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        assert self.test_ds is not None
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            worker_init_fn=_seed_worker,
            generator=self._make_generator(),
            persistent_workers=self.num_workers > 0,
        )
