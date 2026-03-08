from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Callable

import lightning as L
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import os
import ast


# 你可以把 MimicBPDataset 放在同目录 dataset 文件里，然后这里 import 


def seed_worker(worker_id: int):
    # 可选：如果你已经用 L.seed_everything(..., workers=True)，可以不写这个
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)


class MimicBPDataModule(L.LightningDataModule):
    """
    Lightning DataModule for MimicBP.

    目录约定：
      data_dir/ppg/{pid}_ppg.npy
      data_dir/ecg/{pid}_ecg.npy
    split 文件：支持 python list 字符串 或 每行一个 pid。
    """
    def __init__(
        self,
        data_dir: str,
        train_split: str = "train_subjects.txt",
        val_split: str = "val_subjects.txt",
        test_split: str = "test_subjects.txt",
        files: tuple[str, str] = ("ppg", "ecg"),
        num_segments_per_patient: int = 30,
        fs: int = 125,
        return_dict: bool = True,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        # loader params
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
        persistent_workers: bool = False,
        seed_for_loader: int = 42,
    ):
        super().__init__()
        self.data_dir = str(data_dir)
        self.train_split = str(train_split)
        self.val_split = str(val_split)
        self.test_split = str(test_split)

        self.files = files
        self.num_segments_per_patient = int(num_segments_per_patient)
        self.fs = int(fs)
        self.return_dict = bool(return_dict)
        self.transform = transform

        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.drop_last = bool(drop_last)
        self.seed_for_loader = int(seed_for_loader)

        # placeholders
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def prepare_data(self):
        """
        可选：做数据存在性检查。
        不建议在这里构建 dataset（Lightning 约定：prepare_data 只做一次性 I/O）。
        """
        root = Path(self.data_dir)
        if not root.exists():
            raise FileNotFoundError(f"data_dir not found: {root}")
        # 轻量检查：至少有 ppg/ecg 目录
        for sub in self.files:
            p = root / sub
            if not p.exists():
                raise FileNotFoundError(f"missing subfolder: {p}")

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            self.train_ds = MimicBPDataset(
                data_dir=self.data_dir,
                split_file=self.train_split,
                transform=self.transform,
                files=self.files,
                num_segments_per_patient=self.num_segments_per_patient,
                fs=self.fs,
                return_dict=self.return_dict,
            )
            self.val_ds = MimicBPDataset(
                data_dir=self.data_dir,
                split_file=self.val_split,
                transform=self.transform,
                files=self.files,
                num_segments_per_patient=self.num_segments_per_patient,
                fs=self.fs,
                return_dict=self.return_dict,
            )

        if stage in (None, "test"):
            self.test_ds = MimicBPDataset(
                data_dir=self.data_dir,
                split_file=self.test_split,
                transform=self.transform,
                files=self.files,
                num_segments_per_patient=self.num_segments_per_patient,
                fs=self.fs,
                return_dict=self.return_dict,
            )

    def _make_generator(self):
        g = torch.Generator()
        g.manual_seed(self.seed_for_loader)
        return g

    def train_dataloader(self):
        assert self.train_ds is not None, "Call setup('fit') before train_dataloader"
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            worker_init_fn=seed_worker,
            generator=self._make_generator(),
            persistent_workers= self.num_workers > 0 ,
        )

    def val_dataloader(self):
        assert self.val_ds is not None, "Call setup('fit') before val_dataloader"
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=self._make_generator(),
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        assert self.test_ds is not None, "Call setup('test') before test_dataloader"
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=self._make_generator(),
            persistent_workers=self.num_workers > 0,
        )




def _load_subject_ids(split_path: str) -> List[str]:
    """
    兼容两种 split 文件格式：
    1) 你当前的：一个 Python list 字符串，例如 ['123', '456']
    2) 更常见的：每行一个 subject/patient id
    """
    with open(split_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    # 形式 1：像 python list 形如 ['123', '456', '789']
    if content[0] in ["[", "("]:
        ids = ast.literal_eval(content)
        if isinstance(ids, (list, tuple)):
            return [str(x) for x in ids]
        raise ValueError(f"split file parsed but not list/tuple: {type(ids)}")

    # 形式 2：每行一个id
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    return [str(x) for x in lines]


class MimicBPDataset(Dataset):
    """
    读取 mimic-bp 数据集的 PPG/ECG 分段数据。

    约定（按你原始实现）：
    - data_dir/ppg/{pid}_ppg.npy  -> shape: (num_segments, T) 或 (num_segments, ...)
    - data_dir/ecg/{pid}_ecg.npy

    返回默认是 dict（更利于你后续扩写 trainer）：
      {
        "x": Tensor[1,T],  # PPG
        "y": Tensor[1,T],  # ECG
        "meta": {...}
      }
    如果你更想保持旧版 tuple 返回，可设置 return_dict=False。
    """

    def __init__(
        self,
        data_dir: str,
        split_file: str,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        files: Tuple[str, str] = ("ppg", "ecg"),
        num_segments_per_patient: int = 30,
        fs: int = 125,
        return_dict: bool = True,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.files = list(files)
        self.num_segments_per_patient = int(num_segments_per_patient)
        self.fs = int(fs)
        self.return_dict = bool(return_dict)

        # split_file 既可传相对路径（相对 data_dir），也可传绝对路径
        split_path = split_file
        if not os.path.isabs(split_path):
            split_path = os.path.join(data_dir, split_file)

        self.patient_ids = _load_subject_ids(split_path)

        # 构建样本索引 (pid, segment_idx)
        self.samples = [(pid, idx) for pid in self.patient_ids for idx in range(self.num_segments_per_patient)]

        print(
            f"[MimicBPDataset] Loaded {len(self.samples)} segments "
            f"from {len(self.patient_ids)} patients. split={split_path}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_signal_segment(self, pid: str, wav: str, idx: int) -> np.ndarray:
        wav_dir = os.path.join(self.data_dir, wav)
        npy_path = os.path.join(wav_dir, f"{pid}_{wav}.npy")
        arr = np.load(npy_path)  # arr shape: (num_segments, T) ...

        if idx < 0 or idx >= arr.shape[0]:
            raise IndexError(f"segment idx out of range: idx={idx}, arr.shape={arr.shape}, file={npy_path}")

        seg = arr[idx]
        seg = np.asarray(seg, dtype=np.float32)
        return seg

    def __getitem__(self, i: int):
        pid, idx = self.samples[i]

        # 读取 PPG 和 ECG
        x_np = self._load_signal_segment(pid, self.files[0], idx)  # PPG
        y_np = self._load_signal_segment(pid, self.files[1], idx)  # ECG

        # -> torch [1, T]
        x = torch.from_numpy(x_np).float().unsqueeze(0)
        y = torch.from_numpy(y_np).float().unsqueeze(0)

        if self.transform:
            x = self.transform(x)

        if self.return_dict:
            return {
                "x": x,
                "y": y,
                "meta": {"pid": pid, "segment_idx": idx, "fs": self.fs},
            }
        else:
            return x, y


def build_mimicbp_datasets(data_cfg: Dict[str, Any]):
    """
    构建 train/val/test Dataset（仅 Dataset，不负责 DataLoader）
    """
    data_dir = data_cfg["data_dir"]

    train_split = data_cfg.get("train_split", "train_subjects.txt")
    val_split = data_cfg.get("val_split", "val_subjects.txt")
    test_split = data_cfg.get("test_split", "test_subjects.txt")

    num_segments = int(data_cfg.get("num_segments_per_patient", 30))
    fs = int(data_cfg.get("fs", 125))
    return_dict = bool(data_cfg.get("return_dict", True))

    # transform 先留接口：你后续可以从 scripts/transforms.py 里 build_transform(cfg)
    transform = None

    train_ds = MimicBPDataset(
        data_dir=data_dir,
        split_file=train_split,
        transform=transform,
        num_segments_per_patient=num_segments,
        fs=fs,
        return_dict=return_dict,
    )
    val_ds = MimicBPDataset(
        data_dir=data_dir,
        split_file=val_split,
        transform=transform,
        num_segments_per_patient=num_segments,
        fs=fs,
        return_dict=return_dict,
    )
    test_ds = MimicBPDataset(
        data_dir=data_dir,
        split_file=test_split,
        transform=transform,
        num_segments_per_patient=num_segments,
        fs=fs,
        return_dict=return_dict,
    )
    return train_ds, val_ds, test_ds
