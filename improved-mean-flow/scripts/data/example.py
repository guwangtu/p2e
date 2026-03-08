from __future__ import annotations

from pathlib import Path
import lightning as L
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class ExampleDataModule(L.LightningDataModule):
    def __init__(self, data_dir="data", batch_size=128, num_workers=4):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_tf = None
        self.eval_tf = None

    def prepare_data(self):
        # Optional: Download/extract/preprocess to disk
        pass

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_ds = None
            self.val_ds = None

        if stage in (None, "test"):
            self.test_ds = None

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )




