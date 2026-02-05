from __future__ import annotations

from pathlib import Path
import lightning as L
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 128, num_workers: int = 4):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        # 只下载，不做 split（Lightning 约定）
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            self.mnist_train = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            # MNIST 官方没有 val；这里用 test 作为 val 简化示例（你可自行 random_split）
            self.mnist_val = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

        if stage in (None, "test"):
            self.mnist_test = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
