from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import MulticlassAccuracy
from torchvision.models import resnet18


class LitMNISTResNet18(L.LightningModule):
    def __init__(self, lr: float = 1e-3, weight_decay: float = 1e-4, label_smoothing: float = 0.0):
        super().__init__()
        self.save_hyperparameters()

        # ResNet18 默认 conv1 输入 3 通道，这里改成 1 通道（MNIST）
        self.net = resnet18(weights=None, num_classes=10)
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.train_acc = MulticlassAccuracy(num_classes=10)
        self.val_acc = MulticlassAccuracy(num_classes=10)
        self.test_acc = MulticlassAccuracy(num_classes=10)

    def forward(self, x):
        return self.net(x)

    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=float(self.hparams.label_smoothing))
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        acc = self.train_acc(preds, y)
        self.log_dict(
            {"train/loss": loss, "train/acc": acc},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        acc = self.val_acc(preds, y)
        self.log_dict(
            {"val/loss": loss, "val/acc": acc},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        acc = self.test_acc(preds, y)
        self.log_dict(
            {"test/loss": loss, "test/acc": acc},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
        )
        # 你也可以统一在 config 里加 scheduler
        return opt
