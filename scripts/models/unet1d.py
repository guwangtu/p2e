import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Any
# -------------------------
# Building blocks
# -------------------------
class DoubleConv1D(nn.Module):
    """(Conv1d -> GN -> SiLU) * 2"""
    def __init__(self, in_ch, out_ch, groups=8, k=3):
        super().__init__()
        pad = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, bias=False),
            nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=pad, bias=False),
            nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class Down1D(nn.Module):
    """Downsample by 2 using strided conv, then DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv = DoubleConv1D(out_ch, out_ch)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x

class Up1D(nn.Module):
    """Upsample by 2 using transposed conv, concat skip, then DoubleConv"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv = DoubleConv1D(out_ch + skip_ch, out_ch)

    @staticmethod
    def _match_length(x, target_len):
        """Pad or crop x to match target_len on the last dimension."""
        cur = x.shape[-1]
        if cur == target_len:
            return x
        if cur < target_len:
            pad = target_len - cur
            left = pad // 2
            right = pad - left
            return F.pad(x, (left, right))
        # cur > target_len
        start = (cur - target_len) // 2
        return x[..., start:start + target_len]

    def forward(self, x, skip):
        x = self.up(x)
        x = self._match_length(x, skip.shape[-1])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

# -------------------------
# 1D U-Net (length = 3750 safe)
# -------------------------
class UNet1D(nn.Module):
    """
    Input:  [B, in_ch, 3750]
    Output: [B, out_ch, 3750]
    """
    def __init__(self, in_ch=1, out_ch=1, base_ch=32, depth=4):
        super().__init__()
        assert depth >= 2, "depth 至少为 2"
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.base_ch = base_ch
        self.depth = depth

        self.stem = DoubleConv1D(in_ch, base_ch)

        # Encoder
        enc_channels = [base_ch]
        downs = []
        ch = base_ch
        for _ in range(depth):
            nxt = ch * 2
            downs.append(Down1D(ch, nxt))
            ch = nxt
            enc_channels.append(ch)
        self.downs = nn.ModuleList(downs)

        # Bottleneck
        self.mid = DoubleConv1D(ch, ch)

        # Decoder
        ups = []
        for i in range(depth):
            skip_ch = enc_channels[-(i+2)]  # reverse skips
            ups.append(Up1D(in_ch=ch, skip_ch=skip_ch, out_ch=skip_ch))
            ch = skip_ch
        self.ups = nn.ModuleList(ups)

        self.head = nn.Conv1d(base_ch, out_ch, kernel_size=1)

    @staticmethod
    def _pad_to_multiple_of(x, multiple):
        """Pad last dim so that length is divisible by multiple."""
        L = x.shape[-1]
        target = ((L + multiple - 1) // multiple) * multiple
        if target == L:
            return x, 0, 0
        pad = target - L
        left = pad // 2
        right = pad - left
        return F.pad(x, (left, right)), left, right

    def forward(self, x):
        """
        x: [B, C, L] where L can be 3750
        """
        # To make repeated /2 downsampling safe, pad length to multiple of 2^depth
        multiple = 2 ** self.depth
        x_pad, pad_left, pad_right = self._pad_to_multiple_of(x, multiple)

        # Encoder with skips
        s0 = self.stem(x_pad)
        skips = [s0]
        h = s0
        for down in self.downs:
            h = down(h)
            skips.append(h)

        # Bottleneck
        h = self.mid(h)

        # Decoder (skip the last skip which is the deepest feature)
        for i, up in enumerate(self.ups):
            skip = skips[-(i+2)]
            h = up(h, skip)

        y = self.head(h)

        # Crop back to original length
        if pad_left != 0 or pad_right != 0:
            y = y[..., pad_left:y.shape[-1]-pad_right]
        return y
 
class LitUNet1D(L.LightningModule):
    """
    Lightning wrapper for UNet1D.

    Assumes batch is either:
      - dict with keys: "x" (input), "y" (target)
      - or tuple: (x, y)

    Default task: regression (PPG -> ECG), uses MSE loss.
    """

    def __init__(
        self,
        # ---- net params ----
        in_ch: int = 1,
        out_ch: int = 1,
        base_ch: int = 32,
        depth: int = 4,

        # ---- optim params ----
        lr: float = 1e-3,
        weight_decay: float = 1e-4,

        # ---- loss/metrics params (optional) ----
        loss: str = "mse",          # "mse" | "l1" | "huber"
        huber_delta: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = UNet1D(in_ch=in_ch, out_ch=out_ch, base_ch=base_ch, depth=depth)

        if loss == "mse":
            self.criterion = nn.MSELoss()
        elif loss == "l1":
            self.criterion = nn.L1Loss()
        elif loss == "huber":
            self.criterion = nn.SmoothL1Loss(beta=huber_delta)
        else:
            raise ValueError(f"Unknown loss: {loss}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Support dict batch (recommended) or tuple batch.
        """
        if isinstance(batch, dict):
            x = batch["x"]
            y = batch["y"]
            return x, y
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _shared_step(self, batch: Any) -> torch.Tensor:
        x, y = self._unpack_batch(batch)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch)
        batch_size = batch["x"].shape[0] if isinstance(batch, dict) else batch[0].shape[0]
        self.log("train/loss", loss, on_step=True,batch_size=batch_size, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self._shared_step(batch)
        batch_size = batch["x"].shape[0] if isinstance(batch, dict) else batch[0].shape[0]
        self.log("val/loss", loss, on_step=False, on_epoch=True,batch_size=batch_size, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss = self._shared_step(batch)
        batch_size = batch["x"].shape[0] if isinstance(batch, dict) else batch[0].shape[0]
        self.log("test/loss", loss, on_step=False, on_epoch=True,batch_size=batch_size, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
        )
        return opt