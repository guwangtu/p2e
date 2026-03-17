from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class ConvBlock1D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 7,
        stride: int = 2,
        padding: Optional[int] = None,
        norm: bool = True,
        act: bool = True,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        layers: list[nn.Module] = [
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        ]
        if norm:
            layers.append(nn.BatchNorm1d(out_ch))
        if act:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeconvBlock1D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        norm: bool = True,
        act: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose1d(
                in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding
            )
        ]
        if norm:
            layers.append(nn.BatchNorm1d(out_ch))
        if act:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianHead(nn.Module):
    """Produce mu/logvar for a Gaussian posterior."""

    def __init__(self, in_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.mu = nn.Linear(in_dim, latent_dim)
        self.logvar = nn.Linear(in_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mu(x), self.logvar(x)


class SharedPrivateEncoder1D(nn.Module):
    """
    Simple 1D encoder that outputs shared/private Gaussian posterior params.
    Input:  (B, C, T)
    """

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        hidden_dim: int,
        shared_dim: int,
        private_dim: int,
        num_downsamples: int = 4,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.num_downsamples = int(num_downsamples)

        chs = [32, 64, 128, hidden_dim]
        blocks: list[nn.Module] = []
        prev = in_channels
        for c in chs:
            blocks.append(ConvBlock1D(prev, c, stride=2))
            prev = c
        self.features = nn.Sequential(*blocks)

        downsample_factor = 2**self.num_downsamples
        if self.seq_len % downsample_factor != 0:
            raise ValueError(
                f"seq_len must be divisible by 2^{self.num_downsamples} "
                f"(got seq_len={self.seq_len})"
            )

        feat_len = self.seq_len // downsample_factor
        flat_dim = hidden_dim * feat_len
        self.hidden_dim = hidden_dim
        self.feat_len = feat_len

        self.flatten = nn.Flatten()
        self.shared_head = GaussianHead(flat_dim, shared_dim)
        self.private_head = GaussianHead(flat_dim, private_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.features(x)
        h_flat = self.flatten(h)
        mu_s, logvar_s = self.shared_head(h_flat)
        mu_p, logvar_p = self.private_head(h_flat)
        return {"mu_s": mu_s, "logvar_s": logvar_s, "mu_p": mu_p, "logvar_p": logvar_p}


class Decoder1D(nn.Module):
    """
    Decode latent vector back to waveform.
    Input latent: (B, latent_dim)
    Output: (B, out_channels, T)
    """

    def __init__(
        self,
        out_channels: int,
        seq_len: int,
        hidden_dim: int,
        latent_dim: int,
        num_upsamples: int = 4,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.hidden_dim = int(hidden_dim)
        self.num_upsamples = int(num_upsamples)

        upsample_factor = 2**self.num_upsamples
        if self.seq_len % upsample_factor != 0:
            raise ValueError(
                f"seq_len must be divisible by 2^{self.num_upsamples} "
                f"(got seq_len={self.seq_len})"
            )

        feat_len = self.seq_len // upsample_factor
        self.feat_len = feat_len
        self.fc = nn.Linear(latent_dim, self.hidden_dim * feat_len)

        self.deconv = nn.Sequential(
            DeconvBlock1D(self.hidden_dim, 128),
            DeconvBlock1D(128, 64),
            DeconvBlock1D(64, 32),
            DeconvBlock1D(32, 16),
            nn.Conv1d(16, out_channels, kernel_size=7, padding=3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), self.hidden_dim, self.feat_len)
        x_hat = self.deconv(h)
        if x_hat.size(-1) > self.seq_len:
            x_hat = x_hat[..., : self.seq_len]
        elif x_hat.size(-1) < self.seq_len:
            x_hat = F.pad(x_hat, (0, self.seq_len - x_hat.size(-1)))
        return x_hat


class LitSLVAE(L.LightningModule):
    """
    Shared-Latent Dual-VAE for paired signals (PPG -> ECG).

    Assumes batch is either:
      - dict with keys: "x" (PPG), "y" (ECG)
      - or tuple/list: (x, y)
    """

    def __init__(
        self,
        # ---- data shape ----
        in_ch: int = 1,  # PPG channels
        out_ch: int = 1,  # ECG channels
        seq_len: int = 3750,
        # ---- latent/encoder/decoder ----
        hidden_dim: int = 256,
        shared_dim: int = 64,
        private_dim: int = 32,
        # ---- optim ----
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        # ---- loss weights ----
        beta_shared: float = 1e-3,
        beta_private: float = 1e-3,
        lambda_self_recon: float = 1.0,
        lambda_cross_recon: float = 1.0,
        lambda_align: float = 1.0,
        # ---- loss type ----
        recon_loss: str = "l1",  # "l1" | "mse" | "smooth_l1"
        # ---- behavior ----
        zero_private_for_cross: bool = True,
        log_latent_stats: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.seq_len = int(seq_len)
        self.hidden_dim = int(hidden_dim)
        self.shared_dim = int(shared_dim)
        self.private_dim = int(private_dim)

        self._multiple = 2**4
        self.seq_len_eff = ((self.seq_len + self._multiple - 1) // self._multiple) * self._multiple

        latent_total_dim = self.shared_dim + self.private_dim

        self.ppg_encoder = SharedPrivateEncoder1D(
            in_channels=self.in_ch,
            seq_len=self.seq_len_eff,
            hidden_dim=self.hidden_dim,
            shared_dim=self.shared_dim,
            private_dim=self.private_dim,
        )
        self.ecg_encoder = SharedPrivateEncoder1D(
            in_channels=self.out_ch,
            seq_len=self.seq_len_eff,
            hidden_dim=self.hidden_dim,
            shared_dim=self.shared_dim,
            private_dim=self.private_dim,
        )

        self.ppg_decoder = Decoder1D(
            out_channels=self.in_ch,
            seq_len=self.seq_len_eff,
            hidden_dim=self.hidden_dim,
            latent_dim=latent_total_dim,
        )
        self.ecg_decoder = Decoder1D(
            out_channels=self.out_ch,
            seq_len=self.seq_len_eff,
            hidden_dim=self.hidden_dim,
            latent_dim=latent_total_dim,
        )

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
        L = x.shape[-1]
        target = ((L + multiple - 1) // multiple) * multiple
        if target == L:
            return x, 0, 0
        pad = target - L
        left = pad // 2
        right = pad - left
        return F.pad(x, (left, right)), left, right

    @staticmethod
    def _crop(x: torch.Tensor, left: int, right: int) -> torch.Tensor:
        if left == 0 and right == 0:
            return x
        end = x.shape[-1] - right
        return x[..., left:end]

    # -------------------------
    # batch parsing
    # -------------------------
    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            x = batch["x"]
            y = batch["y"]
            return x.float(), y.float()
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0].float(), batch[1].float()
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    # -------------------------
    # VAE utils
    # -------------------------
    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # clamp logvar to avoid exp overflow / NaNs
        logvar = logvar.clamp(-10.0, 10.0)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.sum(dim=1).mean()

    def _recon_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        t = self.hparams.recon_loss
        if t == "l1":
            return F.l1_loss(pred, target)
        if t == "mse":
            return F.mse_loss(pred, target)
        if t == "smooth_l1":
            return F.smooth_l1_loss(pred, target)
        raise ValueError(f"Unsupported recon_loss: {t}")

    @staticmethod
    def _align_loss(mu_s_ppg: torch.Tensor, mu_s_ecg: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(mu_s_ppg, mu_s_ecg)

    def _make_zero_private(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.private_dim, device=device)

    # -------------------------
    # encode / decode helpers
    # -------------------------
    def encode_ppg(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.ppg_encoder(x)
        out["z_s"] = self.reparameterize(out["mu_s"], out["logvar_s"])
        out["z_p"] = self.reparameterize(out["mu_p"], out["logvar_p"])
        return out

    def encode_ecg(self, y: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.ecg_encoder(y)
        out["z_s"] = self.reparameterize(out["mu_s"], out["logvar_s"])
        out["z_p"] = self.reparameterize(out["mu_p"], out["logvar_p"])
        return out

    def decode_ppg(self, z_s: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:
        return self.ppg_decoder(torch.cat([z_s, z_p], dim=1))

    def decode_ecg(self, z_s: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:
        return self.ecg_decoder(torch.cat([z_s, z_p], dim=1))

    # -------------------------
    # forward: PPG -> ECG
    # -------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pad, left, right = self._pad_to_multiple(x, self._multiple)
        enc = self.encode_ppg(x_pad)
        z_s = enc["z_s"]
        if bool(self.hparams.zero_private_for_cross):
            z_p_ecg = self._make_zero_private(x_pad.size(0), x_pad.device)
        else:
            z_p_ecg = enc["z_p"]
        y_hat = self.decode_ecg(z_s, z_p_ecg)
        return self._crop(y_hat, left, right)

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        x, y = self._unpack_batch(batch)
        x_pad, x_left, x_right = self._pad_to_multiple(x, self._multiple)
        y_pad, y_left, y_right = self._pad_to_multiple(y, self._multiple)

        # Encode both
        ppg_post = self.encode_ppg(x_pad)
        ecg_post = self.encode_ecg(y_pad)

        z_s_ppg, z_p_ppg = ppg_post["z_s"], ppg_post["z_p"]
        z_s_ecg, z_p_ecg = ecg_post["z_s"], ecg_post["z_p"]

        # Self recon
        x_self_hat = self.decode_ppg(z_s_ppg, z_p_ppg)
        y_self_hat = self.decode_ecg(z_s_ecg, z_p_ecg)
        x_self_hat = self._crop(x_self_hat, x_left, x_right)
        y_self_hat = self._crop(y_self_hat, y_left, y_right)
        loss_x_self = self._recon_loss(x_self_hat, x)
        loss_y_self = self._recon_loss(y_self_hat, y)
        loss_self = loss_x_self + loss_y_self

        # Cross recon
        if bool(self.hparams.zero_private_for_cross):
            zero_p = self._make_zero_private(x_pad.size(0), x_pad.device)
            y_cross_hat = self.decode_ecg(z_s_ppg, zero_p)  # PPG -> ECG
            x_cross_hat = self.decode_ppg(z_s_ecg, zero_p)  # ECG -> PPG
        else:
            y_cross_hat = self.decode_ecg(z_s_ppg, z_p_ppg)
            x_cross_hat = self.decode_ppg(z_s_ecg, z_p_ecg)

        y_cross_hat = self._crop(y_cross_hat, y_left, y_right)
        x_cross_hat = self._crop(x_cross_hat, x_left, x_right)
        loss_p2e = self._recon_loss(y_cross_hat, y)
        loss_e2p = self._recon_loss(x_cross_hat, x)
        loss_cross = loss_p2e + loss_e2p

        # KL
        kl_shared = self.kl_divergence(ppg_post["mu_s"], ppg_post["logvar_s"]) + self.kl_divergence(
            ecg_post["mu_s"], ecg_post["logvar_s"]
        )
        kl_private = self.kl_divergence(ppg_post["mu_p"], ppg_post["logvar_p"]) + self.kl_divergence(
            ecg_post["mu_p"], ecg_post["logvar_p"]
        )

        # Align
        loss_align = self._align_loss(ppg_post["mu_s"], ecg_post["mu_s"])

        loss = (
            float(self.hparams.lambda_self_recon) * loss_self
            + float(self.hparams.lambda_cross_recon) * loss_cross
            + float(self.hparams.beta_shared) * kl_shared
            + float(self.hparams.beta_private) * kl_private
            + float(self.hparams.lambda_align) * loss_align
        )

        batch_size = x.shape[0]
        self.log(f"{stage}/loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}/loss_self", loss_self, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/loss_cross", loss_cross, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/loss_p2e", loss_p2e, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/loss_e2p", loss_e2p, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/loss_align", loss_align, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/kl_shared", kl_shared, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/kl_private", kl_private, on_step=False, on_epoch=True, batch_size=batch_size)

        if bool(self.hparams.log_latent_stats):
            self.log(
                f"{stage}/mu_s_ppg_abs",
                ppg_post["mu_s"].abs().mean(),
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}/mu_s_ecg_abs",
                ecg_post["mu_s"].abs().mean(),
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}/mu_p_ppg_abs",
                ppg_post["mu_p"].abs().mean(),
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}/mu_p_ecg_abs",
                ecg_post["mu_p"].abs().mean(),
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: Any, batch_idx: int):
        self._shared_step(batch, stage="val")

    def test_step(self, batch: Any, batch_idx: int):
        self._shared_step(batch, stage="test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
        )

