"""
CycleIndependent: Ablation baseline — per-cycle PPG→ECG without temporal context.

Removes all inter-cycle dependencies:
  - No DualSSM / RSSM (no hidden state across cycles)
  - No RR interval encoding (no Time2Vec)
  - No overlap conditioning (no previous cycle tail)

Keeps the same PPGEncoder, FlowMatchGen1D, ECGDecoder architecture
so the comparison isolates the effect of sequential modeling.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from .cycle_flow import (
    PPGEncoder,
    ECGEncoder,
    ECGDecoder,
    KeypointHead,
    FlowMatchGen1D,
    CrossAttentionBlock,
)


# =============================================================================
# Independent Flow Generator (no cross-attention on SSM context)
# =============================================================================

class IndependentFlowGen(nn.Module):
    """
    Flow Matching generator conditioned only on PPG features (no SSM context).
    Simplified from FlowMatchGen1D: condition = PPG feature only.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        cond_dim: int = 128,       # PPG feature dim only
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Project (z_s, s, cond) into hidden
        self.input_proj = nn.Linear(latent_dim + 1 + cond_dim, hidden_dim)

        # MLP + cross-attention layers
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
        self.layers = nn.ModuleList(layers)

        # Cross-attention on condition
        self.cross_attn = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, cond_dim, n_heads)
            for _ in range(n_layers)
        ])

        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def velocity(
        self,
        z_s: torch.Tensor,    # (B, latent_dim) — noisy latent
        s: torch.Tensor,       # (B,) — flow time
        cond: torch.Tensor,    # (B, cond_dim) — PPG feature
    ) -> torch.Tensor:
        s_emb = s.unsqueeze(-1)  # (B, 1)
        h = self.input_proj(torch.cat([z_s, s_emb, cond], dim=-1))

        for mlp, ca in zip(self.layers, self.cross_attn):
            h = h + mlp(h)
            # Cross-attention: cond as single-token KV
            h = ca(h.unsqueeze(1), cond.unsqueeze(1)).squeeze(1)

        return self.out_proj(h)

    def compute_loss(
        self,
        z_1: torch.Tensor,    # (B, latent) — target latent (ECG encoded)
        cond: torch.Tensor,    # (B, cond_dim) — PPG feature
    ) -> torch.Tensor:
        B = z_1.shape[0]
        # Random flow time
        s = torch.rand(B, device=z_1.device)
        # Random noise
        z_0 = torch.randn_like(z_1)
        # Interpolation: z_s = (1-s)*z_0 + s*z_1
        s_exp = s.unsqueeze(-1)
        z_s = (1 - s_exp) * z_0 + s_exp * z_1
        # Target velocity: z_1 - z_0
        v_target = z_1 - z_0
        # Predicted velocity
        v_pred = self.velocity(z_s, s, cond)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,    # (B, cond_dim)
        n_steps: int = 20,
    ) -> torch.Tensor:
        B = cond.shape[0]
        z = torch.randn(B, self.latent_dim, device=cond.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            s = torch.full((B,), i * dt, device=cond.device)
            v = self.velocity(z, s, cond)
            z = z + v * dt
        return z


# =============================================================================
# Lightning Module
# =============================================================================

class LitCycleIndependent(L.LightningModule):
    """
    Ablation model: each PPG cycle is independently mapped to ECG cycle.
    No DualSSM, no RR encoding, no overlap conditioning.

    Uses same data format (MimicBPCycleDataModule) for fair comparison.
    """

    def __init__(
        self,
        # PPG Encoder
        enc_base_ch: int = 32,
        enc_depth: int = 4,
        feat_dim: int = 128,
        # Flow Matching
        flow_hidden: int = 256,
        flow_layers: int = 4,
        flow_heads: int = 4,
        latent_dim: int = 128,
        # ECG Decoder (simplified: no overlap)
        dec_hidden: int = 256,
        cycle_len: int = 256,
        overlap_len: int = 32,
        # Keypoint Head
        kp_hidden: int = 128,
        # Loss weights
        lambda_recon: float = 1.0,
        lambda_cfm: float = 1.0,
        lambda_morph: float = 0.1,
        # Optimizer
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        # Inference
        n_sample_steps: int = 20,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.ppg_encoder = PPGEncoder(
            in_ch=1, base_ch=enc_base_ch, depth=enc_depth, feat_dim=feat_dim,
        )

        self.ecg_encoder = ECGEncoder(latent_dim=latent_dim)

        # Independent flow generator: conditioned on PPG feature only
        self.flow_gen = IndependentFlowGen(
            latent_dim=latent_dim,
            cond_dim=feat_dim,
            hidden_dim=flow_hidden,
            n_layers=flow_layers,
            n_heads=flow_heads,
        )

        # ECG Decoder: no overlap conditioning (zero overlap)
        self.ecg_decoder = ECGDecoder(
            latent_dim=latent_dim, overlap_dim=overlap_len,
            hidden_dim=dec_hidden, out_ch=1,
            cycle_len=cycle_len, overlap_len=overlap_len,
        )

        self.keypoint_head = KeypointHead(cycle_len=cycle_len, hidden=kp_hidden)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        ppg = batch["ppg_cycles"]       # (B, W, L)
        ecg = batch["ecg_cycles"]       # (B, W, L)
        kp_gt = batch["keypoints"]      # (B, W, 10)
        kp_mask = batch["keypoint_mask"]  # (B, W)

        B, W, L = ppg.shape
        Lo = self.hparams.overlap_len

        # Flatten: treat each cycle independently
        ppg_flat = ppg.reshape(B * W, 1, L)     # (BW, 1, L)
        ecg_flat = ecg.reshape(B * W, 1, L)     # (BW, 1, L)

        # Encode PPG
        f = self.ppg_encoder(ppg_flat)           # (BW, feat_dim)

        # Encode ECG target
        z_1 = self.ecg_encoder(ecg_flat)         # (BW, latent_dim)

        # Flow matching loss: conditioned on PPG feature only
        loss_cfm = self.flow_gen.compute_loss(z_1, f)

        # Decode for reconstruction loss (zero overlap)
        overlap_zero = torch.zeros(B * W, Lo, device=ppg.device)
        y_hat_ext = self.ecg_decoder(z_1, overlap_zero)  # (BW, 1, out_len)
        y_hat_core = y_hat_ext[..., Lo: Lo + L]          # (BW, 1, L)

        loss_recon = F.l1_loss(y_hat_core, ecg_flat) + F.mse_loss(y_hat_core, ecg_flat)

        # Keypoint loss
        kp_pred = self.keypoint_head(y_hat_core)  # (BW, 10)
        kp_gt_flat = kp_gt.reshape(B * W, 10)
        kp_mask_flat = kp_mask.reshape(B * W)
        if kp_mask_flat.any():
            valid = kp_mask_flat.bool()
            loss_morph = F.mse_loss(kp_pred[valid], kp_gt_flat[valid])
        else:
            loss_morph = torch.tensor(0.0, device=ppg.device)

        loss = (
            self.hparams.lambda_cfm * loss_cfm
            + self.hparams.lambda_recon * loss_recon
            + self.hparams.lambda_morph * loss_morph
        )

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("train/loss_cfm", loss_cfm, on_step=False, on_epoch=True, batch_size=B)
        self.log("train/loss_recon", loss_recon, on_step=False, on_epoch=True, batch_size=B)
        self.log("train/loss_morph", loss_morph, on_step=False, on_epoch=True, batch_size=B)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        ppg = batch["ppg_cycles"]
        ecg = batch["ecg_cycles"]

        B, W, L = ppg.shape
        Lo = self.hparams.overlap_len

        ppg_flat = ppg.reshape(B * W, 1, L)
        ecg_flat = ecg.reshape(B * W, 1, L)

        # Encode + sample (full inference path)
        f = self.ppg_encoder(ppg_flat)
        z_sampled = self.flow_gen.sample(f, n_steps=self.hparams.n_sample_steps)

        overlap_zero = torch.zeros(B * W, Lo, device=ppg.device)
        y_hat_ext = self.ecg_decoder(z_sampled, overlap_zero)
        y_hat_core = y_hat_ext[..., Lo: Lo + L]

        loss_recon = F.l1_loss(y_hat_core, ecg_flat) + F.mse_loss(y_hat_core, ecg_flat)

        self.log("val/loss", loss_recon, on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=B)
        self.log("val/loss_recon", loss_recon, on_step=False, on_epoch=True, batch_size=B)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        ppg = batch["ppg_cycles"]
        ecg = batch["ecg_cycles"]

        B, W, L = ppg.shape
        Lo = self.hparams.overlap_len

        ppg_flat = ppg.reshape(B * W, 1, L)
        ecg_flat = ecg.reshape(B * W, 1, L)

        f = self.ppg_encoder(ppg_flat)
        z_sampled = self.flow_gen.sample(f, n_steps=self.hparams.n_sample_steps)

        overlap_zero = torch.zeros(B * W, Lo, device=ppg.device)
        y_hat_ext = self.ecg_decoder(z_sampled, overlap_zero)
        y_hat_core = y_hat_ext[..., Lo: Lo + L]

        loss_recon = F.l1_loss(y_hat_core, ecg_flat) + F.mse_loss(y_hat_core, ecg_flat)

        self.log("test/loss", loss_recon, on_step=False, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("val/loss", loss_recon, on_step=False, on_epoch=True, batch_size=B)

    @torch.no_grad()
    def generate(
        self,
        ppg_cycles: torch.Tensor,    # (1, W, L) or (W, L)
        rr_intervals: torch.Tensor = None,  # ignored, kept for API compatibility
        n_steps: int = 20,
    ) -> torch.Tensor:
        """
        Generate ECG from PPG cycles — each cycle independently.
        Returns: (W * L,) concatenated ECG waveform.
        """
        if ppg_cycles.ndim == 2:
            ppg_cycles = ppg_cycles.unsqueeze(0)

        B, W, L = ppg_cycles.shape
        Lo = self.hparams.overlap_len
        device = ppg_cycles.device

        ppg_flat = ppg_cycles.reshape(B * W, 1, L)
        f = self.ppg_encoder(ppg_flat)
        z = self.flow_gen.sample(f, n_steps=n_steps)

        overlap_zero = torch.zeros(B * W, Lo, device=device)
        y_hat_ext = self.ecg_decoder(z, overlap_zero)
        y_hat_core = y_hat_ext[..., Lo: Lo + L]  # (BW, 1, L)

        ecg_out = y_hat_core.reshape(B, W, L)
        full_ecg = ecg_out.reshape(B, -1)  # (B, W*L)
        return full_ecg.squeeze(0)  # (W*L,)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
        )
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup = LinearLR(opt, start_factor=0.01, total_iters=self.hparams.warmup_steps)
        cosine = CosineAnnealingLR(opt, T_max=self.trainer.estimated_stepping_batches - self.hparams.warmup_steps)
        scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[self.hparams.warmup_steps])
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
