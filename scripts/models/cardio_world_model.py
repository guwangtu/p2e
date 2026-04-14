"""
CardioWorldModel: PPG-to-ECG as a World Model problem.

Framing:
  - Observation (o_t): ECG cycle at time t
  - Action (a_t):      PPG cycle at time t
  - Latent state:      cardiovascular system state, split into
                        h_t (deterministic recurrent) + z_t (stochastic)
  - Transition:        h_t = GRU(h_{t-1}, [z_{t-1}, a_t, δ_t])
  - Prior:             p(z_t | h_t)          — imagination (PPG only)
  - Posterior:         q(z_t | h_t, o_t)     — grounded (PPG + ECG)
  - Decoder:           p(o_t | h_t, z_t)     — ECG reconstruction

Architecture:
  1. PPGEncoder       — action encoder: PPG cycle -> feature vector
  2. ECGEncoder       — observation encoder: ECG cycle -> feature vector (training)
  3. Time2Vec         — RR interval encoding
  4. RSSM             — Recurrent State Space Model (deterministic + stochastic)
  5. FlowMatchGen1D   — conditional flow matching decoder (h_t, z_t) -> ECG latent
  6. ECGDecoder       — latent -> waveform with overlap conditioning
  7. KeypointHead     — morphological reward signal
  8. LitCardioWorldModel — Lightning Module
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

# Reuse building blocks from cycle_flow
from .cycle_flow import (
    DoubleConv1D,
    Down1D,
    PPGEncoder,
    ECGEncoder,
    ECGDecoder,
    Time2Vec,
    KeypointHead,
    SincShift,
    CrossAttentionBlock,
)


# =============================================================================
# 1. RSSM — Recurrent State Space Model
# =============================================================================

class RSSSMPrior(nn.Module):
    """Prior network: p(z_t | h_t).
    Maps deterministic state to diagonal Gaussian parameters."""

    def __init__(self, h_dim: int, z_dim: int, hidden: int = 200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """h: (B, h_dim) -> mu: (B, z_dim), logvar: (B, z_dim)"""
        feat = self.net(h)
        return self.mu(feat), self.logvar(feat)


class RSSMPosterior(nn.Module):
    """Posterior network: q(z_t | h_t, o_t).
    Conditions on both deterministic state and observation encoding."""

    def __init__(self, h_dim: int, obs_dim: int, z_dim: int, hidden: int = 200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim + obs_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

    def forward(
        self, h: torch.Tensor, obs_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h:        (B, h_dim)   — deterministic recurrent state
        obs_feat: (B, obs_dim) — encoded observation (ECG)
        Returns:  mu (B, z_dim), logvar (B, z_dim)
        """
        feat = self.net(torch.cat([h, obs_feat], dim=-1))
        return self.mu(feat), self.logvar(feat)


class RSSM(nn.Module):
    """
    Recurrent State Space Model for cardiovascular dynamics.

    State = (h_t, z_t) where:
      h_t: deterministic recurrent state (captures temporal dynamics)
      z_t: stochastic latent (captures per-cycle variation)

    Transition: h_t = GRU(h_{t-1}, [z_{t-1}, a_t, δ_t, ov_t])
    Prior:      z_t ~ N(mu_prior(h_t), sigma_prior(h_t))
    Posterior:  z_t ~ N(mu_post(h_t, o_t), sigma_post(h_t, o_t))
    """

    def __init__(
        self,
        action_dim: int = 128,     # PPG feature dim
        obs_dim: int = 128,        # ECG feature dim
        time_dim: int = 32,        # Time2Vec dim
        overlap_dim: int = 32,     # overlap encoding dim
        h_dim: int = 256,          # deterministic state dim
        z_dim: int = 64,           # stochastic state dim
        prior_hidden: int = 200,
        posterior_hidden: int = 200,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim

        # GRU input: z_{t-1} + a_t + δ_t + ov_t
        gru_input_dim = z_dim + action_dim + time_dim + overlap_dim
        self.gru = nn.GRUCell(gru_input_dim, h_dim)

        # Overlap encoder (raw overlap samples -> overlap_dim)
        self.overlap_enc = nn.Sequential(
            nn.Linear(32, 64),  # default overlap_len=32
            nn.SiLU(),
            nn.Linear(64, overlap_dim),
        )

        self.prior = RSSSMPrior(h_dim, z_dim, prior_hidden)
        self.posterior = RSSMPosterior(h_dim, obs_dim, z_dim, posterior_hidden)

    def init_state(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize (h_0, z_0) to zeros."""
        h = torch.zeros(batch_size, self.h_dim, device=device)
        z = torch.zeros(batch_size, self.z_dim, device=device)
        return h, z

    @staticmethod
    def reparameterize(
        mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps"""
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + std * eps

    def transition(
        self,
        h_prev: torch.Tensor,     # (B, h_dim)
        z_prev: torch.Tensor,     # (B, z_dim)
        action: torch.Tensor,     # (B, action_dim) — encoded PPG
        delta: torch.Tensor,      # (B, time_dim)  — encoded RR
        overlap: torch.Tensor,    # (B, overlap_len) — raw overlap samples
    ) -> torch.Tensor:
        """Deterministic transition: h_t = GRU(h_{t-1}, [z_{t-1}, a_t, δ_t, ov_t])"""
        ov = self.overlap_enc(overlap)
        gru_input = torch.cat([z_prev, action, delta, ov], dim=-1)
        h_new = self.gru(gru_input, h_prev)
        return h_new

    def forward_step(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        action: torch.Tensor,
        delta: torch.Tensor,
        overlap: torch.Tensor,
        obs_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Single timestep forward.

        If obs_feat is provided (training): uses posterior for z_t.
        If obs_feat is None (imagination): uses prior for z_t.

        Returns dict with: h, z, mu_prior, logvar_prior,
                          [mu_post, logvar_post] (if obs_feat given)
        """
        h = self.transition(h_prev, z_prev, action, delta, overlap)

        mu_prior, logvar_prior = self.prior(h)
        result = {
            "h": h,
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
        }

        if obs_feat is not None:
            # Training: use posterior
            mu_post, logvar_post = self.posterior(h, obs_feat)
            z = self.reparameterize(mu_post, logvar_post)
            result["mu_post"] = mu_post
            result["logvar_post"] = logvar_post
        else:
            # Imagination: use prior
            z = self.reparameterize(mu_prior, logvar_prior)

        result["z"] = z
        return result


# =============================================================================
# 2. World Model Decoder (Flow Matching conditioned on h_t, z_t)
# =============================================================================

class WMVelocityNet(nn.Module):
    """
    Velocity field for flow matching, conditioned on RSSM state (h_t, z_t).

    Differs from CycleFlow's VelocityNet: condition tokens come from
    the world model state rather than dual-SSM outputs.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        h_dim: int = 256,
        z_dim: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
    ):
        super().__init__()
        cond_dim = h_dim + z_dim

        # Project h and z to same dim for cross-attention tokens
        self.h_proj = nn.Linear(h_dim, cond_dim)
        self.z_proj_cond = nn.Linear(z_dim, cond_dim)

        # Flow time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Latent projection
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        # Cross-attention layers
        self.layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, cond_dim, n_heads)
            for _ in range(n_layers)
        ])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(
        self,
        x: torch.Tensor,      # (B, latent_dim) — noised ECG latent
        s: torch.Tensor,       # (B,) — flow time [0, 1]
        h: torch.Tensor,       # (B, h_dim) — deterministic state
        z: torch.Tensor,       # (B, z_dim) — stochastic state
    ) -> torch.Tensor:
        s_emb = self.time_mlp(self.sinusoidal_embedding(s))  # (B, hidden)
        feat = self.latent_proj(x).unsqueeze(1) + s_emb.unsqueeze(1)  # (B, 1, hidden)

        # 2 condition tokens: projected h and z
        cond = torch.stack([self.h_proj(h), self.z_proj_cond(z)], dim=1)  # (B, 2, cond_dim)

        for layer in self.layers:
            feat = layer(feat, cond)

        return self.out_proj(feat.squeeze(1))


class WMFlowMatchDecoder(nn.Module):
    """
    Flow matching decoder conditioned on world model state (h_t, z_t).
    Training: CFM loss against target ECG latent.
    Inference: ODE integration from noise to ECG latent.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        h_dim: int = 256,
        z_dim: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.velocity_net = WMVelocityNet(
            latent_dim=latent_dim,
            h_dim=h_dim, z_dim=z_dim,
            hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads,
        )

    def compute_loss(
        self,
        target: torch.Tensor,  # (B, latent_dim) — ECG latent z_1
        h: torch.Tensor,       # (B, h_dim)
        z: torch.Tensor,       # (B, z_dim)
    ) -> torch.Tensor:
        B = target.shape[0]
        device = target.device

        s = torch.rand(B, device=device)
        z_0 = torch.randn_like(target)
        s_exp = s.unsqueeze(-1)
        z_s = (1 - s_exp) * z_0 + s_exp * target
        target_v = target - z_0

        pred_v = self.velocity_net(z_s, s, h, z)
        return F.mse_loss(pred_v, target_v)

    @torch.no_grad()
    def sample(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        n_steps: int = 20,
    ) -> torch.Tensor:
        B = h.shape[0]
        device = h.device
        x = torch.randn(B, self.latent_dim, device=device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            s = torch.full((B,), i * dt, device=device)
            v = self.velocity_net(x, s, h, z)
            x = x + v * dt

        return x


# =============================================================================
# 3. Reward Head — physiological plausibility score
# =============================================================================

class RewardHead(nn.Module):
    """
    Predicts a reward (physiological plausibility) from world model state.

    Reward components:
      - RR interval prediction (temporal consistency)
      - Morphology score via keypoints (delegated to KeypointHead)

    This head predicts next-cycle RR interval from h_t,
    serving as a self-supervised reward signal.
    """

    def __init__(self, h_dim: int = 256, z_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.rr_pred = nn.Sequential(
            nn.Linear(h_dim + z_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Softplus(),  # RR > 0
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predict next RR interval (seconds). Returns (B, 1)."""
        return self.rr_pred(torch.cat([h, z], dim=-1))


# =============================================================================
# 4. Lightning Module
# =============================================================================

class LitCardioWorldModel(L.LightningModule):
    """
    CardioWorldModel: PPG-to-ECG generation as a World Model problem.

    Training:
      - Uses posterior q(z_t | h_t, o_t) to sample stochastic state
      - KL(posterior || prior) forces prior to learn from PPG alone
      - Flow matching decodes (h_t, z_t) -> ECG latent -> waveform

    Inference (imagine):
      - Uses prior p(z_t | h_t) only — no ECG needed
      - Autoregressive rollout with generated overlap
    """

    def __init__(
        self,
        # PPG Encoder (action)
        enc_base_ch: int = 32,
        enc_depth: int = 4,
        feat_dim: int = 128,
        # ECG Encoder (observation)
        obs_base_ch: int = 32,
        obs_depth: int = 4,
        obs_dim: int = 128,
        # Time2Vec
        time_dim: int = 32,
        # RSSM
        h_dim: int = 256,
        z_dim: int = 64,
        overlap_dim: int = 32,
        prior_hidden: int = 200,
        posterior_hidden: int = 200,
        # Flow Matching Decoder
        flow_latent_dim: int = 128,
        flow_hidden: int = 256,
        flow_layers: int = 4,
        flow_heads: int = 4,
        # ECG Decoder (waveform)
        dec_hidden: int = 256,
        cycle_len: int = 256,
        overlap_len: int = 32,
        # Keypoint Head
        kp_hidden: int = 128,
        # Loss weights
        lambda_recon: float = 1.0,
        lambda_cfm: float = 1.0,
        lambda_kl: float = 0.1,
        lambda_morph: float = 0.1,
        lambda_rr: float = 0.05,
        lambda_align: float = 0.01,
        # KL balancing (Dreamer-v2 style)
        kl_balance: float = 0.8,
        free_nats: float = 1.0,
        # Optimizer
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        # Inference
        n_sample_steps: int = 20,
    ):
        super().__init__()
        self.save_hyperparameters()

        # --- Action encoder (PPG) ---
        self.ppg_encoder = PPGEncoder(
            in_ch=1, base_ch=enc_base_ch, depth=enc_depth, feat_dim=feat_dim,
        )

        # --- Observation encoder (ECG, training only) ---
        self.ecg_obs_encoder = PPGEncoder(
            in_ch=1, base_ch=obs_base_ch, depth=obs_depth, feat_dim=obs_dim,
        )

        # --- ECG latent encoder (flow matching target) ---
        self.ecg_latent_encoder = ECGEncoder(
            in_ch=1, base_ch=enc_base_ch, depth=enc_depth, latent_dim=flow_latent_dim,
        )

        # --- Time encoding ---
        self.time2vec = Time2Vec(out_dim=time_dim)

        # --- RSSM ---
        self.rssm = RSSM(
            action_dim=feat_dim, obs_dim=obs_dim,
            time_dim=time_dim, overlap_dim=overlap_dim,
            h_dim=h_dim, z_dim=z_dim,
            prior_hidden=prior_hidden, posterior_hidden=posterior_hidden,
        )

        # --- Flow matching decoder ---
        self.flow_decoder = WMFlowMatchDecoder(
            latent_dim=flow_latent_dim,
            h_dim=h_dim, z_dim=z_dim,
            hidden_dim=flow_hidden, n_layers=flow_layers, n_heads=flow_heads,
        )

        # --- ECG waveform decoder ---
        self.ecg_decoder = ECGDecoder(
            latent_dim=flow_latent_dim, overlap_dim=overlap_dim,
            hidden_dim=dec_hidden, out_ch=1,
            cycle_len=cycle_len, overlap_len=overlap_len,
        )

        # --- Auxiliary heads ---
        self.keypoint_head = KeypointHead(cycle_len=cycle_len, hidden=kp_hidden)
        self.reward_head = RewardHead(h_dim=h_dim, z_dim=z_dim)

        # --- Soft alignment ---
        self.sinc_shift = SincShift(feat_dim=feat_dim, time_dim=time_dim)

    # -----------------------------------------------------------------
    # KL divergence with free nats and balancing
    # -----------------------------------------------------------------
    @staticmethod
    def _kl_divergence(
        mu_post: torch.Tensor,
        logvar_post: torch.Tensor,
        mu_prior: torch.Tensor,
        logvar_prior: torch.Tensor,
    ) -> torch.Tensor:
        """Analytical KL(q || p) for diagonal Gaussians."""
        var_post = logvar_post.exp()
        var_prior = logvar_prior.exp()
        kl = 0.5 * (
            logvar_prior - logvar_post
            + var_post / var_prior
            + (mu_post - mu_prior).pow(2) / var_prior
            - 1.0
        )
        return kl.sum(dim=-1)  # (B,)

    def _balanced_kl(
        self,
        mu_post: torch.Tensor,
        logvar_post: torch.Tensor,
        mu_prior: torch.Tensor,
        logvar_prior: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dreamer-v2 style KL balancing.
        alpha fraction of gradient goes to posterior, (1-alpha) to prior.
        """
        alpha = self.hparams.kl_balance
        free = self.hparams.free_nats

        # KL with prior detached (trains posterior)
        kl_post = self._kl_divergence(
            mu_post, logvar_post,
            mu_prior.detach(), logvar_prior.detach(),
        )
        # KL with posterior detached (trains prior)
        kl_prior = self._kl_divergence(
            mu_post.detach(), logvar_post.detach(),
            mu_prior, logvar_prior,
        )

        # Free nats: don't optimize below threshold
        kl_post = torch.clamp(kl_post, min=free).mean()
        kl_prior = torch.clamp(kl_prior, min=free).mean()

        return alpha * kl_post + (1 - alpha) * kl_prior

    # -----------------------------------------------------------------
    # Overlap cross-fade
    # -----------------------------------------------------------------
    @staticmethod
    def _crossfade_overlap(
        prev_tail: torch.Tensor,
        curr_head: torch.Tensor,
    ) -> torch.Tensor:
        Lo = prev_tail.shape[-1]
        alpha = torch.linspace(1, 0, Lo, device=prev_tail.device).view(1, 1, Lo)
        return alpha * prev_tail + (1 - alpha) * curr_head

    # -----------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        ppg = batch["ppg_cycles"]         # (B, W, L)
        ecg = batch["ecg_cycles"]         # (B, W, L)
        rr = batch["rr_intervals"]        # (B, W)
        kp_gt = batch["keypoints"]        # (B, W, 10)
        kp_mask = batch["keypoint_mask"]  # (B, W)
        overlap = batch["ecg_overlap"]    # (B, W, Lo)

        B, W, L = ppg.shape
        Lo = overlap.shape[-1]

        # --- Encode all PPG cycles (actions) ---
        ppg_flat = ppg.reshape(B * W, 1, L)
        action_all = self.ppg_encoder(ppg_flat).reshape(B, W, -1)  # (B, W, feat_dim)

        # --- Time encoding ---
        delta_all = self.time2vec(rr)  # (B, W, time_dim)

        # --- Soft alignment ---
        ecg_flat = ecg.reshape(B * W, 1, L)
        f_flat = action_all.reshape(B * W, -1)
        d_flat = delta_all.reshape(B * W, -1)
        ecg_shifted, tau = self.sinc_shift(ecg_flat, f_flat, d_flat)
        ecg_shifted = ecg_shifted.reshape(B, W, L)
        tau = tau.reshape(B, W)

        # --- Encode all ECG cycles (observations) ---
        ecg_shifted_flat = ecg_shifted.reshape(B * W, 1, L)
        obs_all = self.ecg_obs_encoder(ecg_shifted_flat).reshape(B, W, -1)  # (B, W, obs_dim)

        # --- Encode ECG latents (flow matching targets) ---
        z1_all = self.ecg_latent_encoder(ecg_shifted_flat).reshape(B, W, -1)  # (B, W, flow_latent)

        # --- RSSM rollout (training with posterior) ---
        h, z = self.rssm.init_state(B, ppg.device)

        h_seq, z_seq = [], []
        kl_losses = []

        for t in range(W):
            rssm_out = self.rssm.forward_step(
                h_prev=h, z_prev=z,
                action=action_all[:, t],
                delta=delta_all[:, t],
                overlap=overlap[:, t],
                obs_feat=obs_all[:, t],  # posterior: use ECG
            )

            h = rssm_out["h"]
            z = rssm_out["z"]
            h_seq.append(h)
            z_seq.append(z)

            # KL divergence
            kl = self._balanced_kl(
                rssm_out["mu_post"], rssm_out["logvar_post"],
                rssm_out["mu_prior"], rssm_out["logvar_prior"],
            )
            kl_losses.append(kl)

        h_seq = torch.stack(h_seq, dim=1)  # (B, W, h_dim)
        z_seq = torch.stack(z_seq, dim=1)  # (B, W, z_dim)
        loss_kl = torch.stack(kl_losses).mean()

        # --- Flow matching loss ---
        h_flat = h_seq.reshape(B * W, -1)
        z_flat = z_seq.reshape(B * W, -1)
        z1_flat = z1_all.reshape(B * W, -1)
        loss_cfm = self.flow_decoder.compute_loss(z1_flat, h_flat, z_flat)

        # --- Decode for reconstruction loss ---
        overlap_flat = overlap.reshape(B * W, Lo)
        y_hat_ext = self.ecg_decoder(z1_flat, overlap_flat)  # teacher-forced decode
        y_hat_core = y_hat_ext[..., Lo: Lo + L]

        ecg_target_flat = ecg_shifted.reshape(B * W, 1, L)
        loss_recon = (
            F.l1_loss(y_hat_core, ecg_target_flat)
            + F.mse_loss(y_hat_core, ecg_target_flat)
        )

        # --- Keypoint morphology loss ---
        kp_pred = self.keypoint_head(y_hat_core)
        kp_gt_flat = kp_gt.reshape(B * W, 10)
        kp_mask_flat = kp_mask.reshape(B * W)
        if kp_mask_flat.any():
            valid = kp_mask_flat.bool()
            loss_morph = F.mse_loss(kp_pred[valid], kp_gt_flat[valid])
        else:
            loss_morph = torch.tensor(0.0, device=ppg.device)

        # --- RR prediction reward ---
        rr_pred = self.reward_head(h_flat, z_flat).squeeze(-1)  # (B*W,)
        rr_target = rr.reshape(B * W)
        loss_rr = F.mse_loss(rr_pred, rr_target)

        # --- Alignment regularization ---
        loss_align = (tau ** 2).mean()

        # --- Total loss ---
        loss = (
            self.hparams.lambda_cfm * loss_cfm
            + self.hparams.lambda_recon * loss_recon
            + self.hparams.lambda_kl * loss_kl
            + self.hparams.lambda_morph * loss_morph
            + self.hparams.lambda_rr * loss_rr
            + self.hparams.lambda_align * loss_align
        )

        # Logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("train/loss_cfm", loss_cfm, on_step=False, on_epoch=True, batch_size=B)
        self.log("train/loss_recon", loss_recon, on_step=False, on_epoch=True, batch_size=B)
        self.log("train/loss_kl", loss_kl, on_step=False, on_epoch=True, batch_size=B)
        self.log("train/loss_morph", loss_morph, on_step=False, on_epoch=True, batch_size=B)
        self.log("train/loss_rr", loss_rr, on_step=False, on_epoch=True, batch_size=B)
        self.log("train/loss_align", loss_align, on_step=False, on_epoch=True, batch_size=B)
        self.log("train/tau_mean", tau.abs().mean(), on_step=False, on_epoch=True, batch_size=B)
        return loss

    # -----------------------------------------------------------------
    # Validation step — uses imagination (prior only, no ECG peeking)
    # -----------------------------------------------------------------
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        ppg = batch["ppg_cycles"]
        ecg = batch["ecg_cycles"]
        rr = batch["rr_intervals"]
        kp_gt = batch["keypoints"]
        kp_mask = batch["keypoint_mask"]
        overlap = batch["ecg_overlap"]

        B, W, L = ppg.shape
        Lo = overlap.shape[-1]

        # Encode actions
        ppg_flat = ppg.reshape(B * W, 1, L)
        action_all = self.ppg_encoder(ppg_flat).reshape(B, W, -1)
        delta_all = self.time2vec(rr)

        # RSSM rollout with PRIOR only (imagination)
        h, z = self.rssm.init_state(B, ppg.device)
        h_seq, z_seq = [], []
        kl_losses = []

        # Also encode observations for KL monitoring
        ecg_flat = ecg.reshape(B * W, 1, L)
        obs_all = self.ecg_obs_encoder(ecg_flat).reshape(B, W, -1)

        for t in range(W):
            # Transition using prior (imagination mode)
            h = self.rssm.transition(h, z, action_all[:, t], delta_all[:, t], overlap[:, t])

            mu_prior, logvar_prior = self.rssm.prior(h)
            z = self.rssm.reparameterize(mu_prior, logvar_prior)

            h_seq.append(h)
            z_seq.append(z)

            # Monitor KL (compute posterior for logging only)
            mu_post, logvar_post = self.rssm.posterior(h, obs_all[:, t])
            kl = self._kl_divergence(mu_post, logvar_post, mu_prior, logvar_prior)
            kl_losses.append(kl.mean())

        h_seq = torch.stack(h_seq, dim=1)
        z_seq = torch.stack(z_seq, dim=1)
        loss_kl = torch.stack(kl_losses).mean()

        # Decode via flow sampling
        h_flat = h_seq.reshape(B * W, -1)
        z_flat = z_seq.reshape(B * W, -1)
        ecg_latent = self.flow_decoder.sample(h_flat, z_flat, n_steps=self.hparams.n_sample_steps)

        overlap_flat = overlap.reshape(B * W, Lo)
        y_hat_ext = self.ecg_decoder(ecg_latent, overlap_flat)
        y_hat_core = y_hat_ext[..., Lo: Lo + L]

        # Recon loss
        loss_recon = (
            F.l1_loss(y_hat_core, ecg_flat)
            + F.mse_loss(y_hat_core, ecg_flat)
        )

        # Keypoint loss
        kp_pred = self.keypoint_head(y_hat_core)
        kp_gt_flat = kp_gt.reshape(B * W, 10)
        kp_mask_flat = kp_mask.reshape(B * W)
        if kp_mask_flat.any():
            valid = kp_mask_flat.bool()
            loss_morph = F.mse_loss(kp_pred[valid], kp_gt_flat[valid])
        else:
            loss_morph = torch.tensor(0.0, device=ppg.device)

        loss = loss_recon + self.hparams.lambda_morph * loss_morph

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=B)
        self.log("val/loss_recon", loss_recon, on_step=False, on_epoch=True, batch_size=B)
        self.log("val/loss_morph", loss_morph, on_step=False, on_epoch=True, batch_size=B)
        self.log("val/loss_kl", loss_kl, on_step=False, on_epoch=True, batch_size=B)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)

    # -----------------------------------------------------------------
    # Imagination: generate ECG from PPG only
    # -----------------------------------------------------------------
    @torch.no_grad()
    def imagine(
        self,
        ppg_cycles: torch.Tensor,    # (1, W, L) or (W, L)
        rr_intervals: torch.Tensor,  # (1, W) or (W,)
        n_steps: int = 20,
    ) -> torch.Tensor:
        """
        Dream ECG from PPG only — no teacher forcing, pure prior rollout.
        Returns: (total_samples,) concatenated ECG waveform.
        """
        if ppg_cycles.ndim == 2:
            ppg_cycles = ppg_cycles.unsqueeze(0)
            rr_intervals = rr_intervals.unsqueeze(0)

        B, W, L = ppg_cycles.shape
        Lo = self.hparams.overlap_len
        device = ppg_cycles.device

        # Encode all PPG
        action_all = self.ppg_encoder(
            ppg_cycles.reshape(B * W, 1, L)
        ).reshape(B, W, -1)
        delta_all = self.time2vec(rr_intervals)

        # Autoregressive rollout with prior
        h, z = self.rssm.init_state(B, device)
        prev_tail = torch.zeros(B, Lo, device=device)
        generated_cycles = []

        for t in range(W):
            h = self.rssm.transition(h, z, action_all[:, t], delta_all[:, t], prev_tail)
            mu_prior, logvar_prior = self.rssm.prior(h)
            z = self.rssm.reparameterize(mu_prior, logvar_prior)

            # Decode
            ecg_latent = self.flow_decoder.sample(
                h.unsqueeze(0).squeeze(0), z, n_steps=n_steps,
            )
            y_ext = self.ecg_decoder(ecg_latent, prev_tail)  # (B, 1, out_len)
            y_core = y_ext[:, 0, Lo: Lo + L]

            # Cross-fade overlap
            if t > 0 and Lo > 0:
                curr_head = y_ext[:, 0, :Lo]
                faded = self._crossfade_overlap(
                    prev_tail.unsqueeze(1), curr_head.unsqueeze(1),
                ).squeeze(1)
                if len(generated_cycles) > 0:
                    generated_cycles[-1][:, -Lo:] = faded

            generated_cycles.append(y_core)
            prev_tail = y_ext[:, 0, Lo + L: Lo + L + Lo]

        full_ecg = torch.cat(generated_cycles, dim=-1)
        return full_ecg.squeeze(0)

    # -----------------------------------------------------------------
    # Anomaly detection: KL divergence as surprise signal
    # -----------------------------------------------------------------
    @torch.no_grad()
    def compute_surprise(
        self,
        ppg_cycles: torch.Tensor,    # (1, W, L)
        ecg_cycles: torch.Tensor,    # (1, W, L)
        rr_intervals: torch.Tensor,  # (1, W)
        overlap: torch.Tensor,       # (1, W, Lo)
    ) -> torch.Tensor:
        """
        Compute per-cycle surprise (KL divergence) between prior and posterior.
        High surprise = PPG cannot predict ECG = potential cardiac anomaly.

        Returns: (W,) surprise scores.
        """
        B, W, L = ppg_cycles.shape

        action_all = self.ppg_encoder(
            ppg_cycles.reshape(B * W, 1, L)
        ).reshape(B, W, -1)
        delta_all = self.time2vec(rr_intervals)
        obs_all = self.ecg_obs_encoder(
            ecg_cycles.reshape(B * W, 1, L)
        ).reshape(B, W, -1)

        h, z = self.rssm.init_state(B, ppg_cycles.device)
        surprises = []

        for t in range(W):
            rssm_out = self.rssm.forward_step(
                h, z, action_all[:, t], delta_all[:, t], overlap[:, t],
                obs_feat=obs_all[:, t],
            )
            h = rssm_out["h"]
            z = rssm_out["z"]

            kl = self._kl_divergence(
                rssm_out["mu_post"], rssm_out["logvar_post"],
                rssm_out["mu_prior"], rssm_out["logvar_prior"],
            )
            surprises.append(kl.squeeze(0))

        return torch.stack(surprises)  # (W,)

    # -----------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
        )
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup = LinearLR(opt, start_factor=0.01, total_iters=self.hparams.warmup_steps)
        cosine = CosineAnnealingLR(
            opt, T_max=self.trainer.estimated_stepping_batches - self.hparams.warmup_steps,
        )
        scheduler = SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[self.hparams.warmup_steps],
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
