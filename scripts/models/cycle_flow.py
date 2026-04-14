"""
CycleFlow: Cycle-level PPG-to-ECG generation via Dual-SSM + Conditional Flow Matching.

Architecture:
  1. PPGEncoder       — 1D-CNN encoder (reuses Down1D blocks from unet1d.py)
  2. Time2Vec         — Learnable periodic time encoding for RR intervals
  3. DualSSM          — Dual-channel SSM: global trend + local modulation
  4. FlowMatchGen1D   — Conditional Flow Matching generator with cross-attention
  5. ECGDecoder       — Transposed-conv decoder with overlap conditioning
  6. KeypointHead     — Auxiliary ECG morphology constraint head
  7. SincShift        — Differentiable sub-sample time shift for soft alignment
  8. LitCycleFlow     — Lightning Module assembling all components
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


# =============================================================================
# 1. PPG Encoder
# =============================================================================

class DoubleConv1D(nn.Module):
    """(Conv1d -> GroupNorm -> SiLU) x 2"""
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8, k: int = 3):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down1D(nn.Module):
    """Downsample by 2 using strided conv, then DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv = DoubleConv1D(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.down(x))


class PPGEncoder(nn.Module):
    """
    Encode a single PPG cycle (B, 1, L) -> feature vector (B, feat_dim).
    Uses stacked Down1D blocks + global average pooling.
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 32, depth: int = 4, feat_dim: int = 128):
        super().__init__()
        self.stem = DoubleConv1D(in_ch, base_ch)

        blocks = []
        ch = base_ch
        for _ in range(depth):
            nxt = min(ch * 2, 512)
            blocks.append(Down1D(ch, nxt))
            ch = nxt
        self.downs = nn.ModuleList(blocks)
        self.proj = nn.Linear(ch, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, L) -> (B, feat_dim)"""
        h = self.stem(x)
        for down in self.downs:
            h = down(h)
        h = h.mean(dim=-1)  # (B, ch) — deterministic global avg pool
        return self.proj(h)


# =============================================================================
# 2. Time2Vec
# =============================================================================

class Time2Vec(nn.Module):
    """
    Learnable periodic time encoding.
    Maps scalar Δt -> (k+1)-dim vector: [linear, sin_1, ..., sin_k]
    """
    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.out_dim = out_dim
        self.linear_w = nn.Parameter(torch.randn(1))
        self.linear_b = nn.Parameter(torch.randn(1))
        self.periodic_w = nn.Parameter(torch.randn(out_dim - 1))
        self.periodic_b = nn.Parameter(torch.randn(out_dim - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (...,) scalar -> (..., out_dim)"""
        t = t.unsqueeze(-1)  # (..., 1)
        linear_part = self.linear_w * t + self.linear_b  # (..., 1)
        periodic_part = torch.sin(self.periodic_w * t + self.periodic_b)  # (..., out_dim-1)
        return torch.cat([linear_part, periodic_part], dim=-1)


# =============================================================================
# 3. Dual-SSM (GRU-based, with optional Mamba drop-in)
# =============================================================================

class SSMCell(nn.Module):
    """
    Single SSM channel implemented as GRU cell.
    Can be replaced with Mamba/S4 block if mamba_ssm is available.

    Input: (f_t, delta_t, overlap_t) concatenated as input features.
    """
    def __init__(self, input_dim: int, hidden_dim: int, use_mamba: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_mamba = use_mamba

        if use_mamba:
            try:
                from mamba_ssm import Mamba
                # Mamba operates on sequences; we wrap it for step-wise use
                self.mamba = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
                self.input_proj = nn.Linear(input_dim, hidden_dim)
                self._has_mamba = True
            except ImportError:
                print("[SSMCell] mamba_ssm not found, falling back to GRU.")
                self.use_mamba = False
                self._has_mamba = False

        if not self.use_mamba:
            self.cell = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        x: (B, input_dim)  — concatenated input features for one timestep
        h: (B, hidden_dim)  — previous hidden state
        Returns: new hidden state (B, hidden_dim)
        """
        if self.use_mamba and self._has_mamba:
            # Mamba expects (B, L, D) — use L=1 for single step
            inp = self.input_proj(x).unsqueeze(1)  # (B, 1, hidden_dim)
            # Add residual from hidden state
            out = self.mamba(inp + h.unsqueeze(1))  # (B, 1, hidden_dim)
            return out.squeeze(1)
        else:
            return self.cell(x, h)


class DualSSM(nn.Module):
    """
    Dual-channel State Space Model:
      - Global channel: large hidden state, captures slow trends (HR drift, ST shift)
      - Local channel: smaller hidden state, captures per-cycle modulations

    Inputs per step: f_t (PPG features), delta_t (Time2Vec encoded RR),
                     overlap_t (previous cycle tail encoding)
    Outputs: g_t (global condition), m_t (local condition)
    """
    def __init__(
        self,
        feat_dim: int = 128,
        time_dim: int = 32,
        overlap_dim: int = 32,
        global_hidden: int = 256,
        local_hidden: int = 128,
        global_out: int = 128,
        local_out: int = 64,
        use_mamba: bool = False,
    ):
        super().__init__()
        input_dim = feat_dim + time_dim + overlap_dim

        self.global_cell = SSMCell(input_dim, global_hidden, use_mamba=use_mamba)
        self.local_cell = SSMCell(input_dim, local_hidden, use_mamba=use_mamba)

        self.global_proj = nn.Linear(global_hidden, global_out)
        self.local_proj = nn.Linear(local_hidden, local_out)

        self.global_hidden = global_hidden
        self.local_hidden = local_hidden

        # Overlap encoder: compress L_o raw samples into overlap_dim
        self.overlap_enc = nn.Sequential(
            nn.Linear(32, 64),  # default overlap_len=32
            nn.SiLU(),
            nn.Linear(64, overlap_dim),
        )

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h_g = torch.zeros(batch_size, self.global_hidden, device=device)
        h_l = torch.zeros(batch_size, self.local_hidden, device=device)
        return h_g, h_l

    def forward_step(
        self,
        f_t: torch.Tensor,       # (B, feat_dim)
        delta_t: torch.Tensor,    # (B, time_dim)
        overlap_t: torch.Tensor,  # (B, overlap_len)
        h_g: torch.Tensor,        # (B, global_hidden)
        h_l: torch.Tensor,        # (B, local_hidden)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step forward. Returns g_t, m_t, h_g_new, h_l_new."""
        ov = self.overlap_enc(overlap_t)  # (B, overlap_dim)
        inp = torch.cat([f_t, delta_t, ov], dim=-1)  # (B, input_dim)

        h_g = self.global_cell(inp, h_g)
        h_l = self.local_cell(inp, h_l)

        g_t = self.global_proj(h_g)  # (B, global_out)
        m_t = self.local_proj(h_l)   # (B, local_out)
        return g_t, m_t, h_g, h_l

    def forward(
        self,
        f_seq: torch.Tensor,       # (B, W, feat_dim)
        delta_seq: torch.Tensor,    # (B, W, time_dim)
        overlap_seq: torch.Tensor,  # (B, W, overlap_len)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full sequence forward.
        Returns: g_seq (B, W, global_out), m_seq (B, W, local_out)
        """
        B, W, _ = f_seq.shape
        h_g, h_l = self.init_hidden(B, f_seq.device)

        g_list, m_list = [], []
        for t in range(W):
            g_t, m_t, h_g, h_l = self.forward_step(
                f_seq[:, t], delta_seq[:, t], overlap_seq[:, t], h_g, h_l
            )
            g_list.append(g_t)
            m_list.append(m_t)

        g_seq = torch.stack(g_list, dim=1)  # (B, W, global_out)
        m_seq = torch.stack(m_list, dim=1)  # (B, W, local_out)
        return g_seq, m_seq


# =============================================================================
# 4. Conditional Flow Matching Generator
# =============================================================================

class CrossAttentionBlock(nn.Module):
    """Cross-attention: z_s queries (g_t, m_t) as key/value."""
    def __init__(self, d_model: int, cond_dim: int, n_heads: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(cond_dim)
        self.attn = nn.MultiheadAttention(d_model, n_heads, kdim=cond_dim, vdim=cond_dim, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        z:    (B, L_z, d_model)  — noised latent sequence tokens
        cond: (B, N_c, cond_dim) — condition tokens from (g_t, m_t)
        """
        z_normed = self.norm_q(z)
        cond_normed = self.norm_kv(cond)
        attn_out, _ = self.attn(z_normed, cond_normed, cond_normed)
        z = z + attn_out
        z = z + self.ff(z)
        return z


class VelocityNet(nn.Module):
    """
    Velocity field network for flow matching.
    Predicts v_theta(z_s, s, g_t, m_t) where s is the flow time.

    Architecture: MLP + cross-attention layers.
    """
    def __init__(
        self,
        latent_dim: int = 128,
        cond_dim: int = 192,  # global_out + local_out
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Flow time embedding (sinusoidal)
        self.time_mlp = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Project latent z to hidden
        self.z_proj = nn.Linear(latent_dim, hidden_dim)

        # Cross-attention layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(CrossAttentionBlock(hidden_dim, cond_dim, n_heads))

        # Output projection
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
        """t: (B,) -> (B, dim)"""
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(
        self,
        z: torch.Tensor,      # (B, latent_dim) — noised latent
        s: torch.Tensor,       # (B,) — flow time in [0, 1]
        g: torch.Tensor,       # (B, global_out) — global condition
        m: torch.Tensor,       # (B, local_out) — local condition
    ) -> torch.Tensor:
        """Returns predicted velocity (B, latent_dim)."""
        # Time embedding
        s_emb = self.time_mlp(self.sinusoidal_embedding(s))  # (B, hidden)

        # Project z and add time
        h = self.z_proj(z).unsqueeze(1) + s_emb.unsqueeze(1)  # (B, 1, hidden)

        # Condition tokens: stack g and m as 2 tokens
        cond = torch.stack([g, m], dim=1)  # (B, 2, cond_dim) — requires g,m same dim or padded

        # Cross-attention layers
        for layer in self.layers:
            h = layer(h, cond)

        return self.out_proj(h.squeeze(1))  # (B, latent_dim)


class FlowMatchGen1D(nn.Module):
    """
    Conditional Flow Matching generator.

    Training: learns velocity field v_theta via CFM objective
    Inference: integrates ODE from z_0 ~ N(0,I) to z_1 (ECG latent)
    """
    def __init__(
        self,
        latent_dim: int = 128,
        global_out: int = 128,
        local_out: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
    ):
        super().__init__()
        cond_dim = global_out + local_out
        # Pad g and m to same dim for stacking as condition tokens
        self.g_proj = nn.Linear(global_out, cond_dim)
        self.m_proj = nn.Linear(local_out, cond_dim)

        self.velocity_net = VelocityNet(
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
        )
        self.latent_dim = latent_dim

    def compute_loss(
        self,
        z_1: torch.Tensor,    # (B, latent_dim) — target ECG latent
        g: torch.Tensor,       # (B, global_out)
        m: torch.Tensor,       # (B, local_out)
    ) -> torch.Tensor:
        """CFM training loss."""
        B = z_1.shape[0]
        device = z_1.device

        # Sample flow time s ~ U(0, 1)
        s = torch.rand(B, device=device)

        # Sample z_0 ~ N(0, I)
        z_0 = torch.randn_like(z_1)

        # Interpolate: z_s = (1 - s) * z_0 + s * z_1
        s_expand = s.unsqueeze(-1)
        z_s = (1 - s_expand) * z_0 + s_expand * z_1

        # Target velocity: z_1 - z_0 (optimal transport)
        target_v = z_1 - z_0

        # Predicted velocity
        g_cond = self.g_proj(g)
        m_cond = self.m_proj(m)
        pred_v = self.velocity_net(z_s, s, g_cond, m_cond)

        return F.mse_loss(pred_v, target_v)

    @torch.no_grad()
    def sample(
        self,
        g: torch.Tensor,
        m: torch.Tensor,
        n_steps: int = 20,
    ) -> torch.Tensor:
        """ODE integration from z_0 to z_1 via Euler method."""
        B = g.shape[0]
        device = g.device

        z = torch.randn(B, self.latent_dim, device=device)
        dt = 1.0 / n_steps

        g_cond = self.g_proj(g)
        m_cond = self.m_proj(m)

        for i in range(n_steps):
            s = torch.full((B,), i * dt, device=device)
            v = self.velocity_net(z, s, g_cond, m_cond)
            z = z + v * dt

        return z


# =============================================================================
# 5. ECG Encoder (for training target encoding) & Decoder
# =============================================================================

class ECGEncoder(nn.Module):
    """Encode ECG cycle to latent z_1 for flow matching training target."""
    def __init__(self, in_ch: int = 1, base_ch: int = 32, depth: int = 4, latent_dim: int = 128):
        super().__init__()
        self.stem = DoubleConv1D(in_ch, base_ch)
        blocks = []
        ch = base_ch
        for _ in range(depth):
            nxt = min(ch * 2, 512)
            blocks.append(Down1D(ch, nxt))
            ch = nxt
        self.downs = nn.ModuleList(blocks)
        self.proj = nn.Linear(ch, latent_dim)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """y: (B, 1, L) -> (B, latent_dim)"""
        h = self.stem(y)
        for down in self.downs:
            h = down(h)
        h = h.mean(dim=-1)  # deterministic global avg pool
        return self.proj(h)


class ECGDecoder(nn.Module):
    """
    Decode latent z_t -> ECG waveform with overlap conditioning.
    Output length = L + 2 * overlap_len, with overlap regions for cross-fade.
    """
    def __init__(
        self,
        latent_dim: int = 128,
        overlap_dim: int = 32,
        hidden_dim: int = 256,
        out_ch: int = 1,
        cycle_len: int = 256,
        overlap_len: int = 32,
        num_upsamples: int = 4,
    ):
        super().__init__()
        self.cycle_len = cycle_len
        self.overlap_len = overlap_len
        self.out_len = cycle_len + 2 * overlap_len  # extended output

        # Overlap condition encoding
        self.overlap_enc = nn.Sequential(
            nn.Linear(overlap_len, 64),
            nn.SiLU(),
            nn.Linear(64, overlap_dim),
        )

        total_input = latent_dim + overlap_dim
        upsample_factor = 2 ** num_upsamples
        # Ensure output length is achievable
        self.feat_len = self.out_len // upsample_factor
        if self.feat_len < 1:
            self.feat_len = 1

        self.fc = nn.Linear(total_input, hidden_dim * self.feat_len)
        self.hidden_dim = hidden_dim

        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU(inplace=True),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.SiLU(inplace=True),
            nn.Conv1d(16, out_ch, kernel_size=7, padding=3),
        )

    def forward(self, z: torch.Tensor, overlap: torch.Tensor) -> torch.Tensor:
        """
        z:       (B, latent_dim)
        overlap: (B, overlap_len)  — previous cycle tail
        Returns: (B, 1, out_len)
        """
        ov = self.overlap_enc(overlap)  # (B, overlap_dim)
        h = torch.cat([z, ov], dim=-1)  # (B, total_input)
        h = self.fc(h).view(h.size(0), self.hidden_dim, self.feat_len)
        y = self.deconv(h)
        # Crop or pad to exact output length
        if y.size(-1) > self.out_len:
            y = y[..., :self.out_len]
        elif y.size(-1) < self.out_len:
            y = F.pad(y, (0, self.out_len - y.size(-1)))
        return y


# =============================================================================
# 6. Keypoint Head (morphological constraint)
# =============================================================================

class KeypointHead(nn.Module):
    """
    Predict P/Q/R/S/T positions and amplitudes from decoded ECG waveform.
    Output: (B, 10) -> [p_pos, q_pos, r_pos, s_pos, t_pos, p_amp, q_amp, r_amp, s_amp, t_amp]
    """
    def __init__(self, cycle_len: int = 256, hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.SiLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(inplace=True),
        )
        # After stride-2 conv on cycle_len=256 -> 128, then reshape via chunked mean
        self.head = nn.Sequential(
            nn.Linear(32 * 16, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 10),
        )

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        """ecg: (B, 1, L) -> (B, 10)"""
        h = self.conv(ecg)  # (B, 32, L//2)
        # Deterministic pooling: reshape into 16 chunks and mean
        B, C, T = h.shape
        chunk = T // 16
        if chunk > 0:
            h = h[..., :chunk * 16].reshape(B, C, 16, chunk).mean(dim=-1)  # (B, 32, 16)
        else:
            h = h.mean(dim=-1, keepdim=True).expand(B, C, 16)
        h = h.reshape(B, -1)  # (B, 32*16)
        return self.head(h)


# =============================================================================
# 7. SincShift (differentiable sub-sample time shift)
# =============================================================================

class SincShift(nn.Module):
    """
    Differentiable fractional time shift using sinc interpolation.
    Predicts per-cycle shift tau from (f_t, delta_t) and applies it.
    """
    def __init__(self, feat_dim: int = 128, time_dim: int = 32, kernel_half: int = 8):
        super().__init__()
        self.kernel_half = kernel_half
        self.shift_net = nn.Sequential(
            nn.Linear(feat_dim + time_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),  # predict scalar shift (in samples)
            nn.Tanh(),  # constrain to [-1, 1], then scale
        )
        self.max_shift = 10.0  # max shift in samples

    def forward(
        self,
        signal: torch.Tensor,   # (B, 1, L)
        f_t: torch.Tensor,      # (B, feat_dim)
        delta_t: torch.Tensor,  # (B, time_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: shifted signal (B, 1, L), predicted shift tau (B,)
        """
        tau = self.shift_net(torch.cat([f_t, delta_t], dim=-1)).squeeze(-1) * self.max_shift  # (B,)
        shifted = self._apply_shift(signal, tau)
        return shifted, tau

    @staticmethod
    def _apply_shift(signal: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Apply fractional shift via linear interpolation (deterministic, differentiable)."""
        B, C, L = signal.shape
        # Build original index grid: [0, 1, ..., L-1]
        idx = torch.arange(L, device=signal.device, dtype=signal.dtype).unsqueeze(0).expand(B, -1)  # (B, L)
        # Shift: sample at (idx + tau) from original signal
        src = idx + tau.unsqueeze(-1)  # (B, L)
        # Clamp to valid range
        src = src.clamp(0, L - 1)
        # Integer and fractional parts for linear interpolation
        src_floor = src.long()
        src_ceil = (src_floor + 1).clamp(max=L - 1)
        frac = (src - src_floor.float()).unsqueeze(1)  # (B, 1, L)
        # Gather and interpolate per channel
        src_floor_exp = src_floor.unsqueeze(1).expand_as(signal)  # (B, C, L)
        src_ceil_exp = src_ceil.unsqueeze(1).expand_as(signal)
        val_floor = torch.gather(signal, 2, src_floor_exp)
        val_ceil = torch.gather(signal, 2, src_ceil_exp)
        return val_floor + frac * (val_ceil - val_floor)


# =============================================================================
# 8. Lightning Module
# =============================================================================

class LitCycleFlow(L.LightningModule):
    """
    Lightning Module for CycleFlow PPG→ECG generation.

    Combines all components and handles the full training/inference pipeline.
    """

    def __init__(
        self,
        # PPG Encoder
        enc_base_ch: int = 32,
        enc_depth: int = 4,
        feat_dim: int = 128,
        # Time2Vec
        time_dim: int = 32,
        # Dual-SSM
        global_hidden: int = 256,
        local_hidden: int = 128,
        global_out: int = 128,
        local_out: int = 64,
        overlap_dim: int = 32,
        use_mamba: bool = False,
        # Flow Matching
        flow_hidden: int = 256,
        flow_layers: int = 4,
        flow_heads: int = 4,
        latent_dim: int = 128,
        # ECG Decoder
        dec_hidden: int = 256,
        cycle_len: int = 256,
        overlap_len: int = 32,
        # Keypoint Head
        kp_hidden: int = 128,
        # Loss weights
        lambda_recon: float = 1.0,
        lambda_cfm: float = 1.0,
        lambda_morph: float = 0.1,
        lambda_align: float = 0.01,
        # Optimizer
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        # Inference
        n_sample_steps: int = 20,
    ):
        super().__init__()
        self.save_hyperparameters()

        # --- Build modules ---
        self.ppg_encoder = PPGEncoder(
            in_ch=1, base_ch=enc_base_ch, depth=enc_depth, feat_dim=feat_dim,
        )
        self.time2vec = Time2Vec(out_dim=time_dim)

        self.dual_ssm = DualSSM(
            feat_dim=feat_dim, time_dim=time_dim, overlap_dim=overlap_dim,
            global_hidden=global_hidden, local_hidden=local_hidden,
            global_out=global_out, local_out=local_out,
            use_mamba=use_mamba,
        )

        self.flow_gen = FlowMatchGen1D(
            latent_dim=latent_dim,
            global_out=global_out, local_out=local_out,
            hidden_dim=flow_hidden, n_layers=flow_layers, n_heads=flow_heads,
        )

        self.ecg_encoder = ECGEncoder(
            in_ch=1, base_ch=enc_base_ch, depth=enc_depth, latent_dim=latent_dim,
        )

        self.ecg_decoder = ECGDecoder(
            latent_dim=latent_dim, overlap_dim=overlap_dim,
            hidden_dim=dec_hidden, out_ch=1,
            cycle_len=cycle_len, overlap_len=overlap_len,
        )

        self.keypoint_head = KeypointHead(cycle_len=cycle_len, hidden=kp_hidden)

        self.sinc_shift = SincShift(feat_dim=feat_dim, time_dim=time_dim)

    # -----------------------------------------------------------------
    # Overlap cross-fade utility
    # -----------------------------------------------------------------
    @staticmethod
    def _crossfade_overlap(
        prev_tail: torch.Tensor,  # (B, 1, Lo)
        curr_head: torch.Tensor,  # (B, 1, Lo)
    ) -> torch.Tensor:
        """Linear cross-fade in overlap region."""
        Lo = prev_tail.shape[-1]
        alpha = torch.linspace(1, 0, Lo, device=prev_tail.device).view(1, 1, Lo)
        return alpha * prev_tail + (1 - alpha) * curr_head

    # -----------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        ppg = batch["ppg_cycles"]       # (B, W, L)
        ecg = batch["ecg_cycles"]       # (B, W, L)
        rr = batch["rr_intervals"]      # (B, W)
        kp_gt = batch["keypoints"]      # (B, W, 10)
        kp_mask = batch["keypoint_mask"]  # (B, W)
        overlap = batch["ecg_overlap"]  # (B, W, Lo)

        B, W, L = ppg.shape
        Lo = overlap.shape[-1]

        # --- 1. Encode all PPG cycles ---
        ppg_flat = ppg.reshape(B * W, 1, L)
        f_all = self.ppg_encoder(ppg_flat).reshape(B, W, -1)  # (B, W, feat_dim)

        # --- 2. Time encoding ---
        delta_all = self.time2vec(rr)  # (B, W, time_dim)

        # --- 3. Dual-SSM ---
        g_seq, m_seq = self.dual_ssm(f_all, delta_all, overlap)  # (B,W,go), (B,W,lo)

        # --- 4. Soft alignment: shift ECG targets ---
        ecg_flat = ecg.reshape(B * W, 1, L)
        f_flat = f_all.reshape(B * W, -1)
        delta_flat = delta_all.reshape(B * W, -1)
        ecg_shifted, tau = self.sinc_shift(ecg_flat, f_flat, delta_flat)
        ecg_shifted = ecg_shifted.reshape(B, W, L)
        tau = tau.reshape(B, W)

        # --- 5. Encode ECG targets to latent (flow matching target) ---
        z_1_all = self.ecg_encoder(ecg_shifted.reshape(B * W, 1, L)).reshape(B, W, -1)  # (B, W, latent)

        # --- 6. Flow matching loss ---
        g_flat = g_seq.reshape(B * W, -1)
        m_flat = m_seq.reshape(B * W, -1)
        z_1_flat = z_1_all.reshape(B * W, -1)
        loss_cfm = self.flow_gen.compute_loss(z_1_flat, g_flat, m_flat)

        # --- 7. Decode for reconstruction loss ---
        # Sample z from flow (use z_1 directly during training for recon, or teacher-force)
        overlap_flat = overlap.reshape(B * W, Lo)
        y_hat_ext = self.ecg_decoder(z_1_flat, overlap_flat)  # (B*W, 1, out_len)
        out_len = y_hat_ext.shape[-1]
        # Extract the core cycle region (skip overlap)
        y_hat_core = y_hat_ext[..., Lo: Lo + L]  # (B*W, 1, L)

        loss_recon = (
            F.l1_loss(y_hat_core, ecg_shifted.reshape(B * W, 1, L))
            + F.mse_loss(y_hat_core, ecg_shifted.reshape(B * W, 1, L))
        )

        # --- 8. Keypoint morphology loss ---
        kp_pred = self.keypoint_head(y_hat_core)  # (B*W, 10)
        kp_gt_flat = kp_gt.reshape(B * W, 10)
        kp_mask_flat = kp_mask.reshape(B * W)
        if kp_mask_flat.any():
            valid = kp_mask_flat.bool()
            loss_morph = F.mse_loss(kp_pred[valid], kp_gt_flat[valid])
        else:
            loss_morph = torch.tensor(0.0, device=ppg.device)

        # --- 9. Alignment regularization ---
        loss_align = (tau ** 2).mean()

        # --- Total loss ---
        loss = (
            self.hparams.lambda_cfm * loss_cfm
            + self.hparams.lambda_recon * loss_recon
            + self.hparams.lambda_morph * loss_morph
            + self.hparams.lambda_align * loss_align
        )

        # Logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("train/loss_cfm", loss_cfm, on_step=False, on_epoch=True, batch_size=B)
        self.log("train/loss_recon", loss_recon, on_step=False, on_epoch=True, batch_size=B)
        self.log("train/loss_morph", loss_morph, on_step=False, on_epoch=True, batch_size=B)
        self.log("train/loss_align", loss_align, on_step=False, on_epoch=True, batch_size=B)
        self.log("train/tau_mean", tau.abs().mean(), on_step=False, on_epoch=True, batch_size=B)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        ppg = batch["ppg_cycles"]
        ecg = batch["ecg_cycles"]
        rr = batch["rr_intervals"]
        kp_gt = batch["keypoints"]
        kp_mask = batch["keypoint_mask"]
        overlap = batch["ecg_overlap"]

        B, W, L = ppg.shape
        Lo = overlap.shape[-1]

        # Encode
        ppg_flat = ppg.reshape(B * W, 1, L)
        f_all = self.ppg_encoder(ppg_flat).reshape(B, W, -1)
        delta_all = self.time2vec(rr)
        g_seq, m_seq = self.dual_ssm(f_all, delta_all, overlap)

        # Generate via ODE sampling (inference mode)
        g_flat = g_seq.reshape(B * W, -1)
        m_flat = m_seq.reshape(B * W, -1)
        z_sampled = self.flow_gen.sample(g_flat, m_flat, n_steps=self.hparams.n_sample_steps)

        # Decode
        overlap_flat = overlap.reshape(B * W, Lo)
        y_hat_ext = self.ecg_decoder(z_sampled, overlap_flat)
        y_hat_core = y_hat_ext[..., Lo: Lo + L]

        # Recon loss (against original ECG, no shift in val)
        ecg_flat = ecg.reshape(B * W, 1, L)
        loss_recon = F.l1_loss(y_hat_core, ecg_flat) + F.mse_loss(y_hat_core, ecg_flat)

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

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        # Same as validation
        self.validation_step(batch, batch_idx)
        # Re-log under test/ prefix
        ppg = batch["ppg_cycles"]
        ecg = batch["ecg_cycles"]
        rr = batch["rr_intervals"]
        overlap = batch["ecg_overlap"]
        B, W, L = ppg.shape
        Lo = overlap.shape[-1]

        ppg_flat = ppg.reshape(B * W, 1, L)
        f_all = self.ppg_encoder(ppg_flat).reshape(B, W, -1)
        delta_all = self.time2vec(rr)
        g_seq, m_seq = self.dual_ssm(f_all, delta_all, overlap)
        g_flat = g_seq.reshape(B * W, -1)
        m_flat = m_seq.reshape(B * W, -1)
        z_sampled = self.flow_gen.sample(g_flat, m_flat, n_steps=self.hparams.n_sample_steps)
        overlap_flat = overlap.reshape(B * W, Lo)
        y_hat_ext = self.ecg_decoder(z_sampled, overlap_flat)
        y_hat_core = y_hat_ext[..., Lo: Lo + L]
        ecg_flat = ecg.reshape(B * W, 1, L)

        loss_recon = F.l1_loss(y_hat_core, ecg_flat) + F.mse_loss(y_hat_core, ecg_flat)
        self.log("test/loss", loss_recon, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=B)

    # -----------------------------------------------------------------
    # Full inference: generate ECG from PPG cycle sequence
    # -----------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        ppg_cycles: torch.Tensor,    # (1, W, L) or (W, L)
        rr_intervals: torch.Tensor,  # (1, W) or (W,)
        n_steps: int = 20,
    ) -> torch.Tensor:
        """
        Generate full ECG waveform from PPG cycles with overlap cross-fade.
        Returns: (total_samples,) — concatenated ECG waveform.
        """
        if ppg_cycles.ndim == 2:
            ppg_cycles = ppg_cycles.unsqueeze(0)
            rr_intervals = rr_intervals.unsqueeze(0)

        B, W, L = ppg_cycles.shape
        Lo = self.hparams.overlap_len
        device = ppg_cycles.device

        # Encode PPG
        f_all = self.ppg_encoder(ppg_cycles.reshape(B * W, 1, L)).reshape(B, W, -1)
        delta_all = self.time2vec(rr_intervals)

        # Run SSM autoregressively with generated overlap
        h_g, h_l = self.dual_ssm.init_hidden(B, device)
        prev_tail = torch.zeros(B, Lo, device=device)
        generated_cycles = []

        for t in range(W):
            g_t, m_t, h_g, h_l = self.dual_ssm.forward_step(
                f_all[:, t], delta_all[:, t], prev_tail, h_g, h_l,
            )

            # Sample from flow
            z_t = self.flow_gen.sample(g_t, m_t, n_steps=n_steps)  # (B, latent)

            # Decode
            y_ext = self.ecg_decoder(z_t, prev_tail)  # (B, 1, out_len)
            y_core = y_ext[:, 0, Lo: Lo + L]  # (B, L) — core cycle

            # Cross-fade with previous tail if not first cycle
            if t > 0 and Lo > 0:
                curr_head = y_ext[:, 0, :Lo]  # (B, Lo)
                faded = self._crossfade_overlap(
                    prev_tail.unsqueeze(1), curr_head.unsqueeze(1),
                ).squeeze(1)  # (B, Lo)
                # Replace the tail of the previous cycle
                if len(generated_cycles) > 0:
                    generated_cycles[-1][:, -Lo:] = faded

            generated_cycles.append(y_core)
            prev_tail = y_ext[:, 0, Lo + L: Lo + L + Lo]  # tail for next

        # Concatenate all cycles
        full_ecg = torch.cat(generated_cycles, dim=-1)  # (B, total_L)
        return full_ecg.squeeze(0)  # (total_L,) if B=1

    # -----------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
        )
        # Cosine annealing with warmup
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup = LinearLR(opt, start_factor=0.01, total_iters=self.hparams.warmup_steps)
        cosine = CosineAnnealingLR(opt, T_max=self.trainer.estimated_stepping_batches - self.hparams.warmup_steps)
        scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[self.hparams.warmup_steps])
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
