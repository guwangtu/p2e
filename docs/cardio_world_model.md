# CardioWorldModel: PPG2ECG as a World Model Problem

## 1. Motivation

传统的 PPG-to-ECG 方法将问题建模为**条件信号生成**（conditional generation）：给定 PPG 输入，直接回归或生成 ECG 输出。这种范式忽略了一个关键事实：PPG 和 ECG 都是同一个**心血管系统**的不同投影。

我们提出将 PPG2ECG 重新建模为一个 **World Model（世界模型）** 问题，借鉴强化学习中 Dreamer 系列的 RSSM 架构，显式地建模心血管系统的隐状态动力学。

### 为什么这个映射是自然的

| 信号 | 生理本质 | World Model 角色 |
|------|---------|-----------------|
| **ECG** | 心脏电活动的直接观测 | Observation $o_t$ |
| **PPG** | 心脏泵血经血管系统传导到外周的容积脉搏波 | Action $a_t$ |
| **心血管状态** | 心肌收缩力、血管阻力、自主神经张力等 | Latent State $(h_t, z_t)$ |

PPG 可以被理解为心脏（agent）对血管系统（environment）施加的 "动作"——心脏泵出的血流经过动脉树的阻抗、顺应性等变换后，在指尖产生了 PPG 信号。ECG 则是心脏电活动的直接观测。World Model 的目标是：**从 PPG（action）学会预测 ECG（observation）**，通过建模中间的心血管隐状态。

## 2. Architecture

### 2.1 Overall Framework

```
Training (posterior — grounded in ECG):
                                        ┌──────────────┐
  PPG cycle ──► PPGEncoder ──► a_t ──┐  │              │
                                     ├──► GRU ──► h_t ──┬──► Posterior q(z|h,o) ──► z_t
  RR interval ─► Time2Vec ──► δ_t ──┘  │              │  │                          │
  Overlap ────► OverlapEnc ──► ov_t ──┘  │              │  │  ECGObsEncoder           │
                                        └──────────────┘  │      │                    │
                                                          │  ECG cycle ──► obs_feat   │
                                                          │                           │
                                                          └───────────┬───────────────┘
                                                                      ▼
                                              (h_t, z_t) ──► FlowMatchDecoder ──► ECG latent
                                                                                      │
                                                              Overlap ──► ECGDecoder ──► ECG waveform


Inference (prior — imagination, no ECG):

  PPG cycle ──► PPGEncoder ──► a_t ──┐
                                     ├──► GRU ──► h_t ──► Prior p(z|h) ──► z_t
  RR interval ─► Time2Vec ──► δ_t ──┘                                      │
  Overlap ────► OverlapEnc ──────────┘                                      │
                                                                            ▼
                                              (h_t, z_t) ──► FlowMatchDecoder ──► ECG latent
                                                                                      │
                                                              Overlap ──► ECGDecoder ──► ECG waveform
```

### 2.2 RSSM (Recurrent State Space Model)

RSSM 是世界模型的核心，将隐状态分为两部分：

- **$h_t$（确定性递归状态）**：由 GRU 维护，捕获心血管系统的长期动态趋势（心率漂移、ST 段变化等）
- **$z_t$（随机隐变量）**：捕获逐拍的随机变异（呼吸性窦性心律不齐、偶发早搏等）

状态转移方程：

$$h_t = \text{GRU}(h_{t-1},\ [z_{t-1},\ a_t,\ \delta_t,\ ov_t])$$

$$\text{Prior: } z_t \sim \mathcal{N}(\mu_\theta(h_t),\ \sigma_\theta(h_t))$$

$$\text{Posterior: } z_t \sim \mathcal{N}(\mu_\phi(h_t, o_t),\ \sigma_\phi(h_t, o_t))$$

### 2.3 Component Summary

| Component | Class | Input | Output |
|-----------|-------|-------|--------|
| Action Encoder | `PPGEncoder` | PPG cycle $(B, 1, L)$ | feature $(B, \text{feat\_dim})$ |
| Obs Encoder | `PPGEncoder` (separate) | ECG cycle $(B, 1, L)$ | feature $(B, \text{obs\_dim})$ |
| ECG Latent Encoder | `ECGEncoder` | ECG cycle $(B, 1, L)$ | latent $(B, \text{flow\_latent\_dim})$ |
| Time Encoder | `Time2Vec` | RR interval scalar | $(B, \text{time\_dim})$ |
| World Model | `RSSM` | $(h_{t-1}, z_{t-1}, a_t, \delta_t, ov_t, [o_t])$ | $(h_t, z_t, \mu, \sigma)$ |
| Flow Decoder | `WMFlowMatchDecoder` | $(h_t, z_t)$ | ECG latent $(B, \text{latent\_dim})$ |
| Waveform Decoder | `ECGDecoder` | ECG latent + overlap | ECG waveform $(B, 1, L)$ |
| Keypoint Head | `KeypointHead` | ECG waveform | P/Q/R/S/T pos+amp $(B, 10)$ |
| Reward Head | `RewardHead` | $(h_t, z_t)$ | predicted RR interval $(B, 1)$ |

## 3. Training Objectives

### 3.1 Loss Function

$$\mathcal{L} = \lambda_\text{cfm} \mathcal{L}_\text{CFM} + \lambda_\text{recon} \mathcal{L}_\text{recon} + \lambda_\text{kl} \mathcal{L}_\text{KL} + \lambda_\text{morph} \mathcal{L}_\text{morph} + \lambda_\text{rr} \mathcal{L}_\text{RR} + \lambda_\text{align} \|\tau\|^2$$

| Loss | Purpose | Default Weight |
|------|---------|---------------|
| $\mathcal{L}_\text{CFM}$ | Flow matching velocity field regression | $\lambda = 1.0$ |
| $\mathcal{L}_\text{recon}$ | L1 + MSE waveform reconstruction | $\lambda = 1.0$ |
| $\mathcal{L}_\text{KL}$ | KL(posterior \|\| prior) — **world model core** | $\lambda = 0.1$ |
| $\mathcal{L}_\text{morph}$ | ECG keypoint (P/Q/R/S/T) morphology | $\lambda = 0.1$ |
| $\mathcal{L}_\text{RR}$ | RR interval prediction (self-supervised reward) | $\lambda = 0.05$ |
| $\|\tau\|^2$ | SincShift alignment regularization | $\lambda = 0.01$ |

### 3.2 KL Balancing (Dreamer-v2)

直接优化 $D_{KL}(q \| p)$ 容易导致 posterior collapse 或 prior 训练不稳定。我们采用 Dreamer-v2 的 balanced KL 策略：

$$\mathcal{L}_\text{KL} = \alpha \cdot D_{KL}(q_\phi \| \text{sg}[p_\theta]) + (1 - \alpha) \cdot D_{KL}(\text{sg}[q_\phi] \| p_\theta)$$

- $\alpha = 0.8$：80% 梯度训练 posterior（更准确地编码 ECG），20% 训练 prior（更好地从 PPG 预测）
- **Free nats** = 1.0：KL 低于阈值时不优化，避免过度压缩隐空间

### 3.3 Training vs Inference

| | Training | Inference (Imagination) |
|---|---------|----------------------|
| $z_t$ 来源 | Posterior $q(z_t \mid h_t, o_t)$ | Prior $p(z_t \mid h_t)$ |
| 需要 ECG | Yes | **No** |
| ECG 生成 | Teacher-forced decode | Flow ODE sampling |
| 用途 | 学习心血管动力学 | 纯 PPG → ECG 推理 |

## 4. Comparison with CycleFlow

| | CycleFlow | CardioWorldModel |
|---|-----------|-----------------|
| 序列建模 | DualSSM (global + local) | RSSM (deterministic + stochastic) |
| 隐空间性质 | 确定性 $(g_t, m_t)$ | $h_t$ 确定性 + $z_t$ 随机 |
| 训练信号 | CFM + recon + morph + align | CFM + recon + **KL** + morph + **RR reward** + align |
| 推理模式 | Flow sampling with overlap | **Imagination** — 纯 prior rollout |
| 不确定性估计 | 无 | Prior vs Posterior KL → 逐拍置信度 |
| 异常检测 | 无 | `compute_surprise()` |
| 参数量 | ~10M | ~13.9M |

## 5. Unique Capabilities

### 5.1 Imagination (Dreaming)

```python
model = LitCardioWorldModel.load_from_checkpoint("path/to/checkpoint.ckpt")
ecg_waveform = model.imagine(ppg_cycles, rr_intervals, n_steps=20)
```

模型在 latent space 中 autoregressive rollout，每一步只用 prior（不看 ECG），实现纯 PPG → ECG 转换。

### 5.2 Anomaly Detection via Surprise

```python
surprise = model.compute_surprise(ppg_cycles, ecg_cycles, rr_intervals, overlap)
# surprise: (W,) — per-cycle KL divergence scores
# High surprise = PPG 无法预测 ECG = 可能存在心律异常
```

当 prior（仅 PPG 预测的状态）和 posterior（结合 ECG 的真实状态）差距大时，说明当前心拍的 ECG 形态超出了 PPG 可预测的范围，可能存在心律失常或其他异常。

### 5.3 Uncertainty Quantification

每个生成的 ECG cycle 都有对应的 prior 方差 $\sigma_\theta(h_t)$，可以直接作为不确定性估计：

- 方差大 → 模型对该 cycle 的预测不确定
- 方差小 → 模型对该 cycle 的预测有信心

## 6. File Locations

```
D:\p2e\
├── scripts/models/cardio_world_model.py   # Model implementation (~550 lines)
│   ├── RSSSMPrior                         # Prior network p(z|h)
│   ├── RSSMPosterior                      # Posterior network q(z|h,o)
│   ├── RSSM                              # Full RSSM with transition/prior/posterior
│   ├── WMVelocityNet                      # Flow matching velocity field
│   ├── WMFlowMatchDecoder                 # CFM decoder conditioned on (h, z)
│   ├── RewardHead                         # RR prediction reward
│   └── LitCardioWorldModel               # Lightning Module (train/val/imagine/surprise)
│
├── configs/mimic_cardio_wm.yaml           # Training config
│
├── scripts/models/__init__.py             # Registered LitCardioWorldModel
│
└── docs/cardio_world_model.md             # This document
```

Reused from `cycle_flow.py`:
- `DoubleConv1D`, `Down1D` — convolutional building blocks
- `PPGEncoder` — action encoder (also used as observation encoder)
- `ECGEncoder` — ECG latent encoder for flow matching target
- `ECGDecoder` — latent → waveform with overlap conditioning
- `Time2Vec` — RR interval encoding
- `KeypointHead` — morphological constraint
- `SincShift` — differentiable time alignment
- `CrossAttentionBlock` — cross-attention for flow matching

## 7. Usage

### 7.1 Training

```bash
python run.py fit --config configs/mimic_cardio_wm.yaml --default configs/default.yaml
```

### 7.2 Testing

```bash
python run.py test --config configs/mimic_cardio_wm.yaml --default configs/default.yaml --ckpt runs/<run_dir>/checkpoints/best.ckpt
```

### 7.3 Inference (Python API)

```python
import torch
from scripts.models.cardio_world_model import LitCardioWorldModel

# Load trained model
model = LitCardioWorldModel.load_from_checkpoint("path/to/checkpoint.ckpt")
model.eval()

# --- Imagination: PPG → ECG ---
ppg_cycles = torch.randn(1, 32, 256)      # (batch, n_cycles, cycle_len)
rr_intervals = torch.rand(1, 32) * 0.4 + 0.6  # (batch, n_cycles) in seconds

ecg_waveform = model.imagine(ppg_cycles, rr_intervals, n_steps=20)
# ecg_waveform: (total_samples,) — concatenated ECG

# --- Anomaly Detection ---
ecg_cycles = torch.randn(1, 32, 256)
overlap = torch.randn(1, 32, 32)

surprise = model.compute_surprise(ppg_cycles, ecg_cycles, rr_intervals, overlap)
# surprise: (32,) — per-cycle KL score
anomaly_threshold = surprise.mean() + 2 * surprise.std()
anomalous_cycles = (surprise > anomaly_threshold).nonzero()
```

### 7.4 Key Hyperparameters to Tune

| Parameter | Config Key | Default | Notes |
|-----------|-----------|---------|-------|
| KL weight | `lambda_kl` | 0.1 | Too high → blurry ECG; too low → prior ignores PPG |
| KL balance | `kl_balance` | 0.8 | Higher → stronger posterior; lower → stronger prior |
| Free nats | `free_nats` | 1.0 | Higher → more expressive latent; lower → tighter |
| Stochastic dim | `z_dim` | 64 | Larger → more capacity for per-cycle variation |
| Deterministic dim | `h_dim` | 256 | Larger → more capacity for temporal dynamics |
| RR reward weight | `lambda_rr` | 0.05 | Helps temporal consistency |

## 8. Future Directions

1. **Latent Planning (CEM/MPPI)**：在 latent space 中搜索最优 ECG 轨迹，而非直接解码
2. **Few-shot Patient Adaptation**：冻结 transition model，只微调 posterior → 新患者快速适应
3. **Multi-modal Observation**：将 BP (血压) 作为额外 observation channel
4. **Hierarchical World Model**：beat-level + segment-level 双层 RSSM
5. **Contrastive Reward**：用对比学习替代 MSE 作为 reward signal
