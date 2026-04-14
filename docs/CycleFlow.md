# CycleFlow: Cycle-Level PPG-to-ECG Generation

> Dual State Space Models + Conditional Flow Matching

---

## 1. Overview

CycleFlow 是一个以**心动周期（cardiac cycle）为基本单元**的 PPG→ECG 条件生成框架。与传统方法在固定长度片段上做端到端回归不同，CycleFlow 将连续信号分割为逐周期序列，用状态空间模型建模跨周期时序动态，用条件流匹配在隐空间中生成 ECG 波形。

**核心创新点：**
- 心动周期作为 SSM 离散时间步，天然对齐生理语义
- 双通道 SSM 分离全局趋势与局部调制
- 条件 Flow Matching + cross-attention，建模 PPG→ECG 的一对多映射
- Overlap cross-fade 保证周期拼接平滑
- 形态关键点约束 + 可微对齐补偿 PTT 偏移

**参数量：** ~10.5M

---

## 2. 代码结构

```
D:\p2e\
├── scripts/
│   ├── models/
│   │   └── cycle_flow.py          ← 核心模型（所有模块）
│   └── data/
│       └── mimicbp_v2.py          ← 周期级数据集 + 预处理
├── configs/
│   └── mimic_cycle_flow.yaml      ← 训练配置
├── evaluation/
│   └── adapters/models/
│       └── cycle_flow_adapter.py  ← 评估适配器
├── test2.ipynb                    ← 可视化测试 notebook
├── run.py                         ← 训练/测试入口
└── docs/
    └── CycleFlow.md               ← 本文档
```

---

## 3. 方法详述

### 3.1 问题定义

给定连续 PPG 记录，首先通过 R-peak 检测分割为心动周期序列 {x₁, x₂, ..., x_T}，每个周期重采样至固定长度 L=256。目标是学习条件生成模型 p(ŷ_t | x_{≤t}, Δ_{≤t})，逐周期生成 ECG 波形。

### 3.2 整体架构

```
PPG cycle x_t ──→ [PPGEncoder] ──→ f_t ──┐
                                          │
RR interval Δ_t ─→ [Time2Vec] ──→ δ_t ──┤
                                          ├──→ [DualSSM] ──→ g_t (global), m_t (local)
prev cycle tail ──→ [OverlapEnc] ──→ ov_t┘           │
                                                      ↓
                                           [FlowMatchGen1D]
                                          (cross-attention on g_t, m_t)
                                                      │
                                                      ↓ z_t
                                            [ECGDecoder]
                                    (conditioned on overlap for smooth join)
                                                      │
                                                      ↓
                                             ŷ_t (ECG waveform)
                                                      │
                                            [KeypointHead] → 形态约束损失
```

### 3.3 各模块说明

#### PPGEncoder (`cycle_flow.py:59-83`)

1D-CNN 编码器，提取单周期 PPG 的形态特征。

- 结构：DoubleConv1D stem → 4 层 Down1D（stride-2 下采样）→ global mean pool → Linear
- 输入：`(B, 1, 256)` → 输出：`(B, 128)` 特征向量
- **f_t 仅送入 SSM，不直接送入生成器**，强制模型通过时序路径整合信息

#### Time2Vec (`cycle_flow.py:89-106`)

将标量 RR 间期 Δ_t 映射为高维表示，捕捉非线性时间依赖（如 QT-RR 的 Bazett 关系）。

```
δ_t = [ω₀·Δ_t + φ₀, sin(ω₁·Δ_t + φ₁), ..., sin(ω_k·Δ_t + φ_k)]
```

- 输入：`(B,)` 标量 → 输出：`(B, 32)` 向量
- ω_i, φ_i 均为可学习参数

#### DualSSM (`cycle_flow.py:131-252`)

双通道状态空间模型，分离不同时间尺度的上下文信息。

| 通道 | 隐状态维度 | 捕捉内容 | 输出 |
|---|---|---|---|
| Global | 256 | 心率趋势、ST 段基线漂移等低频变化 | g_t (128-d) |
| Local | 128 | 当前周期因呼吸/运动等引起的瞬时调制 | m_t (64-d) |

- 默认实现：GRUCell（确定性兼容）
- 可选：设 `use_mamba: true` 切换为 Mamba SSM（需 `pip install mamba-ssm`）
- 额外输入：上一周期尾部 overlap 经 MLP 编码后拼入，使隐状态感知边界

#### FlowMatchGen1D (`cycle_flow.py:296-407`)

条件流匹配生成器，在隐空间中建模 PPG→ECG 的一对多映射。

**训练：** 学习速度场 v_θ(z_s, s, g_t, m_t)

```
CFM Loss = E_{s,z₀,z₁} ‖v_θ(z_s, s, g_t, m_t) - (z₁ - z₀)‖²
```

其中 z₁ = Enc_ECG(y_t)，z₀ ~ N(0,I)，z_s = (1-s)z₀ + sz₁

**推理：** ODE 积分 z₀ → z₁（Euler method，默认 20 步）

**条件注入：** g_t、m_t 分别投影后作为 2 个 token，通过 cross-attention 被 z_s query

- VelocityNet：sinusoidal time embedding + 4 层 CrossAttentionBlock + output proj
- 隐空间维度：128

#### ECGDecoder (`cycle_flow.py:436-497`)

将隐表征 z_t 解码为 ECG 波形，输出长度 L + 2L_o = 320（含前后各 32 点 overlap 区域）。

- 输入：z_t (128-d) + overlap condition (32-d) → FC → 4 层转置卷积上采样
- 拼接时相邻周期 overlap 区域做线性 cross-fade：

```
ŷ[n] = α[n]·prev_tail[n] + (1-α[n])·curr_head[n],  α: 1→0
```

#### KeypointHead (`cycle_flow.py:504-531`)

辅助 ECG 形态约束模块，从解码波形预测 P/Q/R/S/T 关键点。

- 输出：`(B, 10)` → [P_pos, Q_pos, R_pos, S_pos, T_pos, P_amp, Q_amp, R_amp, S_amp, T_amp]
- 位置归一化到 [0, 1]，幅度为实际值
- 仅对有标注的周期计算损失（通过 keypoint_mask 过滤）

#### SincShift (`cycle_flow.py:537-585`)

可微分亚采样点时间平移，补偿 PPG 与 ECG 之间的脉搏传导时间（PTT）偏移。

- 从 (f_t, δ_t) 预测偏移量 τ_t ∈ [-10, +10] samples
- 通过 grid-based 线性插值实现可微平移（deterministic-safe）
- 正则化：L_align = ‖τ_t‖²，防止偏移过大

### 3.4 训练目标

```
L = λ_cfm · L_CFM + λ_recon · L_recon + λ_morph · L_morph + λ_align · L_align
```

| 损失项 | 公式 | 默认权重 | 说明 |
|---|---|---|---|
| L_CFM | MSE(v_pred, v_target) | 1.0 | 流匹配速度场损失 |
| L_recon | L1 + MSE | 1.0 | 波形重建损失（teacher-forcing） |
| L_morph | MSE(kp_pred, kp_gt) | 0.1 | 形态关键点损失 |
| L_align | ‖τ‖² | 0.01 | 对齐偏移正则化 |

### 3.5 数据预处理流水线

**输入：** 原始 PPG/ECG npy 文件（v1 格式：`{pid}_ppg.npy`, shape `(n_segments, T)`)

**预处理步骤：**

1. **带通滤波** — ECG: 0.5-40 Hz, PPG: 0.5-8 Hz (Butterworth 4th order)
2. **陷波滤波** — 50 Hz 去工频干扰
3. **Z-score 归一化** — 每段独立
4. **R-peak 检测** — neurokit2 (优先) 或 scipy find_peaks (fallback)
5. **周期切分** — 以 R-peak 为锚点，前 30% 后 70% 切分
6. **质量筛选** — RR ∈ [0.4, 1.5]s，周期长度 ≥ 10 samples
7. **三次样条重采样** — 每个周期 → 256 points
8. **ECG 关键点标注** — P/Q/R/S/T 位置+幅度（neurokit2 DWT 或 heuristic fallback）

**输出目录：**

```
data_dir/cycles/
  {pid}_ppg_cycles.npy      # (N_cycles, 256)
  {pid}_ecg_cycles.npy      # (N_cycles, 256)
  {pid}_rr_intervals.npy    # (N_cycles,)
  {pid}_keypoints.npy       # (N_cycles, 10)
  {pid}_cycle_meta.npz      # original_lengths, num_cycles
```

**代码位置：** `scripts/data/mimicbp_v2.py` 中的 `preprocess_and_save()` 和 `segment_cycles()`

---

## 4. 数据集类

### MimicBPCycleDataset (`mimicbp_v2.py:225-320`)

每个样本 = 连续 W 个周期的滑动窗口：

```python
{
    "ppg_cycles":    (W, 256),    # PPG 周期序列
    "ecg_cycles":    (W, 256),    # ECG 目标
    "rr_intervals":  (W,),        # RR 间期（秒）
    "keypoints":     (W, 10),     # P/Q/R/S/T 位置+幅度
    "ecg_overlap":   (W, 32),     # 上一周期尾部 overlap
    "keypoint_mask": (W,),        # 关键点是否有效
    "meta": {"pid", "start_cycle", "fs"}
}
```

- 训练集：stride=16（窗口重叠 50%）
- 验证/测试集：stride=W（无重叠）
- 数据用 `mmap_mode="r"` 懒加载，内存友好

### MimicBPCycleDataModule (`mimicbp_v2.py:330-430`)

Lightning DataModule，支持两种模式：
- `preprocessed=true`：直接读取 `cycles/` 目录
- `preprocessed=false`：首次运行自动预处理

---

## 5. 使用方法

### 5.1 环境准备

```bash
conda activate p2e

# 可选：安装 Mamba SSM（用于替换 GRU）
pip install mamba-ssm

# 可选：安装 neurokit2（更准确的 R-peak 检测和 ECG delineation）
pip install neurokit2
```

### 5.2 数据预处理

**方式 A：自动预处理（推荐）**

在 `configs/mimic_cycle_flow.yaml` 中设置 `preprocessed: false`，首次训练时自动执行。

**方式 B：手动预处理**

```python
from scripts.data.mimicbp_v2 import preprocess_and_save

preprocess_and_save(
    data_dir="E:\\datasets\\ppg2ecg\\MIMIC-BP",
    output_dir="E:\\datasets\\ppg2ecg\\MIMIC-BP",
    fs=125,
    cycle_len=256,
)
```

预处理完成后将 `preprocessed` 改回 `true` 以跳过后续重复处理。

### 5.3 训练

```bash
# 基本训练
python run.py fit --config configs/mimic_cycle_flow.yaml

# 如需 deterministic 模式（yaml 中 deterministic: true）
# Windows PowerShell:
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
python run.py fit --config configs/mimic_cycle_flow.yaml

# bash:
CUBLAS_WORKSPACE_CONFIG=:4096:8 python run.py fit --config configs/mimic_cycle_flow.yaml
```

训练日志与 checkpoint 保存在 `runs/<timestamp>_mimicbp_cycle_flow/`。

**TensorBoard 监控：**

```bash
tensorboard --logdir runs/
```

关注指标：
- `train/loss_cfm`：流匹配损失，核心指标
- `train/loss_recon`：重建损失
- `train/tau_mean`：PTT 偏移量，应稳定在较小值
- `val/loss`：验证总损失

### 5.4 测试

```bash
python run.py test \
    --config configs/mimic_cycle_flow.yaml \
    --ckpt runs/<your_run>/checkpoints/best.ckpt
```

### 5.5 可视化

打开 `test2.ipynb`，修改第一个 cell 中的 `CKPT_PATH`，依次运行：

| Section | 内容 |
|---|---|
| 1 | 原始 PPG/ECG 波形 |
| 2 | 滤波 + R-peak 检测 |
| 3 | 全景图 + 周期切分竖线标记 |
| 4 | 放大 5s 窗口，绿色=起点 / 橙色=终点 / 红点=R-peak |
| 5 | 加载模型生成 ECG，三行对比 + 单周期 overlay |
| 6 | 逐周期 RMSE / MAE / Pearson r |

如果没有训练好的 checkpoint，留空 `CKPT_PATH` 可用随机权重跑通流程。

### 5.6 推理 API

```python
from scripts.models.cycle_flow import LitCycleFlow
import torch

model = LitCycleFlow.load_from_checkpoint("path/to/best.ckpt")
model.eval()

# ppg_cycles: (1, W, 256), rr_intervals: (1, W)
ecg_waveform = model.generate(ppg_cycles, rr_intervals, n_steps=20)
# ecg_waveform: (W*256,) 连续 ECG 波形
```

---

## 6. 关键配置参数

配置文件：`configs/mimic_cycle_flow.yaml`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `window_size` | 32 | 每个样本包含的周期数 |
| `cycle_len` | 256 | 每周期重采样点数 |
| `overlap_len` | 32 | 周期间 overlap 区域长度 |
| `use_mamba` | false | 是否使用 Mamba 替换 GRU |
| `latent_dim` | 128 | 流匹配隐空间维度 |
| `flow_layers` | 4 | VelocityNet cross-attention 层数 |
| `lambda_cfm` | 1.0 | 流匹配损失权重 |
| `lambda_recon` | 1.0 | 重建损失权重 |
| `lambda_morph` | 0.1 | 形态约束权重 |
| `lambda_align` | 0.01 | PTT 对齐正则化权重 |
| `lr` | 1e-4 | 学习率 |
| `warmup_steps` | 1000 | 线性 warmup 步数 |
| `n_sample_steps` | 20 | 推理时 ODE 步数 |
| `batch_size` | 8 | 每 batch 窗口数 |
| `accumulate_grad_batches` | 2 | 梯度累积（有效 batch=16 windows） |

---

## 7. 评估适配器

用于接入项目统一评估框架 (`evaluation/`)：

```yaml
# evaluation config 示例
model:
  name: "cycle_flow"
  config:
    ckpt_path: "runs/<your_run>/checkpoints/best.ckpt"
    n_sample_steps: 20
```

代码位置：`evaluation/adapters/models/cycle_flow_adapter.py`

注册方式：通过 `@REGISTRY.register_model("cycle_flow")` 装饰器自动注册。
