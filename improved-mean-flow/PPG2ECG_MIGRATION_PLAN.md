# PPG2ECG IMF工作流 

## 1）当前代码库在做什么

当前仓库是一个基于 JAX/Flax 的图像生成训练框架（iMeanFlow）：

- 入口：`main.py`
  - 初始化 JAX 分布式环境。
  - 读取 `configs/load_config.py:<mode>` 配置。
  - 进入训练 `train.train_and_evaluate(...)` 或评估 `train.just_evaluate(...)`。
- 训练主循环：`train.py`
  - 构建数据加载器。
  - 创建 `TrainState`（优化器、EMA）。
  - 执行 pmap 训练 step、周期性采样、checkpoint、评估。
- 算法核心：`imf.py`
  - iMF 训练目标与采样过程。
- 主干网络：`models/imfDiT.py`
  - 面向 2D 图像 patch 的 Transformer 双头结构。
- 指标工具：`utils/sample_util.py`、`utils/fid_util.py`
  - 当前默认是图像侧 FID/IS 评估。

一句话：这是一个成熟的“训练框架 + 图像模型实现”，我们后续主要改“任务相关模块”。

## 2）目标任务

新任务：`PPG -> ECG` 配对波形生成。

- 输入：PPG 一维时序。
- 输出：与输入时间对齐的 ECG 一维时序。
- 目标：先做稳定可训练的基线版本，再考虑复杂生成策略。

## 3）迁移策略（核心原则）

核心原则：**尽量少改，优先保留现有训练基础设施，只把 2D 模型等价改成 1D 时序模型。**

- 保留：
  - `main.py` 调度逻辑
  - 配置加载机制
  - 训练状态、优化器、EMA、日志、checkpoint
  - 训练循环框架（epoch/step/log/save）
- 最小必改：
  - 数据：从图像 latent 改为 PPG/ECG 配对序列
  - 模型：`2D patch Transformer -> 1D time-series Transformer`
  - 目标与指标：从图像评估改为波形监督损失和波形指标

## 4）模型改造主线（2D -> 1D，最快方案）

结论：直接把现有 `imfDiT` 做“结构等价的 1D 化”是最快、风险最低方案。

### 4.1 具体改法

- `PatchEmbedder(2D)` 改 `PatchEmbedder1D`
  - 输入从 `[B, H, W, C]` 改为 `[B, T, C]`
  - token 输出保持 `[B, N, D]`
- `unpatchify(2D)` 改 `unpatchify1D`
  - 从 token 恢复到 `[B, T, C]`
- Transformer block、attention、双头（`u/v`）尽量不动
  - 只适配维度和位置编码长度
- RoPE 继续使用 token 序列长度
  - 由 2D token 总数改为 1D token 总数
- 条件输入保留“条件 token”机制
  - 但任务条件从图像类别逐步迁移为 PPG 条件特征

### 4.2 参数建议（首版）

- `patch_size_1d`：建议先试 `8` 或 `16`
- `seq_len`：先固定（例如 1024 或 2048）
- 模型容量：先用接近 B 级别配置，确保能跑通和收敛

## 5）数据与损失的最小闭环

虽然主线是模型 2D->1D，但要跑起来必须同步做两件事：

- 数据闭环：
  - dataloader 输出配对 `ppg, ecg`
  - shape 对齐到训练 step 期望
- 损失闭环：
  - 首版使用 `L1 + L2`
  - 可选加一阶导数损失（后续再加）

首版不做 FID，不做复杂 CFG 搜索，先保证训练稳定和可评估。

## 6）明天执行清单（最小改动版）

1. 新增 `utils/wave_input_pipeline.py`
   - 读取配对波形，输出 `ppg/ecg` batch。
2. 基于 `models/imfDiT.py` 新增 `models/imfDiT_1d.py`
   - 尽量复制现有结构，只改 1D patch/embed/unpatchify。
3. 新增 `wave_train.py`
   - 复用 `train.py` 结构，替换 batch 字段和损失计算。
4. 新增 `utils/wave_metrics.py`
   - 先实现 MAE、RMSE、Pearson。
5. 配置新增
   - `configs/wave_train_config.yml`
   - `configs/wave_eval_config.yml`
6. 在 `main.py` 增加最小分流
   - 通过 `task.type` 选择 `train.py` 或 `wave_train.py`。

## 7）第一版验收标准

- 可以完整跑通 1 个 epoch，不崩溃。
- 训练 loss 在小样本子集上有下降趋势。
- 日志能看到 MAE/RMSE/Pearson。
- checkpoint 可保存并恢复。
- 给定一段 PPG，能输出同长度 ECG 预测。

## 8）风险与规避

- 风险：
  - PPG/ECG 对齐问题
  - 采样率与归一化不一致
  - 数据泄漏（同一受试者跨 train/val）
- 规避：
  - 严格按受试者划分数据集
  - 首步做数据完整性检查（长度、fs、缺失）
  - 先固定窗口长度与预处理策略，不频繁改动

## 9）明天建议顺序

1. 先打通数据加载（shape 正确）。
2. 再打通 1D 模型 forward（单步 loss 正常）。
3. 接上完整训练循环与 checkpoint。
4. 补评估和推理输出。
5. 最后再考虑复杂损失或更大模型。

---

这份文档已按“最小改动、2D 等价迁移到 1D”更新，可直接作为明天开发路线。
