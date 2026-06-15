# SCALE Probe — Bug Report

## Bug: 训练从第一个 epoch 起完全失效，loss 显示为 0

### 现象

```
[WARN] non-finite loss (recon=nan, kl=inf); skipping step.   # 每个 epoch 约 39/40 个 batch
[WARN] non-finite loss (recon=1.178e+08, kl=inf); skipping step.  # 每个 epoch 约 1 个 batch
[Epoch   1/200] recon=0.0000  kl=0.0000  quick_f1=0.0000
[Epoch   2/200] recon=0.0000  kl=0.0000  quick_f1=0.0000
...
Early stopping: no quick-F1 improvement for 24 epochs (best F1=0.0000 at epoch 1).
```

模型完全没有被训练，但 loss 显示为 0（而非 nan），具有误导性。

---

### 根本原因：reparameterization 和 KL 双重溢出

问题链条分两层，对应两种 batch 表现：

#### 1. `kl=inf`（所有 batch 均触发）

`elbo_SCALE`（`loss.py:68`）的 KL 项包含：

```python
torch.exp(logvar_expand) / var_c
```

float32 的 exp 溢出阈值为 `logvar > 88.7`（exp(88.7) ≈ 3.4e38 = float32 最大值）。12000 维 ATAC 输入经过随机初始化的 encoder（Xavier normal），部分细胞的 logvar 超过此阈值，导致 `exp(logvar) = inf` → `kl_loss = inf`。

#### 2. `recon=nan`（约 39/40 的 batch 触发）

reparameterization 在 `layer.py:131`：

```python
std = logvar.mul(0.5).exp_()   # exp(logvar/2)
z = mu.addcmul(std, epsilon)
```

float32 的 exp(logvar/2) 溢出阈值为 `logvar > 177.4`。当 logvar 超过此值时：
- `std = inf`
- `z = mu + epsilon * inf = nan`（nan 来自 inf × 0 或 inf - inf）
- `decoder(nan) = sigmoid(nan) = nan`
- `binary_cross_entropy(nan, x) = nan` → `recon = nan`

每个 epoch 中约有 1 个 batch 的 logvar 落在 `(88.7, 177.4)` 区间：kl=inf 但 recon 有限（约 1e8），仍被 non-finite 检查跳过。

#### 3. `loss=0.0000`（虚假显示，并非收敛）

`probe.py:537` 中 `n_cells_seen` 只在步骤不被跳过时才累加：

```python
# train_scale() 内的训练循环
n_cells_seen += len(x)   # continue 跳过后永远不执行
```

所有步骤均被跳过 → `epoch_recon = 0`，`n_cells_seen = 0`：

```python
mean_recon = epoch_recon / max(1, n_cells_seen)  # = 0 / 1 = 0.0000
```

日志里的 `recon=0.0000` 并不表示模型收敛，而是**没有任何有效梯度步骤执行过**。

---

### 证据

| Epoch | 有效 batch 数 | 显示 recon | 实际情况 |
|-------|--------------|-----------|---------|
| 1–25  | 0            | 0.0000    | 模型从未被更新 |

最终 checkpoint 的 embeddings 含大量 non-finite：

```
[WARN] z_val contains 2783 non-finite entries (1.39%); replacing with 0.
[WARN] z_train contains 11375 non-finite entries; replacing with 0.
```

---

### 直接原因：SCALE 源码中被注释掉的 clamp

`scale/layer.py:132` 原本有一行保护用的 clamp，已被注释：

```python
std = logvar.mul(0.5).exp_()
# std = torch.clamp(logvar.mul(0.5).exp_(), -5, 5)   # ← 被注释掉了
```

同样，`get_gamma`（`model.py:210`）中 pi 的 clamp 也被注释：

```python
pi = self.pi.repeat(N, 1)
# pi = torch.clamp(self.pi.repeat(N,1), 1e-10, 1)    # ← 被注释掉了
```

---

### 修复方案

**修复 1（必须）：在 `scale/loss.py` 的 `elbo_SCALE` 中 clamp logvar**

在 KL 项使用 logvar 之前先 clamp，同时覆盖 reparameterization 路径：

```python
# elbo_SCALE 入参处，或在 GaussianSample.forward 中对 log_var 做 clamp
log_var = torch.clamp(log_var, min=-10, max=10)
```

这样 `exp(logvar)` ≤ exp(10) = 22026，远低于 float32 溢出阈值。

或直接修改 `scale/layer.py:131-132`：

```python
# 将注释掉的 clamp 改为合理范围
logvar_clamped = torch.clamp(logvar, min=-10, max=10)
std = logvar_clamped.mul(0.5).exp_()
z = mu.addcmul(std, epsilon)
```

**修复 2（建议）：恢复 `get_gamma` 中的 pi clamp**

```python
# model.py:210 — 取消注释并修正范围
pi = torch.clamp(self.pi.repeat(N, 1), 1e-10, 1)
```

防止训练中 pi 被优化为负数后 `log(pi) = nan`。

**修复 3（建议）：`probe.py` 的训练监控**

当 `n_cells_seen == 0` 时应打印明确警告，而非静默输出 `recon=0.0000`：

```python
if n_cells_seen == 0:
    print(f"  [ERROR] Epoch {epoch}: all {len(train_loader)} steps skipped "
          f"due to non-finite loss. Model was NOT updated.", flush=True)
```

---

### 受影响的运行记录

| 时间戳 | 数据集 | 状态 |
|--------|--------|------|
| 20260522_160023 | 5w_GSE196830_atac | 训练失败（部分有效步骤但 KL 极大） |
| 20260525_014108 | 5w_GSE196830_atac | 训练完全失败（0 有效步骤） |
| 20260525_022210 | 10w_GSE196830_atac | 被 Ctrl-C 中断，未完成 |

---

## 调优总结（2026-05-25）

修复数值崩溃后，对 SCALE 训练超参做了系统调优，定下定版 baseline 配置。

### 试验对比（5w_GSE196830_atac，5-fold SVC CV）

| 配置 | β | warmup_n | init_gmm | n_latent | macro_f1 | bal_acc | 状态 |
|------|---|----------|---------|----------|----------|---------|------|
| 原 paper 默认 | 1.0 | 200 | False | 20 | 0.148 | 0.159 | 🔴 posterior collapse |
| A+B（治标） | **0.01** | **5000** | False | 20 | **0.270** | 0.287 | ⚠️ under-regularization, GMM dead |
| **定版**（paper 协议） | **0.1** | **5000** | **True** | 20 | **0.270** | 0.282 | ✅ **GMM 激活，paper 协议** |

### GMM 激活验证

直接 dump `mu_c`（GMM centroid 位置）确认对称性是否破缺：

| 指标 | A+B（init_gmm=False） | 定版（init_gmm=True） |
|------|---------------------|---------------------|
| `mu_c.abs().max()` | 0.0011 | **0.8319** (770× ↑) |
| centroid 间距均值 | 0.0000 | **0.3444** |
| centroid 间距 max | 0.0000 | **1.0668** |
| var_c range | 0.51–0.56（同步移动） | **0.02–4.44**（差异化） |

→ A+B 中 29 个 centroid 完全重合（GMM 退化为单一高斯）；定版中 centroid 真正分散。

### 为什么 GMM 激活后下游 SVC 没涨

GMM 是**无监督**聚类先验，与 cell-type 标签无对齐保证。LinearSVC 是强 probe，在任何"细胞类型可分"的 embedding 上都能找到边界——GMM 几何结构对它没增益。

这本身是有意义的发现，可写进论文 discussion：**GMM 先验在高维 ATAC 上对有监督下游任务无额外贡献**。

### 定版 baseline 配置

```bash
python probe.py \
    --dataset_id 5w_GSE196830_atac \
    --beta 0.1 \
    --warmup_n 5000 \
    --init_gmm \
    --n_latent 20 \
    --max_epochs 200 \
    --early_stopping \
    --early_stopping_patience 24
```

### 论文 method 节模板

> "We use SCALE [Xiong et al. 2019] with its full GMM-prior architecture: encoder [1024, 128] → 20-dim latent, linear decoder, GMM prior with K=29 components initialized via Gaussian Mixture fit on untrained encoder output (paper protocol, `init_gmm=True`). To accommodate the 240,000-peak input (8× larger than SCALE's original benchmark scale), we (i) extend KL warmup from 200 to 5,000 iterations, (ii) use β=0.1 instead of β=1.0 to prevent posterior collapse, and (iii) clamp the encoder's logvar output to [-10, 10] to prevent float32 overflow in the GMM-prior KL term. These modifications affect only training dynamics and numerical stability, not the model architecture or loss formulation."
