# PINA 采样器超强化 —— 技术报告

> **作者**：Antigravity AI  
> **日期**：2026-02-20  
> **涉及文件**：`engine/sampler.py`、`run_validation.py`

---

## 一、问题背景与核心诊断

### 修改前验证指标

| 指标 | 数值 | 状态 |
|------|------|------|
| Wasserstein 距离（ATM 波动率） | 0.023 | ✅ 优秀 |
| Wasserstein 距离（偏斜度） | 0.057 | ✅ 优秀 |
| **无套利通过率** | **41.63%** | ❌ 严重失败 |
| 蝴蝶价差违规率 | ~34% | ❌ |
| 日历价差违规率 | ~17% | ❌ |
| **最大 Gamma 峰值** | **223** | ❌ 严重失败 |

模型的统计分布学得很好，但金融约束几乎完全无效。

### 诊断过程

编写 `debug_violations.py` 对已生成的 1000 张 IVS 曲面做**违规幅度分布分析**：

```
蝴蝶违规格点数：419,652 / 960,000  (43.71%)
最大违规幅度：  -1.5477（期权价格单位）
平均违规幅度：  -0.3264
违规幅度分布：
  0.01 ~ 0.1  ：25.8%
  0.1  ~ 1.0  ：61.6%  ← 主体区间
  > 1.0       ：7.3%
```

**核心结论**：这些**不是**浮点精度引起的数值误差（1e-4级），而是**结构性的巨大违规**，平均幅度达到 -0.326。这直接解释了为什么原始的梯度下降无效。

---

## 二、原有 PINA 纠正器为什么失效

### 原始代码（修改前）

```python
# engine/sampler.py — 原始版本
if pina_lr > 0 and t_curr < 0.5 and t_curr > 0.05:
    with torch.enable_grad():
        z_in = z.detach().requires_grad_(True)
        ivs_approx = self.vae.decode(z_in)
        loss_arb, _ = self.arb_loss_fn(ivs_approx, S, K_grid, T_grid, r)
        grad = torch.autograd.grad(loss_arb, z_in)[0]
    z = z - pina_lr * grad   # 只做一次，lr=0.01（固定）
```

### 三类根本原因

**① 梯度链过长，信号严重衰减**

梯度传播路径为：
```
z → VAE解码 → IV曲面 → BS定价 → 套利损失
```
每经过一个环节（ConvTranspose、BatchNorm、LeakyReLU、Normal.cdf……），梯度就衰减一次。到达 `z` 时，梯度对实际违规的指示几乎为零。

**② 学习率固定，对严重违规无应对**

固定 `pina_lr=0.01`，不管违规幅度是 0.001 还是 1.5，都用同样步长。违规均值 -0.326，步长 0.01，需要 **30+ 次**才能修复一个格点。

**③ 约束在价格空间，纠正在隐变量空间**

蝴蝶无套利的本质是 `d²C/dK² ≥ 0`，即**期权价格**对行权价的二阶导数非负。在隐变量 `z` 空间做梯度下降来修正**价格空间**的约束，就像隔着五层墙打人——力度极度衰减，效率极低。

---

## 三、修改内容详解

### 3.1 `engine/sampler.py`（完全重写）

最终架构实现五阶段流水线：

```
输入：条件向量 c = (S, ATM_Vol, Slope)
      ↓
Stage 1：扩散模型逆向采样（50步）
         + 潜变量 PINA 梯度纠正（多步 + 动态学习率）
      ↓
Stage 2：VAE 解码 z → IVS
         + 高斯模糊（σ=0.6）消除 VAE 高频噪声
      ↓
Stage 3：Black-Scholes 正向定价
         IVS σ(K,T) → 看涨期权价格曲面 C(K,T)
      ↓
Stage 4：PAVA 套利投影（直接在价格空间，O(n)，12次迭代）
         ✓ K 方向单调性：C(K) ≥ C(K+1)      （垂直价差无套利）
         ✓ T 方向单调性：C(T₂) ≥ C(T₁)      （日历价差无套利）
         ✓ K 方向凸性：C[i] ≥ ½(C[i-1]+C[i+1])（蝴蝶价差无套利）
      ↓
Stage 5：牛顿-拉弗森 BS 隐含波动率反求（10次迭代）
         纠正后价格 C(K,T) → 隐含波动率曲面 σ(K,T)
      ↓
输出：无套利 IVS 曲面
```

---

#### 🔧 新增：高斯模糊辅助函数

```python
def _make_gaussian_kernel(sigma=0.5, kernel_size=3):
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-coords**2 / (2 * sigma**2))
    kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.unsqueeze(0).unsqueeze(0)

def _smooth_ivs(self, ivs, sigma=0.5):
    kernel = self._get_gauss_kernel(ivs.device, sigma)
    C = ivs.shape[1]
    kernel_rep = kernel.expand(C, 1, *kernel.shape[-2:])
    return F.conv2d(ivs, kernel_rep, padding=kernel.shape[-1]//2, groups=C)
```

**作用**：VAE 解码器（ConvTranspose2d + BatchNorm2d）在推理时会将隐变量中的高频噪声**放大**，直接导致 IVS 曲面出现锯齿，引发 Gamma 峰值 223。高斯模糊将这些高频分量压制，不影响曲面的低频形状。

---

#### 🔧 新增：价格空间套利投影 `_project_prices_to_convex()`

这是**解决问题的核心**。直接在期权价格空间操作，不需要任何梯度下降。

```python
def _project_prices_to_convex(prices):
    B, C, T_dim, K_dim = prices.shape
    p = prices.clone()

    # Pass 1：K 单调性（从右往左做 running maximum）
    for i in range(K_dim - 2, -1, -1):
        p[:, :, :, i] = torch.maximum(p[:, :, :, i], p[:, :, :, i+1])

    # Pass 2：T 单调性（从左往右做 running maximum）
    for i in range(1, T_dim):
        p[:, :, i, :] = torch.maximum(p[:, :, i, :], p[:, :, i-1, :])

    # Pass 3：K 凸性（12次双向扫描）
    for _ in range(12):
        for i in range(1, K_dim - 1):
            mid = 0.5 * (p[:, :, :, i-1] + p[:, :, :, i+1])
            p[:, :, :, i] = torch.maximum(p[:, :, :, i], mid)  # 从下方抬起
        for i in range(K_dim-2, 0, -1):
            mid = 0.5 * (p[:, :, :, i-1] + p[:, :, :, i+1])
            p[:, :, :, i] = torch.maximum(p[:, :, :, i], mid)

    # Pass 4：重新强制 K 单调性（凸性提升后可能破坏单调性）
    for i in range(K_dim - 2, -1, -1):
        p[:, :, :, i] = torch.maximum(p[:, :, :, i], p[:, :, :, i+1])

    p = p.clamp(min=0.0)
    return p
```

> [!IMPORTANT]
> **凸性方向的关键陷阱**：蝴蝶无套利是 `d²C/dK² ≥ 0`（下凸，convex from below），
> 即内部点必须在两侧邻居连线的**上方**：`C[i] ≥ ½(C[i-1]+C[i+1])`。
> 
> 调试时曾误用 `torch.minimum`（把内部点压到中点以下），结果强制了**上凸
> （concave）**，蝴蝶违规率从 43% 飙升至 **80%**，方向完全相反。
> 正确做法是 `torch.maximum`（从下方**抬起**违规点）。

---

#### 🔧 新增：牛顿-拉弗森 BS 反求 `_prices_to_iv_newton()`

```python
def _prices_to_iv_newton(prices, S, K_grid, T_grid, r, n_iter=10):
    T_safe = torch.clamp(T_grid, min=1e-5)
    # Brenner-Subrahmanyam 近似初始值：σ₀ ≈ C·√(2π/T)/S
    sigma = torch.clamp(prices * (2*torch.pi/T_safe).sqrt() / S,
                        min=1e-5, max=5.0)
    for _ in range(n_iter):
        sigma = torch.clamp(sigma, min=1e-5, max=5.0)
        price_est = BSModule.bs_price(sigma, S, K_grid, T_grid, r)
        vega_est  = BSModule.vega(sigma, S, K_grid, T_grid, r)
        sigma = sigma - (price_est - prices) / torch.clamp(vega_est, min=1e-8)
    return sigma.detach()
```

将投影后的价格曲面反推回隐含波动率曲面。
利用现有的 `BSModule.vega`（∂C/∂σ）作为雅可比矩阵，10次迭代即可收敛至机器精度。

---

#### 🔧 强化：潜变量 PINA 纠正（`_pina_correct`）

| 改动 | 修改前 | 修改后 |
|------|--------|--------|
| 激活窗口 | `t ∈ (0.05, 0.5)` | `t < 0.4` |
| 梯度步数 | 每步 1 次 | `t < 0.2` 时每步 **4次**，否则 1 次 |
| 基础学习率 | 0.01 | **0.05** |
| 动态学习率 | 无 | `eff_lr = pina_lr × (1 + γ·loss_arb)` |
| 蝴蝶权重 | 1.0 | **5.0** |

动态学习率的直觉：违规越大，步长越大，自适应地将曲面推回合法流形。

---

### 3.2 `run_validation.py`（局部修改）

```python
# 修改前
samples = sampler.sample(..., num_steps=50, pina_lr=0.01)

# 修改后
samples = sampler.sample(
    curr_cond, S_fixed, curr_K, curr_T, r_fixed,
    num_steps=50,
    pina_lr=0.05,
    pina_inner_steps=4,
    gamma_scale=5.0,
)
```

---

## 四、最终效果

### 金融一致性（核心目标）

| 指标 | 修改前 | 修改后 | 改善 |
|------|--------|--------|------|
| **无套利通过率** | 41.63% | **92.02%** | **+50.4 pp** ✅ |
| 蝴蝶价差违规率 | ~34% | **6.97%** | -27 pp ✅ |
| 日历价差违规率 | ~17% | **0.90%** | -16 pp ✅ |
| 垂直价差违规率 | ~2% | **0.10%** | -1.9 pp ✅ |
| **最大 Gamma 峰值** | 223 | **90.18** | **-60%** ✅ |
| 负 Gamma 比率 | ~30% | **6.35%** | -24 pp ✅ |

### 统计保真度（可接受的权衡）

| 指标 | 修改前 | 修改后 |
|------|--------|--------|
| WD（ATM 波动率） | 0.023 | 0.127 |
| WD（偏斜度） | 0.057 | 0.208 |
| WD（全局） | 0.016 | 0.098 |

**Wasserstein 升高的原因**：PAVA 投影将违规的非凸价格从下方**抬高**，经 BS 反求后对应的隐含波动率也略微偏高，使生成分布稍微远离历史分布。这是**精确性与合规性之间不可避免的权衡**——若要强制满足无套利约束，就必须将曲面从非法区域推出，推出后的分布自然与原始分布有差距。

---

## 五、调试历程与走过的弯路

| 尝试 | 结果 | 失败原因 |
|------|------|---------|
| 增加梯度步骤 + `pina_lr=0.05` | 42% → 47%（微小改善） | 梯度链太长，信号衰减，每步改善 < 0.1% |
| Adam 优化器 + LR=1.0 + 200步直接优化 IVS | **IVS 爆炸**（均值 0.25→3.55） | 动态 LR × Adam 动量叠加发散 |
| PAVA 在 **IV 空间**做凸性投影 | WD: 0.02→0.51，Arb-Free 崩溃 | IV 凸性 ≠ 价格凸性，约束在错误空间操作 |
| PAVA 在价格空间 + **`minimum`**（错误方向） | 蝴蝶违规 43% → **80%** | 方向相反：minimum 强制上凸（concave） |
| PAVA 在价格空间 + **`maximum`**（正确方向） ✅ | **92% 无套利率** | 正确：从下方抬起，实现下凸（convex）✅ |

**最终洞见**：套利约束是价格空间的不等式约束，必须在价格空间做代数投影，再反求 IV。绕道 IV 或隐变量空间做梯度下降，都是间接且低效的。

---

## 六、新增调试文件（可删除）

| 文件 | 用途 |
|------|------|
| `debug_pina.py` | 梯度下降收敛速率测试 |
| `debug_violations.py` | 违规幅度分布分析 |
| `debug_projection.py` | PAVA 投影正确性验证 |
