# 0059 预注册:Gate Q1 —— suffix action-value 梯度是否修复 local→suffix 落差(2026-09-01 00:05,launch 前写定)

来源:Codex **0058**(第一性原理重置 + Q-OPSD)。本文只实现并预注册 **Gate Q1**;
Gate Q2 与训练臂按 0058 §7/§8,**Q1 不过则不做**。

## ① 要检验的命题

canonical DiffusionOPSD 隐含假设

    ∇_y R(D(y))  ≈  ∇_y Q_old(s,y),   Q_old(s,y) = R(D(F_old(s,y)))

即"局部 clean 代理的梯度 ≈ 真正的 action-value 梯度"。
我们已实测该近似**随训练恶化**:local→suffix 增益保持率 **63.4% → 43.0%**(§4.47)。

**Q1 直接问:换成真正的 `∇_y Q_old`,suffix 层的增益是否变好?**

## ② 三个方向,同 state、同半径、全部走 exact suffix

| 记号 | 定义 | 角色 |
|---|---|---|
| `d_clean` | `normalize(∇_y R(D(y0)))` | canonical 用的(= `opa_g0`) |
| **`d_Q`** | **`normalize(∇_y R(D(F_old(s,y0))))`** | **本方法** |
| `d_rand` | 同半径随机单位向量 | 归因对照 |

半径统一为 `rho·‖y0‖`(与 baseline 的 trust region 同一 budget)。
每个方向走一步得 `y_j`,再经 **exact suffix** 到终点打三轴分;
锚点用 `F_old(s,y0)` 的终点分,故所有增益同基准。

## ③ 判据(**0058 §7 建议值,本文采纳为正式阈值,先写后测**)

| # | 判据 | 阈值 |
|---|---|---|
| Q1-a | `d_Q` 的 negative-gain rate 相对 `d_clean` | **降低 ≥50%** |
| Q1-b | p10 分位的 suffix 增益 | 由负/近零 **恢复为正** |
| Q1-c | mean suffix 增益相对 `d_clean` | **提高 ≥15%** |
| Q1-d | 三个目标轴 | **无系统性显著下降** |
| Q1-e | 换 seed 复现方向 | 一致 |

**Q1-a~d 由本次探针直接判;Q1-e 需第二次探针(换 `RPI_SEED`),仅在 a~d 全过时才跑。**

> **全部通过才写 Q-OPSD 训练臂。任一条不过 ⇒ Q-OPSD 终止,
> 不调 `rho` 复活**(0058 §7 原话,本文照此执行)。

**必报的对照**:`d_rand` 的同一批数字。若 `d_Q` 只是略好于 `d_rand`,
说明收益来自扰动本身而非方向信息。

## ④ 预写失败模式

- **(a) `d_Q` ≈ `d_clean`**(`cos` 接近 1)⇒ 后缀 Jacobian 近似恒等,
  Q-OPSD 退化为 canonical,**不会有增益**。这是最可能的失败方式,
  故 `cos(d_clean,d_Q)` 是必报量,并按 `R_old` 分层看。
- **(b) `d_Q` 更好但幅度 <15%** ⇒ 不过 Q1-c;记为"方向确实更优但不足以支撑一条臂"。
- **(c) 数值不稳**:穿后缀反传可能 NaN / 梯度爆炸。必报 `gnorm`,出现 NaN 即判实现问题,
  **查真因,不绕过**。
- **(d) OOM**:反传图含 1 次 transformer 前向 + VAE 解码 + 3 个 reward 模型。
  若 OOM,**先减 `opa_mb` 而不是改方法**。

## ⑤ launch 前已完成的正确性验证

1. **可微后缀与原版逐位一致**(null sanity):同一 `forced_x0` 下
   `_dpm2_continue_from_grad` 与 `_dpm2_continue_from` 的终点**相对差 = 0.000e+00**。
2. **梯度确实回传到 `forced_x0`**:`‖∇_y Q‖ = 4.69e+01`(非零)。
3. **有限差分核对(double 输入)**:在信号最强的 `eps=1e-2` 上**相对误差 1.69e-05**。
   误差随 eps **减小而增大**(1e-6 时 0.225),这是**舍入主导**的签名 ——
   真因是 `_dpm2_continue_from_grad` 内部有 `z_k.detach().float()`,**计算在 float32**,
   传 double 也无效。⇒ **不是梯度 bug**,已定位并解释。
4. **守卫链五处全部同步放宽**(`_GATEA_LAST` 写入 / 兼容性校验 / `prev_x0` **生产** 与
   **消费** / 探针块),见 [[opsd-flag-chain-completeness]] —— 这类"只改一端"已栽过三次。
5. preflight 全绿;fork 与 pristine 仍只差 3 个已声明文件;`RPI_QGRAD_PROBE` 未设时
   代码路径与 baseline 逐字节相同。

## ⑥ 成本与规模

`UPDATES=6`(与 Gate A 同规模,便于对照)。
每 chunk 额外开销:1 次穿后缀反传 + 4 次 exact suffix 前向(锚点 + 三方向)+ 4×3 次打分。
**显著慢于 Gate A**,故先用小预算验证可行性。
