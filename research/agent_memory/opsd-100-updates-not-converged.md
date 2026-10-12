---
name: opsd-100-updates-not-converged
description: DiffusionOPSD 官方的 100 updates 远未收敛(ck50→ck100 仍 +0.051, t≈13 = 5×MDE);所有 ck100 的 A/B 只是"任意早停点比较"
metadata:
  type: project
---

**2026-09-02 实测:Open3 目标合计在 ck50→ck100 之间仍以 t≈+13 上升,两条独立 baseline 一致。**

| | ck50 | ck100 | Δ | t |
|---|---|---|---|---|
| baseline np4 | 1.5843 | 1.6353 | +0.0510 | +12.74 |
| baseline np3 | 1.5827 | 1.6347 | +0.0520 | +12.98 |

Δ 是本设计 MDE(0.01014)的 **5 倍**。训练无 LR scheduler(常数 `train.learning_rate=3e-4`,
`config/base.py:69`),所以"跑更久"就是同配方延长,没有调度形状的干扰。

**Why 这件事影响全部结论:** 我们所有 A/B 都在 ck100 判定。既然 baseline 在该点仍在陡升,
那么每个判定其实是"在一个任意早停点上比高低":
- 只加快**收敛速度**的臂,会在这里显得更好;
- 只抬高**渐近值**、但前期慢的臂,会在这里显得更差。
两者都不是我们想问的问题。

**How to apply:**
1. 写任何 ck100 结论时,把"baseline 在该点未收敛"作为**前提写出来**,别默认读者知道;
2. 有正向苗头的臂,必须在**收敛点**再比一次,否则不能声称"更好";
3. 反过来,b4x(4× batch / 1/4 步数)在同样本量下差 −0.0924,但它自身 10→25 步还在涨 +0.0801
   ⇒ 它是被**步数**掐住,不能读成"大 batch 有害"。同一逻辑适用于任何改 B_eff 的臂。

验证中:`base200_np3`(job 1038944)从 ck100 续跑到 200,判据已预登记
(ck200−ck100 > 0.0101 则确认未收敛)。见 EXPERIMENT_LOG §4.144。

相关: [[opsd-open3-baseline-structure]] [[opd-need-positive-results]] [[no-smoke-configs]]
