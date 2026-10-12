---
name: train-metric-not-heldout-proxy
description: 训练曲线不能用来**比较不同的臂**(可整体反号);但同一条臂**内部的变化**它追踪得很好(r=0.82–0.98)
metadata:
  type: feedback
---

**2026-09-02:Q-OPSD 训练侧 +0.0514(自相关校正 t=+23.5)→ held-out −0.0305。**
我据此写了"d_Q 过拟合到训练 prompt",**这个解释是错的**,被 codex 审计(0069 §7)推翻,我已逐条复核确认:

1. **训练根本没重复看同一批 prompt。** `train_rewards/epoch_*.jsonl` 记录了实际用过的 prompt:
   100 epoch 共 115200 行、去重后仅 **4391 条**,而语料有 25415 条 ⇒ **约 83% 从未被训练**。
   抽样是 `torch.randperm(len(dataset))[:48]`,每 epoch 重抽。
   且 jsonl 里有 `global_step_before_update` ⇒ reward 在本 epoch 更新**之前**记录。
   ⇒ 训练曲线量的是"对新 Pick-a-Pic prompt 的**域内泛化**",不是记忆。
2. **训练与正式评测同时改了四件事**,不止 prompt 集:

   | | 训练 | 正式评测 |
   |---|---|---|
   | prompt | Pick-a-Pic 用户语料(中位 12 词) | DrawBench(中位 9 词,组合性) |
   | sampler | DPM2,**10 步** | flow,**40 步** |
   | 每 prompt 图数 | 24 | 1 |
   | 权重 | outer-loop `old`(behavior) | checkpoint 的 default/EMA |

   (注:仓库把 DrawBench 放在 `data/pickapic/test.txt`,与 `data/drawbench/test.txt` **字节相同**;
   路径会误导人以为评测在 Pick-a-Pic 上。)
3. 对 **Q 方法**尤其致命:`d_Q` 的梯度里含**特定 suffix solver 的 Jacobian**。
   在 DPM2-10 上更正确的方向,对 flow-40 不一定还正确。**sampler 迁移必须单独测。**

**Why:** "训练指标与评测目标是同一个标量化"只说明**打分函数**相同,不说明**生成分布和被评权重**相同。
把多因子同时变化的 gap 归给其中一个因子,是我这次犯的错。

**2026-09-08 量化补充(16 个 (臂,阶段) 点,lr2x/lr4x/lr8x/lr10x):**

```
臂内相关(同一臂跨阶段):  lr2x r=+0.82   lr4x r=+0.91   lr8x r=+0.90   lr10x r=**+0.98**
臂间相关(稳定臂的均值):  r=+0.99
混在一起(全部 16 点):    r=+0.43,R²=0.19,**不显著**
```

混合后崩掉,是因为 **lr10x 的训练侧 Δ 整体落在负值区(−0.019 ~ −0.364),而 held-out Δ 仍为正 ⇒ 三个点全部反号。**

⇒ **可用 / 不可用的分界很清楚:**
- ✅ **同一条臂内部的变化**(如"ck30 → ck50 掉了多少")—— 追踪得很好,r=0.82–0.98。
- ❌ **比较不同的臂**(如"lr10x 是否优于 lr8x")—— 可以整体反号,完全不可用。

这解释了同一天的两个结果:用臂间比较做的预测失败了;用同臂变化做的验证成功了。

**How to apply:**
1. 训练曲线领先**仍然**不能当结论 —— 这条不变,判正判负一律等 held-out。
2. 但写"为什么没泛化"之前,**先把训练与评测的协议逐项列表数清楚**。
   差 N 项就有 N 个候选解释,只有做**单因子迁移的 2×2**才能分离。
3. 有消融臂时用"干预 vs 自身消融"的传导率做归因(qopsd−qbt 只传导 4.4%),
   比"干预 vs baseline"锋利 —— 这条仍成立。
4. **别把"某个数很小"直接翻译成一个机制故事。** 数是对的,故事可以是错的。

相关: [[small-n-significance-flips]] [[dont-kill-ideas-before-running]] [[opsd-100-updates-not-converged]] [[opsd-mirror-degeneracy-bug]]

**2026-09-10 强实例**:`lr10x` 的训练奖励在 ep10→40 一路正常上涨(1.4296→1.5415),
与 `lr8x` 只差 0.04–0.06;而同期 **held-out 质量已比 `lr8x` 差 5–7×MDE**(ep30 −0.0229、ep40 −0.0175)。
它到 ep56 才触发 §4.284 的发散判据。
⇒ **不是反号,是严重失校**:训练曲线"健康上涨"完全不能排除臂已经废了。
**高 lr 臂必须在 ep20/ep30 就排 held-out 评测**,不能只看训练曲线没崩。
