---
name: per-image-pattern-is-seed-dominated
description: "OPSD 逐图效应模式主要由训练 seed 决定,不是由干预决定;跨臂比较逐图模式必须跨 seed,否则相似度里全是 seed 共模"
metadata:
  type: project
---

**逐图效应向量 e_A = obj(臂A) − obj(baseline)(1000 维)的相似度,更多反映训练 seed,而不是干预。**
2026-09-08 实测(EXPERIMENT_LOG §4.385,官方口径 ck100):

- 同臂**不同 seed**:lr2x 0.304 / lr4x 0.345 / lr8x 0.497  ← 这才是可复现性天花板
- 同 seed **不同臂**:0.41 ~ 0.63  ← **比天花板还高**

**How to apply:** 任何"两条臂逐图模式很像 ⇒ 它们是同一个轴/同一种干预"的论证,
只要两臂**共用 seed** 就不成立。必须跨 seed 比,并按天花板去衰减:
`ρ = r_cross / sqrt(r_self(A)·r_self(B))`,且要 bootstrap 误差棒(比值统计量)。
去衰减后实测:lr4x×lr8x ρ=0.925(CI 覆盖 1),但 lr2x×lr4x ρ=0.621、lr2x×lr8x ρ=0.451
(CI 都不覆盖 1)⇒ **lr 族在逐图层面不是一个固定方向,方向随剂量旋转,低剂量端更快。**
⇒ [[opsd-open3-baseline-structure]] 里"所有改进在一条一维线上"(§4.298)只在
**三轴粗投影**下成立,逐图分辨率下已不成立。

硬件混杂可忽略:同一臂对的跨硬件读数 0.381 vs 同硬件读数 0.384(差 0.003)。
"逐图饱和"这个替代解释**无签名**(r=−0.008, t=−0.24;可排除 |r|>0.07)。

相关:[[small-n-significance-flips]] [[print-the-series-first]] [[identity-is-not-evidence]]
