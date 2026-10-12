---
name: opsd-noise-floor-measured
description: "canonical 臂两两差实测:中位 0.39×MDE、最大 1.23×MDE ⇒ 单 seed 判定阈要 2.5×MDE 或必须第二 seed 同号"
metadata:
  type: project
---

2026-09-29 实测(EXPERIMENT_LOG §4.614):**8 个未做任何干预的 canonical OPSD 臂**、
全部 ck100 / 官方四维 / 逐图配对 / n=1000,**28 个两两差**:

```
|Δ| 中位 = 0.39×MDE    均值 = 0.46×MDE    最大 = 1.23×MDE
```

**三轴全同的纯重跑**也能差到 **+0.56×MDE(t=+1.95)**(`baseAA100` − `baserep`,都是 A100/np3/seed20260830)。
跨硬件的差(0.04–1.11×MDE)**没有一致方向** ⇒ 硬件不是稳定偏移,被 run-to-run 方差盖住。
顺带标定:**np4 − np3 = +0.17×MDE(t=+0.58)**。
前提已核:脚本**无任何 LR 调度**(常数 AdamW 3e-4,无 warmup/scheduler.step)⇒ u200 的 ck100 与
u100 的 ck100 是同一训练轨迹点,可混入。

**Why:** 我此前用的判定阈是"|Δ|≥1×MDE 且 t>2 算正向"。上表里**一对 canonical-vs-canonical
就达到 1.23×MDE** ⇒ 那个阈值不安全。逐图 t 是对**图**的,不是对**训练 run** 的;
n=1000 让 t 轻易过 2,但它测不到 run-to-run 方差。

**How to apply:**
- 单 seed 判正/负要求 **|Δ| ≥ 2.5×MDE**(≈2× 观测到的最大 canonical 间距);
- **|Δ| 落在 1–2.5×MDE 时必须第二 seed 同号复现**才能判,否则只能记"未判"。
- 远在地板外的旧结论不受影响(`lr8x` +17.95、`qcoreQ7` −108 等是地板的 15–90×)。
- 落在地板内的旧判定(§4.561 −0.56、§4.563 +0.30、`qstrain7` −0.32、`rho2x` −0.23)
  结论不变但理由换成"落在 canonical 自己的重跑散布里"。

⚠️ **[[multik-multistate-distill]] 的 +1.46×MDE 正好落在含糊带**(最大 canonical 间距的 1.19×)
⇒ 单 seed **判不了**,等 seed 20260905 的同号复现或 ck100。
另见 [[small-n-significance-flips]]、[[aesthetic-not-a-heldout-axis]]、[[per-image-pattern-is-seed-dominated]]。
