---
name: opsd-pairing-must-match-three-axes
description: OPSD 逐图配对前必须核对三条:训练硬件(provenance)、评测硬件(文件名后缀)、外层 epoch(JSON 的 ckpt 路径)
metadata:
  type: feedback
---

CA-OPD/OPSD 做 arm-vs-baseline 逐图配对时,**三条都必须核对,缺一条就会算出假结果**:

1. **训练硬件** —— 必须同时读**两个**字段,**不能看目录名**:
   - `arm_provenance.json` 的 `gpu`(该 run 跑在哪张卡上)
   - `run_attempts.jsonl` 的 **`resumed`**(是否从别的 run 续跑)
   `baseline_np3` 是 H200 训练,`baseAA100` 是 A100 训练("AA"=A100)。
   δ_hw@o100 = baseAA − basenp3 = +0.0029。日志里"零校正"指的是零*训练*硬件校正。
   ⚠️ **2026-09-08 踩到**:三个 `*_u200` run 的 `gpu` 都显示 H100/A100,但 `resumed=1` ——
   它们是从对应的 `*_u100`(H200)**续跑**的 ⇒ 前 100 个 epoch 的权重其实来自 H200。
   `base200@ck100` 与 `basenp3@ck100` 的评测**逐图完全相同(1000/1000)**,就是同一个 checkpoint。
   只看 `gpu` 会误以为它是"纯 H100 训练的 baseline"。
2. **评测硬件** —— 文件名后缀 `_a100` / `_l40` / 无后缀(=A40)指的是**评测**卡。
   实测 A40 比 A100 系统性高 +0.00098(t=+3.59, 符号置换 p=0.031, n=7 独立 ckpt),
   A40−L40 ≈ +0.0013。跨评测硬件最大引入 55% MDE。同硬件配对则精确抵消。
3. **外层 epoch** —— 标签里的数字是 **ck 号**,必须读 JSON 的 `ckpt` 路径 + 该臂的 CK_MULT
   还原。`gacc4_200_l40` 是外层 50 不是 100;`inner2b_400_l40` 是外层 200 且来自 u200 run。

**Why:** 2026-09-07 一轮里我同时踩了 (1) 和 (3):按目录名给 A100 训练的臂配了 H200 的
baseline,又把 ck 号当外层 epoch,算出 gacc4=−0.01055 这种和日志完全矛盾的数。
查 provenance 后才确认是重算错了、日志是对的。

**How to apply:** 写配对脚本时先打印一张 `标签 → 训练目录 / 训练GPU / ck / CK_MULT / 外层epoch /
评测硬件` 的表,肉眼核对完再算 Δ。三条里任意一条不匹配就不要出数。
相关:[[opsd-flag-chain-completeness]] [[small-n-significance-flips]] [[print-the-series-first]]

**2026-09-10 补(第二次栽在同一处,新形式:同名臂散落在不同硬件)**:
`lr8x` 的 u100 在 **A100**、u200 在 **H200**;`gacc2` 的 s20260830 在 A100、s20260905 在 H200。
⇒ **"臂名 + seed" 不足以确定训练硬件**。我据此把 §4.523 的 lr8x 参照值报成 +18.34×MDE,
正确的纯 H200 值是 **+17.01×MDE**(差 1.33×MDE,恰等于 §4.480 实测的 H200−A100 偏移)。
`rho2x` 同样:对 H200 基线是 +0.16×MDE,对匹配的 A100 基线是 −0.56×MDE。
**How to apply:** 引用任何历史评测前,先读该 JSON 的 `ckpt` 字段 → 反查 `ab_open3_*` 目录 →
读 `arm_provenance.json` 的 host。**臂内**差值(同 ckpt 的步数 gap 等)对该偏移免疫,可放心用。

**2026-09-10 补(文件名陷阱)**:`outputs/matched_pi_<tag>_<N>_...` 里的 `<N>` 是 **ck 编号**,
inner 系列要除以 CK_MULT=2 才是外层 epoch —— `inner2100` 是 u100@ck100 = **外层 ep50**,不是 ep100。
另外 `_a100` / `_l40` 后缀指**评测**硬件,和训练硬件无关。
我据此取错文件,双重差算出 +0.03864 而正确值是 −0.00664(差 0.045)。
**How to apply:** 一律读 JSON 的 `ckpt` 字段反查臂目录与 ck 编号,再除 CK_MULT;不要从文件名推断。
