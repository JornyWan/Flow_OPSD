---
name: msp-extra-grad-balanced-residual
description: "MSP 额外态梯度 = 好/坏样本两半大力相减的小量(cos≈−0.97,各为净值 3–4×);删坏样本那半(ADVPOS)= 拆配重,放出同向大梯度"
metadata:
  node_type: memory
  type: project
  originSessionId: 5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe
  modified: 2026-10-07T07:09:54.477Z
---

2026-10-07 smoke(§4.875 补 1)实测:MSP 额外态 loss 按原始 r1<0.5 拆两半,两半梯度范数各是净值的 3–4 倍、夹角 cos≈−0.97。
(2r1−1) 组内零均值 ⇒ 一致性项 C 的共模在好坏样本间抵消,剩下的才是有用信号。
ADVPOS(r1 夹到 ≥0.5)把配重删了:额外态梯度 ×3.3、跨步同向 cos 0.87、一步更新主项 loss +22% —— 不是"去掉正反馈"。

**Why:** §4.839 用代数把"坏样本那一半 κ>0"当成待删的不稳定源,差点花 15 h H200 跑 ADVPOS;smoke 的分解读数一眼否掉了这个设计。
后期突发(Adam 有效步长 0.2→0.73、梯度幅度 ~7×)的新假设是"两半平衡被打破",由 OPT_DIAG=2 的 fork 检验。

**How to apply:**
- 改 MSP 额外态的任何加权(按 advantage 截断/夹取/只取一侧)前,先用 `RPI_OPT_DIAG=2` 的 smoke 看两半的范数与夹角,别只看代数里的 κ。
- 相关:[[opsd-target-vold-gain-must-be-one]](κ 只看单样本,看不到跨样本抵消)、[[check-magnitude-or-direction-enters-training]]
