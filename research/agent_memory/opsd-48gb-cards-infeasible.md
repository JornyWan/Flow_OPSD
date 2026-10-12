---
name: opsd-48gb-cards-infeasible
description: "OPSD/MSP 训练在 48GB 卡(L40S/A40)上不可行:加载 reward 模型时已占 43 GiB 就 OOM,梯度检查点 / opa_mb 都无效;48GB 卡只用于评测"
metadata:
  type: project
---

2026-10-11 smk_l40(ai-gpu13 L40S,44.4 GiB 可用):baseline / multikF4gate / opa_mb=2 四格全部在 `_load_reward_scorer`(clip_scorer `.to()`)时 OOM,
已分配 43.1 GiB —— 还没到第一步训练。`SD3_TRANSFORMER_GRADCKPT=1` 与 `opa_mb` 只管激活,管不到静态权重。

**Why:** 周末大卡被占时曾想用 ~19 张空闲 48GB 卡开新硬件组,白等 5 h 才发现。
**How to apply:** 训练只排 A100 / H100 / H200(≥80GB);48GB 卡(A40 / L40)只给评测。若真要用,得先把文本编码器 / reward 模型挪 CPU 或预计算(工程量大)。
相关:[[opd-cross-partition-equivalence]]、[[dont-edit-launcher-while-bash-reads-it]]
