---
name: opd-goal-40step-every-ckpt
description: "用户定的首要目标:40 步在每个 ckpt 都胜 canonical;最终交付 = open3 300 步 + 单 reward 100 步都胜 canonical;canonical 已复现完不再跑"
metadata:
  node_type: memory
  type: project
  originSessionId: 5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe
  modified: 2026-10-11T06:56:47.453Z
---

用户(2026-10-11):"不管怎样最重要的就是 40 步我们效果最好,并且在每个 ckpt 阶段都最好。"
(延续 2026-10-03 §4.837:"同一个方法、任意训练长度都胜过 OPSD",不接受后期关掉方法的方案。)

**Why:** 论文主张要落在满步质量上;MSP 的稳健优势目前只在 4 步,40 步只在 ck100 领先、之后缩小或崩。

**How to apply:**
- 排实验与选修复时以 flow-40 对 canonical 在 ck100/150/200/250/300 每一格的显著性为主判据;flow-4 只作副读数。
- 少步专长的改动(查询 σ↑、6 步 rollout、前后缀交换的 4 步结论)不再作为主线卖点。
- 判定必须同硬件配对、至少两个 seed;汇报时先给 40 步逐 ck 表。
- **最终交付(用户 2026-10-11 19:00)**:multi-reward(open3)**300 步**胜 canonical + single-reward **100 步**胜 canonical。
  canonical 已复现完(open3 300 步 H200×4 两 seed `o3canon300`/`canon300s2`;单 reward 100 步 `rs*canon` 一 seed,pickscore/hpsv2 = H100 NVL×4、
  clipscore/imagereward = H200 NVL×4),**不再跑 canonical,方法 run 按同硬件型号 / np4 / seed 去配现成对照**。
- 相关:[[opd-ideas-tracker]]、[[multik-multistate-distill]]、[[opsd-pairing-must-match-three-axes]]
