---
name: opd-ideas-tracker
description: "OPD/MSP 项目的 idea 与实验总表在 opd_related/IDEAS_TRACKER.md;用户要求每个 idea、在跑和跑完的实验都不能丢,每轮更新"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe
  modified: 2026-10-08T21:36:49.975Z
---

用户(2026-10-08):"请你妥善管理现在的每一个idea,不要忘记每个我们还在尝试以及已经跑完的实验和对应的idea"。
总表:`/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/IDEAS_TRACKER.md`
(§0 在跑/排队/hold 一览;§1–§5 按 idea 列状态、job、判定与 EXPERIMENT_LOG 节号)。

**Why:** 会话跨度长、多次压缩,idea 与实验(含 hold 的旧臂、被取消的臂、未判的探针)散在 4 万行日志里,用户担心被遗忘。

**How to apply:**
- 每次状态检查 / 投新实验 / 判定 / 取消 / hold 后,同步改 IDEAS_TRACKER.md 的 §0 和对应 idea 条目(写"最后更新"时间)。
- 用户转来新 idea 时先在总表里建条目(问题、实验、判据节号、新颖性边界),再动手。
- 汇报"情况如何"时以总表 §0 为骨架,不要只报最近一条线。
- 相关:[[opd-caopd-project]]、[[maximize-experiment-throughput]]
