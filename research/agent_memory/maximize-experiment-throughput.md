---
name: maximize-experiment-throughput
description: "原则——尽可能多排实验找突破口;minilab/scu 无卡数上限,排着等也值"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe
  modified: 2026-09-29T21:55:53.271Z
---

用户 2026-09-29 明确的工作原则:**尽可能多地做实验去找突破口**,找到真能带来收益的方向,
直到显著到可以发表。不要因为"还没判完上一个"就不排下一个。

**Why:** 排队本身要数小时到数天;等判定再排等于白等。而排错了的代价只是一次 `scancel`。
本项目至今 15/15 目标端干预为负 —— 在这种命中率下,**吞吐量就是找到正向结果的主要变量**。

**How to apply:**
- **`minilab-gpu` 与 `scu-gpu` 没有卡数上限**,可以随便排;只有 `sablab-gpu` 有**同时 6 卡**的上限
  (且必须用 `AllocTRES` 算实得,`squeue %b` 是申请值会高估)。
- 提交前仍照旧扫四个分区取最早起跑时间,但**硬件按判据的对照臂选**,不是按"哪个先空" ——
  跑在错硬件上的臂判不了(见 [[opsd-pairing-must-match-three-axes]])。
- 上一个臂判定前就排它的消融/变体是**对的**;若主臂落显著负,直接 scancel。
- 每个新臂仍必须:preflight 过、watcher 挂上、核 CK_MULT、训练臂申请**整节点 4 张**
  (申请 3 会被部分授予,§4.591 因此死过一次)、判据**数据前**登记。
- 探针优先:1 卡、几十分钟、能定死一条线的设计选择 —— 这类的性价比远高于训练臂。

与 [[dont-kill-idea-on-first-negative]] 配合:多排是为了**广度**,首轮为负先归因是为了**深度**,两者都要。
