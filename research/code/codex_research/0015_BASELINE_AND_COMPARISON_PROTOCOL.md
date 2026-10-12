# 0015 — Guarded/continual alignment 的 baseline 与比较协议

Date: 2026-08-29  
Status: comparison protocol  
Jobs submitted by this entry: none

## 一句话答案

不是在 MOPD（DiffusionOPD）和 DiffusionOPSD 之间二选一：

- **released DiffusionOPD Student** 是共同初始化、旧能力 anchor 和 no-training baseline；
- **从同一个 Student 出发的 DiffusionOPSD-direct** 是我们方法最重要的 paired control；
- **从同一个 Student 继续 MOPD** 是次要但论文中应有的 multi-teacher/replay baseline；
- 正式表格必须以我们自己在统一协议下重跑的结果为准，官方数字只作为外部 reference / reproduction check。

---

## 1. 各 baseline 回答的是不同问题

| arm | 初始化 | 额外训练 | 它回答的问题 | 优先级 |
|---|---|---|---|---|
| `Student-0` | released DiffusionOPD Student | 无 | 原有 OCR/GE/Pick 能力到底是多少；所有 retention budget 的零点 | 必须 |
| `Student→OPSD-direct` | 同一个 Student | 只优化新 reward | 不加保护时能获得多少 Pick 增益、会遗忘多少旧能力 | **最关键主对照** |
| `Student→Ours` | 同一个 Student | OPSD + proposed guard/projection | 相对 direct-only，能否保留 reward gain 且减少遗忘 | **主方法** |
| `Student→MOPD-continue` | 同一个 Student | 继续多 teacher OPD | 传统 multi-teacher rehearsal/continued distillation 能否做到相同目标 | 正式 paper 应有 |
| `Student→fixed-hinge` | 同一个 Student | OPSD + fixed KL penalty | scalarized penalty 为什么不够；0010 已有负面证据 | 强 ablation/baseline |
| `Student→primal-dual` | 同一个 Student | OPSD + adaptive multipliers | 与已有 constrained-learning 思路相比，projection 是否有增量 | geometry gate 通过后再做 |

### 最重要的因果比较

```text
Student→Ours  vs  Student→OPSD-direct
```

两臂必须只差 capability-preservation mechanism。这个比较回答：

> 在同一个起点、同一个新 reward、同样的训练数据和预算下，guard 是否真正改善
> reward–forgetting trade-off？

单独把 Ours 与官方 DiffusionOPD Student 比，不能隔离方法作用：Ours 用了额外算力和新 reward，
Student 没有。Student-0 很重要，但只是 absolute anchor，不是唯一的算法对照。

### 为什么还要 MOPD-continue

审稿人自然会问：既然有旧 teachers，为什么不用原始 DiffusionOPD 继续蒸馏或 replay？

所以应当从同一个 Student checkpoint 继续训练 MOPD，并尽量匹配：

- 额外 optimizer steps；
- 各任务 prompt 数；
- teacher forwards / 总训练 FLOPs（至少同时报告）；
- checkpoint/evaluation schedule。

它是相关范式对照，但不是最接近 Ours 的主对照，因为它并没有使用完全相同的 OPSD direct
objective。

---

## 2. 为什么不能主要引用官方文档数字

正式因果结论必须用我们自己生成的结果。官方表格通常可能与当前实验在以下方面不同：

- 起始 checkpoint：base SD3.5-M vs released multi-capability Student；
- prompt train/test split；
- sampling seed、guidance scale、steps 和 resolution；
- evaluator 版本与 preprocessing；
- LoRA rank、训练步数、batch size 和 EMA/raw checkpoint；
- 单任务 OPSD setting vs 当前 continual multi-capability setting。

尤其是官方 DiffusionOPSD 结果若从 base model 开始，而我们从 DiffusionOPD Student 开始，
二者回答的已经不是同一个问题，不能直接做显著性比较。

官方数字有三个合理用途：

1. **reproduction sanity check**：我们的复现是否大致落在合理范围；
2. **背景/reference table**：说明 published method 报告过什么；
3. **发现实现偏差**：若我们复现差得很远，先查 evaluator/config，不急着比较新方法。

论文表格中可把官方数字用 `reported` / `†` 明确标注，但不能把它与我们的 seed-level 数据混在
同一个 t-test、置信区间或“显著优于”结论里。

---

## 3. 哪些需要自己重跑，哪些可以直接复用

### 必须自己跑

1. `Student→OPSD-direct`：必须和 Ours 同起点、同代码分支、同 seed；
2. `Student→Ours`；
3. `Student→MOPD-continue`：正式 paper 阶段；
4. 所有方法的 held-out evaluation：统一从本地 checkpoint 重新生成图并评分。

### 不一定要从头重训

`Student-0` 可以直接使用 released DiffusionOPD Student checkpoint，不需要重新完整训练
DiffusionOPD；但必须由我们的 evaluator 在相同 held-out prompts、seeds 和 generation config 下
重新评测。

### 现有结果能否复用

- 0010 的 `gdirect` 可作为 fixed-hinge 实验的 matched control；
- 但它使用 smoke tiny prompt pool，不能作为未来 real-task projection 实验的主对照；
- 新 geometry/projection 实验必须重新跑一个 real-task `Student→OPSD-direct` control；
- 0010 的 fixed-hinge 负结果可保留为工程/方法 ablation，但最好在 real prompts 上补一个短版本，
  不必再跑 200 epochs。

---

## 4. 正式 comparison protocol

### 4.1 所有可训练 arms 的共同条件

```text
initial LoRA       = released DiffusionOPD Student
new reward         = PickScore（第一阶段）
train prompt split = 完全相同的正式 split
eval prompt split  = 完全相同且不参与训练
sampling seeds     = paired
optimizer budget   = matched steps + 同时报告 wall-clock/GPU-hours
checkpoint steps   = 预先固定
checkpoint type    = raw 与 EMA 明确分开，不混用
```

对 OCR / GenEval constraint，使用各自的正式 prompt distribution；不能再把三个 slot 都指向
PickScore prompts。

### 4.2 两种都要报告的公平口径

#### Equal compute / equal steps

在相同训练步数或 GPU-hours 下比较。这回答工程效率。

#### Matched target gain / Pareto curve

在达到相近 PickScore 增益的 checkpoints 之间比较 OCR/GE retention，或者报告完整：

```text
x-axis = PickScore gain from Student-0
y-axis = OCR / GenEval drop from Student-0
```

这对 constrained method 很重要，因为它可能学习较慢。只看同一 step，可能把“更保守”误读成
“完全无效”；只看任意最佳 checkpoint，又会产生选择偏差。因此两种口径都要有，checkpoint
集合需要预注册。

### 4.3 报告绝对值和 delta

每个 metric 同时报告：

```text
absolute score
delta from Student-0
delta from paired OPSD-direct
seed-level CI / paired statistics
```

只报告 overall average 不够，因为当前最关键的现象就是“一轴提高换三轴下降”。

---

## 5. 当前算力有限时的最小对照集

geometry gate / 20-step prototype 阶段只需要：

1. `Student-0`；
2. `Student→OPSD-direct`；
3. `Student→QP-projection`。

其中 Student-0 只评测，不训练。若 QP 没有正信号，就停止，不需要花算力补 MOPD/primal-dual。

一旦 QP 通过 20-step gate，正式小规模实验再加入：

4. `Student→MOPD-continue`；
5. `Student→fixed-hinge`；
6. `Student→primal-dual`。

---

## 6. 最终论文中建议的表格结构

### Table A — reproduction/context

- official reported DiffusionOPD / DiffusionOPSD numbers（明确 `reported`）；
- our evaluation of released Student；
- our reproduction of relevant direct baseline。

这张表只说明环境与复现可信度。

### Table B — main controlled comparison

- Student-0；
- Student→OPSD-direct；
- Student→MOPD-continue；
- Student→primal-dual；
- Student→Ours。

全部为我们的 runs、同协议、同 evaluator。这张表才支持方法结论。

### Figure — reward/retention frontier

用各预注册 checkpoints 画 PickScore gain 对 OCR/GE degradation。主张应是 Ours 的曲线是否支配
或明显改善 direct/MOPD 的 frontier，而不是挑一个 checkpoint 的 overall score。

---

## 当前决定

1. 共同起点固定为 released DiffusionOPD Student；
2. 最近邻主 baseline 固定为我们自己运行的 `Student→DiffusionOPSD-direct`；
3. MOPD-continue 在 paper-scale comparison 中加入，不替代 direct baseline；
4. 官方数字只作 sanity/reference，不作为显著性结论的数据；
5. 下一轮 geometry probe 通过后，所有正式 arms 都必须用 real task prompts 重新跑。

