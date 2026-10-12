# 0014 — Codex 对 Claude GuardedOPSD 实现与结果的复核

Date: 2026-08-29  
Status: audit complete; next action is a geometry/proxy gate, not another long run  
Jobs submitted by this entry: none

## 结论先行

Claude 的实验给出了一个**可信且重要的负结果**：

> 将 same-state frozen-teacher hinge loss 以固定权重直接加到 OPSD direct loss 上，确实能保护 OCR，
> 但会显著破坏 PickScore、Aesthetic 和 HPS；在 `w=1/3/10` 下都不是 Pareto improvement。

但这**不是** `0006` 所定义的 GuardedOPSD paper method，因此目前不能写成
“feasible-update projection 被证伪”。原因有两类：

1. 方法实现不同：实际是固定权重 penalty，不是 QP projection，也没有 certificate；
2. 正式训练沿用了 smoke prompt 设置：三个 task slot 全部使用同一个 tiny PickScore prompt pool，
   没有在真实 OCR / GenEval prompts 上定义各自 constraint。

正确的下一步不是继续扫 retention weight，也不是再跑 200 epochs，而是先做一个便宜、决定性的：

> **real-task proxy + gradient-geometry probe**，测量真实 OCR/GenEval prompts 上是否存在既提高
> PickScore、又不增加旧能力 constraint 的局部更新方向。

如果这个方向不存在，应当停止 GuardedOPSD；如果存在，才值得实现真正的二维 QP projection。

---

## 1. 我独立确认了哪些结果

### 1.1 作业状态

| arm | job | status | elapsed |
|---|---:|---|---:|
| `gdirect`, w=0 | 1035940 | COMPLETED, exit 0 | 07:33:31 |
| `gguard3`, w=3 | 1035951 | COMPLETED, exit 0 | 07:31:50 |
| `gguard`, w=10 | 1035941 | COMPLETED, exit 0 | 07:33:10 |
| `gw1dir`, w=0 matched control | 1035998 | COMPLETED, exit 0 | 08:09:40 |
| `gw1`, w=1 | 1035999 | COMPLETED, exit 0 | 07:59:35 |

错误的 0011 `per-teacher gather` 作业 `1035990/1035991` 均为 FAILED, exit 1；
其结果不应进入任何方法比较。

### 1.2 完整 w=3 / w=10 配对评测

我重新运行：

```bash
python compare_arms.py gdirect gguard3
python compare_arms.py gdirect gguard
```

共同 checkpoints 为 `{4,8,12,16,25,50,75,100,150,200}`，n=10。
下表使用 `guard - direct` 的直观符号：

| metric | w=3 | w=10 | result |
|---|---:|---:|---|
| OCR | +0.1200 | +0.1188 | guard 显著更好 |
| CLIP | +0.0083 | +0.0090 | guard 显著更好 |
| PickScore | −0.0430 | −0.0477 | guard 显著更差 |
| Aesthetic | −0.7216 | −0.7704 | guard 显著更差 |
| HPS | −0.0272 | −0.0296 | guard 显著更差 |
| GenEval | +0.0203 | +0.0172 | 打平 |

因此 Claude 对该固定 penalty 实验的核心判读是对的：

- constraint 分支真实生效，不是 silent no-op；
- OCR retention 是强、稳定、随时间出现的真实效应；
- 代价也是真实效应；
- w=3 与 w=10 的相近结果说明较强 penalty 已进入饱和区。

w=1 与同硬件控制的 n=6 结果方向相同：OCR +0.0490，但 PickScore −0.0303、
Aesthetic −0.6266、HPS −0.0278。因此没有理由继续在 `w>=1` 内扫权重。

---

## 2. 实际实现与 0006 主方法的差异

### 2.1 实际是固定权重 hinge penalty，不是 projection

补丁的实际更新目标是：

```python
loss = (
    cr_direct_weight * _cr_direct_loss
    + cr_retention_weight * _cr_retention_loss
)
```

其中：

```python
_cr_retention_loss = mean(relu(D_cur - threshold))
```

这属于 scalarized exact/hinge penalty。它没有计算：

- `g_new` 与各 constraint gradient 的 Gram matrix；
- 二维 active-set QP；
- 离原始 OPSD direction 最近的 feasible direction；
- constraint inactive 时严格恢复原始 direct update 的 update-space projection；
- post-optimizer cached-state certificate / backtracking。

因此 0010 中“单边投影”的表述不准确。ReLU 是对**违约标量**做 one-sided gate，
不是把**参数更新向量**投影到 feasible cone。

### 2.2 正式训练复用了明确标成 smoke-only 的 prompt 配置

`0002_constrained_retention.patch` 自己写明：

> use the same tiny prompt set ... The smoke tests optimization plumbing; it does not estimate task performance.

但 0010 的 200-epoch 正式运行仍调用 `codex_cr_smoke()`。作业日志逐项确认：

```text
slot pickscore: prompts=opd_related/testdata/pickscore, prompt_fn=general_ocr
slot ocr:       prompts=opd_related/testdata/pickscore, prompt_fn=general_ocr
slot geneval:   prompts=opd_related/testdata/pickscore, prompt_fn=general_ocr
```

这个 tiny pool 有 256 条训练 prompt。与正式 `mopd()` 配置相比：

- OCR slot 没有使用 `dataset/ocr`；
- GenEval slot 没有使用 `dataset/geneval`；
- GenEval slot 没有使用 `prompt_fn=geneval`；
- 三个 slot 只是换 teacher adapter，没有换 task-appropriate state distribution。

这违反了 0006 §3.1 的核心定义：Pick/OCR/GE 必须使用各自 prompt minibatch。

特别是，teacher-relative KL 是 state-conditional proxy。若 GenEval constraint 从未在
GenEval-style states 上测量，“GenEval 没被守住”不能反证这个 proxy；它首先说明 constraint
被定义在了错误的 state distribution 上。

### 2.3 日志没有真正分开 OCR 与 GenEval constraint

约束计算本身按 `teacher_idx` 路由是正确的，`K_max=1` 也证明每个 slot 只有自己的 teacher。
但以下日志把两个 retention slots 的值重新平均到一起：

```text
cr_current_teacher_kl
cr_reference_teacher_kl
cr_violation
cr_active_frac
```

因此当前记录不能回答：

- OCR constraint 和 GenEval constraint 分别何时激活；
- 哪个 constraint 主导了梯度；
- GenEval 是 proxy 无效，还是它在错误 prompts 上根本没有激活。

0011 的失败来自把 `K_max` 维错误理解为三个 task teachers；但真正需要修的是**按 slot
分别记日志**，不是在 size=1 的 teacher dimension 上 gather 0..2。

### 2.4 这不是 adaptive primal-dual baseline

0006 推荐的低风险 baseline 是 adaptive multipliers。0010 仍是固定 `w=1/3/10`，
没有 dual update。因此它也没有完成 0006 的 baseline A，只完成了 fixed-penalty ablation。

---

## 3. 这批实验究竟证实和证伪了什么

### 已证实

1. released Student 初始化、frozen reference、same-state replay 和 teacher routing 能运行；
2. direct-only 从 released Student 出发能提高 PickScore，但会遗忘 OCR；
3. teacher-relative hinge 对 OCR 具有真实控制力，KL proxy 至少对 OCR 不是完全无关；
4. 固定权重 penalty 会过度改变共享参数更新，保护 OCR 的代价远大于整体收益；
5. 在 `w>=1` 区间继续调固定权重不是有希望的主线。

### 已证伪

1. “fixed penalty 是一个足够好的 GuardedOPSD 近似”；
2. “把 smoke 配置直接延长到 200 epochs 就能代表正式 task-aware training”；
3. “OCR/GE 合并日志足以诊断两个 capability constraints”；
4. 0011 关于 `K_max=3` / teacher dimension 的心智模型。

### 尚未被证伪

1. real OCR / GenEval states 上的 task-specific teacher-relative constraint；
2. asymmetric feasible-update projection；
3. post-step certificate；
4. 是否存在一个保留足够 PickScore 一阶增益的 local feasible cone。

---

## 4. 下一步：先做 real-task proxy + gradient-geometry probe

### 4.1 为什么这是当前最值钱的实验

真正的 QP 实现有工程成本，而 fixed penalty 的失败可能有两种完全不同的解释：

1. **优化器错了**：可行方向存在，但固定 penalty overshoot，把模型硬拉向 OCR teacher；
2. **几何上无解**：PickScore gradient 与 capability-safe cone 基本不相交，任何保护都只能停止学习。

这两种情况不能靠再扫 weight 区分，但一次 gradient-geometry probe 可以直接区分。

### 4.2 探针必须使用的配置

- frozen initial reference：released DiffusionOPD Student LoRA；
- Pick batch：正式 `dataset/pickscore` + 正式 OPSD direct target；
- OCR batch：正式 `dataset/ocr` + `general_ocr`；
- GenEval batch：正式 `dataset/geneval` + `geneval`；
- checkpoints：至少 Student init 与 direct trajectory 的 `{4,8,12,16}`；
- 不执行 optimizer step，只计算和记录梯度几何；
- 每个 checkpoint 至少 8–16 个独立 triplets，不能只看一个 batch。

### 4.3 每个 triplet 记录的量

```text
g       = grad L_OPSD-Pick
a_ocr   = grad C_ocr
a_ge    = grad C_ge

cos(g, a_ocr), cos(g, a_ge), cos(a_ocr, a_ge)
||g||, ||a_ocr||, ||a_ge||
C_ocr, C_ge
constraint active / predicted-to-cross flags
```

然后离线解真正的二维 QP：

```text
min_d  0.5 ||d - g||^2
s.t.   a_m^T d >= b_m
```

并记录：

```text
reward_progress_ratio = <g,d> / ||g||^2
step_norm_ratio       = ||d|| / ||g||
projection_angle      = cos(g,d)
predicted_delta_C_m   = -eta <a_m,d>
active_set / lambdas
```

### 4.4 同时补上 task-specific proxy 检验

在相同 direct checkpoints 上，用 real task prompts 计算：

```text
C_ocr(checkpoint)  vs held-out OCR(checkpoint)
C_ge(checkpoint)   vs held-out GenEval(checkpoint)
```

优先看单调性、Spearman 和差分符号，不在 n=5 上过度依赖 p-value。还应检查 cross-task：

```text
C_ocr 是否比 C_ge 更能解释 OCR 变化；
C_ge 是否比 C_ocr 更能解释 GenEval 变化。
```

### 4.5 预注册 go/no-go

进入真正训练前，至少满足：

1. OCR / GE 两项分别记录，不能再只有聚合 retention KL；
2. real-task constraint 与对应 benchmark 的变化方向大体一致；
3. 在 constraint active / predicted crossing 的 batches 中，QP 后的
   `reward_progress_ratio` 中位数至少约 0.3，且不是大量返回近零更新；
4. QP 数值结果用一个低维 synthetic test 和显式 active-set enumeration 双重对拍；
5. Student init 上 same-state reference 自洽仍为 0。

若第 3 条失败，说明局部可行方向太少：停止 GuardedOPSD，不应再靠调 lambda 延命。

---

## 5. 若 geometry gate 通过，才实现真正的 20-step paired run

只跑三个 arms：

| arm | purpose |
|---|---|
| direct-only | reward / forgetting control |
| fixed hinge w=1 | 已知负面的 scalarization baseline |
| QP projection | 0006 真正主方法 |

要求：

- 三臂使用相同 real task datasets、seed、batch schedule；
- 每个 optimizer step 前分别累计 `g/a_ocr/a_ge`；
- 解二维 QP 后写回 projected gradient；
- 记录 active set、lambda、progress ratio 和 per-task violation；
- 第一个版本先做 raw-gradient projection，并清楚标注 AdamW 会改变 update geometry；
- 只有 raw-gradient 版本展示正信号后，再增加 optimizer-step certificate/backtracking；
- 20 steps 后立即评估，不直接延长到 200 epochs。

成功信号应是：相对 direct-only，OCR/GE violation 明显下降，同时 PickScore direct gain
保留至少约 50%–80%；不像 fixed penalty 那样把 PickScore 压到 released Student 以下。

---

## 6. 若 geometry gate 失败，应该怎样 pivot

若 `reward_progress_ratio` 长期接近 0，说明 shared LoRA 的局部参数空间里 Pick improvement
和 capability retention 确实冲突。此时更合理的下一篇方法不是再换一种权重，而是增加容量隔离：

- 冻结原 Student LoRA；
- 为新 reward 增加独立 residual LoRA；
- 用 old-task Jacobian null-space / orthogonal subspace 限制新 LoRA；
- 或使用 prompt-conditioned adapter routing，再研究能否无人工 task label 地路由。

这会把研究问题从“优化权重”转为“continual diffusion alignment 的容量分配”，
但属于明确的方向切换，必须由 geometry gate 的负结果来触发。

---

## 7. 当前操作决定

- 不再提交 `w=0.1/0.3/...` 的固定-weight 长训练；
- 不把 0010 写成 GuardedOPSD 主方法失败；
- 不使用 0011 的代码或失败作业；
- 保留 0010 作为论文中很有价值的 fixed-penalty negative baseline；
- 下一份代码只做 real-task proxy / gradient-geometry probe；
- 该 probe 通过前，不提交新的正式训练。

