# 0027 RG-OPSD：把“选择”从 target/trajectory space 后移到真实 optimizer update 之后

**日期：2026-08-30**  
**状态：方法建议与分阶段预注册；本条没有额外提交训练任务。**  
**建议名称：Realization-Gated OPSD（RG-OPSD）；备选标题：Post-Fit OPSD。**

## 0. 结论先行

现在最有希望得到正向结果、同时保持 paper novelty 清楚的方向，不是再发明一个更复杂的
positive/negative target，也不是立刻把 Self-OPD 的 `K=8` branches 搬进来，而是：

> **先让若干 proposal 从完全相同的参数与 AdamW state 出发，各自完成一次真实 finite fitting；
> 再在同一组缓存状态和 common random numbers 上，按“新 reward 的实际改善 + 旧能力/通用质量
> 的实际约束”选择一个候选，最后只提交被选中的参数和 optimizer state。**

DiffusionOPSD 在 **target construction 前后**区分了 `G_construct` 与 `G_realized`，但仍在
真实 AdamW 更新之前决定要拟合哪个 target。Self-OPD 在 **transition branch rollout 后**选
高/低 reward branches，但也仍在真实 parameter fitting 之前决定 supervision。两者共同留下的
空位是：**没有根据 finite optimizer realization 选择最终更新**。

我们的数据恰好说明这个空位重要：

- DiffusionOPSD 自己的一步拟合实验已经出现 target 排名与 realized ranking 反转；
- 我们测得实际 AdamW 参数更新与累积 raw gradient 的 cosine 只有 `0.339`；
- `w=0` 与 `w=10` 产生很清楚的互补：`w=10` 大幅守住 OCR/GenEval，却付出 Pick/AES 代价；
- `w=3` 没有改善 Pareto，说明继续静态扫 scalar weight 的收益很低；
- 相对初始 Student，当前最好 guarded checkpoint 仍有明显 HPS/通用质量退化，说明只约束
  OCR/GenEval teacher KL 不足以覆盖 broad retention。

因此 RG-OPSD 不赌“某个固定 loss weight 永远正确”，而是学习**当前这一步到底该走 direct、
guarded、quality-aware，还是不走**。其最小版本只需要现成代码，不依赖尚未完整公开的
Self-OPD mixed implementation。

---

## 1. 两篇方法的精确对照，以及真正可做的新东西

### 1.1 DiffusionOPSD：强 target construction，薄弱点是 finite realization

在行为策略给出的低噪声 query `s=(c,z_q,σ_q)` 上，DiffusionOPSD 先构造 clean anchor

```text
y0 = zq - σq · v_old(s)
```

再利用可微 reward 对 `y0` 做有界、归一化的 ascent/descent，得到 detached `y+` 与 `y-`。
endpoint reward 决定正负分支权重，trainable policy 用有限次 AdamW 拟合 target，之后才刷新
behavior EMA。

这个设计的优势已经被论文的 ablation 支持：reward-gradient target direction 的 construction
gain 与局部 reward alignment 都明显优于 residual/random target。其关键诊断则是

```text
G_realized = G_construct - G_fit
```

即 target 是好的，并不推出有限拟合后的模型真的按同一方向改善；论文的一步 AdamW 实验甚至
观察到 target ranking reversal。因此当前最稳的选择是**保留 canonical target constructor**，
直接处理 `G_fit`，而不是同时替换 target 和 optimizer。

Primary sources：<https://arxiv.org/abs/2608.24646>，
<https://github.com/worldbench/DiffusionOPSD>。

### 1.2 Self-OPD：强 self-exploration，薄弱点仍在 branch-to-parameter realization

Self-OPD 在当前 deterministic transition 周围采样 `K=8` 个 SDE branches；每个 branch 用
当前 student 的 ODE sampler rollout 到 endpoint，并与 deterministic ODE self-reference 比较：

```text
A_k = (r_k - r_ode) / std(r_1,...,r_K)
```

正 advantage branch pull，负 advantage branch push，另有 direction-aware attenuation 与
transition-variance normalization。它的关键贡献是把 **student 自己探索出的 transition**变成
step-wise supervision，并可在 reward level 合成 black-box scores。

但它选择的是“哪个随机 branch 适合作为 supervision”，不是“拟合后哪个 parameter update
真的最好”。若 AdamW 有旋转、under-realization 或 overshoot，branch endpoint reward 仍不能保证
拟合后的 policy reward。multi-reward branch score 本质上也还是有限候选上的 scalar compromise，
不是相对初始 Student 的 capability budget。

Primary sources：<https://arxiv.org/abs/2608.26872>，
<https://github.com/Shiy-Zhang/Self-OPD>。截至本记录，官方 mixed reward config/checkpoint 仍未
完整公开，细节见 `0024_SELF_OPD_LITERATURE_IMPACT.md`。

### 1.3 我们应借什么、不要急着借什么

应借：

- DiffusionOPSD 的 on-policy query、强 reward-gradient target 和 construction/realization 分解；
- Self-OPD 的 “branch → 用实际 reward 与 self-reference 比较 → 只接受有利 branch” 思路；
- Self-OPD 对 black-box reward 的兼容性，但只把 black-box scorer 用于 **post-fit selection**，
  不要求它提供梯度。

暂时不借：

- 不先引入 Self-OPD 的 `K=8` transition branches；否则 target、rollout compute、optimizer guard
  三个变量同时改变，正结果也无法归因；
- 不再扫更多固定 hinge weights；`w=3` 已表明简单中间权重不自动改善 Pareto；
- 不把尚未通过真实 same-state probe 的线性 QP 当成主方法必要组件。

---

## 2. RG-OPSD 的具体算法

### 2.1 两类 batch，避免把训练拟合误当成泛化

每个 outer iteration 从 frozen behavior `θ_b` 收集两组不重叠 prompt：

1. `B_fit`：构造 DiffusionOPSD targets、计算 candidate gradients；
2. `B_gate`：不参与 candidate gradient，只用于 post-fit selection。

两组都缓存 prompt、query timestep、latent、teacher/reference outputs 和 suffix RNG。所有候选共享
完全相同的 `B_fit/B_gate` 及 RNG。Phase-1 诊断可以先在 same-state/same-window 上做；进入短训练
后必须使用 disjoint rotating gate buffer，否则 selection score 会偏乐观。

### 2.2 capability constraint

对 OCR、GenEval 等旧能力，继续采用当前已经验证过定义：

```text
C_m(θ; B) = D_m(θ,T_m;B)
            - (1+τ_rel) D_m(θ_init,T_m;B) - τ_abs
```

其中 `θ_init` 是发布的 DiffusionOPD Student，`T_m` 是对应 frozen specialist teacher。为了让
identity candidate 永远可行，并在模型已经越界时至少不继续恶化，iteration `t` 的 gate 是

```text
C_m(θ_candidate; B_gate) ≤ max(C_m(θ_t; B_gate), 0) + ε_cert .
```

- 当前仍在 budget 内时，它要求 candidate 也在 budget 内；
- 当前已经越界时，它要求不进一步增加 violation；
- 它不伪称可以一步修复历史累计遗忘。

### 2.3 broad-quality constraint：直接对准我们实际掉的 HPS/AES

仅有 OCR/GenEval teacher KL 无法解释或阻止 held-out HPS/AES 退化。RG-OPSD 的 gate 不需要
scorer 可微，因此可以在一个与最终 test prompts 严格隔离的 general-prompt calibration buffer
上加入：

```text
C_hps(θ) = HPS(θ_init; B_general) - HPS(θ; B_general) - τ_hps
C_aes(θ) = AES(θ_init; B_general) - AES(θ; B_general) - τ_aes
```

用 fixed suffix 和 common seeds 比较 candidate 与 current policy。`B_general` 只能来自训练用
prompt pool，不能使用最终 DrawBench/Pick-a-Pic test prompts。第一版只加 HPS，AES 作为报告轴；
若两个 noisy gates 同时上，很容易让每步都选 identity。

这一步是与 Self-OPD 最有价值的结合：**black-box scorer 参与实际候选选择，但不把多个 reward
gradient 生硬相加，也不要求额外 specialist teacher。**

### 2.4 candidate portfolio：从最小三臂开始

所有 candidate 都从完全相同的 `(θ_t, AdamW_state_t, LR_state_t)` 出发，使用相同 `B_fit`、
gradient accumulation、clip、mixed precision 和 RNG。

#### Phase-2a 最小三臂（先验证核心机制）

```text
A0: identity / no-op
AD: canonical DiffusionOPSD direct，走一次真实 AdamW
AG: 当前已验证的 w=10 OCR/GenEval guarded loss，走一次真实 AdamW
```

这是最干净的首个实验：`AD` 是最高新 reward 进度候选，`AG` 是目前实验证明能保旧能力的候选，
`A0` 保证在两者都不安全时不制造新损失。它不需要任何新 loss，也不依赖 QP。

#### Phase-2b 只在三臂诊断后加两个候选

如果 `AD` 经常跨界而 `AG` 经常安全但代价大，加入：

```text
AL: look-ahead guarded candidate
    先用 AD 的 exact post-fit result 找到 at-risk constraints；
    回到原始 θ/optimizer state，对这些 m 使用 raw C_m（不是当前点可能为 0 的 ReLU）
    生成一次真实 AdamW candidate。

AQ: quality-aware candidate
    用公开 DiffusionOPSD mixed Pick+HPS target，再加同一 OCR/GE guard，
    生成一次真实 AdamW candidate。
```

`AL` 解决一个明确问题：pre-step hinge 可能尚未 active，但 direct step 会跨过边界；此时普通
`w=10` 与 direct 可能完全相同。`AQ` 则给 selector 一个能主动修正 HPS 的方向，避免只有
identity 才能通过 HPS gate。

若 `0025/0026` 的真实 Phase-1 projection gate 通过，可以再加入 `AP`（optimizer-delta projected
candidate），但它是 portfolio 中的可替换 arm，不是 RG-OPSD 成立的前提。提交 `AP` 时必须明确
optimizer moments 如何同步；不能只改 parameters、却静默沿用与 projected delta 不一致的 state。

Self-OPD proposal 也采用同一规则：只有在公开代码 fidelity smoke 与 20-step single-reward gate
证明它优于 `AD` 后，才把它加为 `AS` candidate。这样新方法是 proposal-agnostic，而不是被某篇
不完整 release 绑住。

### 2.5 post-fit exact selection

对每个 candidate `a`：

1. 完成它自己的真实 AdamW finite update，得到 coherent `(θ_t^a, state_t^a)`；
2. 在 `torch.inference_mode()` 下，用相同 `B_gate` 精确重算 OCR/GE constraints；
3. 用相同 suffix seeds 生成/解码，计算实际 Pick gain 与 HPS gate；
4. 多卡 all-reduce 后统一选择，不能各 rank 自选 candidate。

定义新目标的 realized gain：

```text
G_a = J_pick(θ_t^a; B_gate, ξ) - J_pick(θ_t; B_gate, ξ)
```

其中 `ξ` 是对所有候选相同的 suffix randomness。选择规则：

```text
feasible(a) := 所有 capability / quality gates 均通过

a* = argmax_a LCB(G_a)
     subject to feasible(a) and LCB(G_a) ≥ δ_gain
```

若没有 non-identity candidate 通过，就选 `A0`。初期可把 `LCB` 简化为 paired sample mean 加一个
冻结的小 margin；短训练后再依据 gate-buffer paired variance 设置信赖界。禁止根据最终 test
metric 反调 margin。

最后只提交 `(θ_t^{a*}, state_t^{a*})`，再更新 behavior EMA。若选 `A0`，parameter、optimizer
moments、scheduler step 和 EMA 都不前进；只推进数据 outer iteration。

---

## 3. 它能保证什么，不能保证什么

### 3.1 一个简单但真实的 one-step certificate

由于 candidate set 包含 identity，且 gate threshold 使用
`max(C_m(θ_t),0)+ε_cert`，在数值容差内总有一个可行候选。若 selection 只接受
`LCB(G_a)≥δ_gain≥0` 的非 identity 候选，则在固定 `B_gate` 与 RNG 上：

1. 新 Pick realized gain 不小于 0；
2. 当前在 budget 内的旧能力不会越过 budget；
3. 当前已越界的旧能力不会继续增加 violation；
4. 如果某个固定 candidate（例如 `AG`）可行，selector 在该 gate batch 上的 Pick gain 不会低于
   它。

这不是复杂的 convergence theorem，但它比在 raw-gradient space 里的预测更贴合我们的实测问题：
certificate 是在 **真实 AdamW、gradient clipping、weight decay 和有限拟合之后**给出的。

### 3.2 必须诚实写出的边界

- 这是 cached calibration-state certificate，不是 population/global guarantee；
- 多步训练分布会变化，所以 `B_gate` 必须轮换，且 paper 结果必须用从未参与 gate 的 final test；
- noisy scorer 上的单 batch improvement 可能不泛化，因此需要 paired CRN、margin 与独立 seed；
- identity 能保证安全，但不能保证训练一定有进展；identity rate 本身是重要 failure signal；
- 若所有非 identity proposals 都同时伤害 Pick 与 HPS，selector 不能凭空发明新方向，必须增加
  `AQ` 或经过验证的 Self-OPD proposal；
- 多 candidate training 比 direct DiffusionOPSD 贵，必须报告 wall-clock/GPU-hours，不能只按
  optimizer step 比较。

---

## 4. 为什么它比当前几个备选更可能得到正结果

### 对比“继续调 fixed hinge weight”

`w=3` 与 `w=10` 在 OCR/GE 收益上不可分辨，但 `w=3` 的 AES 代价反而更大，`w=10` 在当前
trajectory Pareto 上更好。这说明 loss scale 与最终 Pareto 不是单调关系。RG-OPSD 直接按 post-fit
结果选 update，不依赖“中间权重应该产生中间效果”的假设。

### 对比“立刻换成 Self-OPD”

Self-OPD 的 branch target 很有希望，但官方 mixed code/checkpoint 目前不完整，而且同时引入
`K=8` rollouts、branch sampling 与新 target 会改变太多因素。DiffusionOPSD target direction 已有
强证据；先修 realization 是风险更低的因果干预。Self-OPD 以后是一个 proposal arm，而不是
整个方法成败的前提。

### 对比“只做 QP projection”

projection 依赖局部 constraint geometry，并且当前真实 probe 连续遇到 GradScaler/OOM 工程问题，
尚无真实 diffusion-state pass 结果。RG-OPSD 的最终 gate 不依赖线性近似：QP 预测错了就不选
`AP`，direct 或 guarded candidate 仍可被选。因此它能吸收 QP 的潜在收益，却不把风险全押在 QP。

### 对比“继续优化 target construction”

DiffusionOPSD 已经显示 reward-gradient target construction 很强，且论文明确观察到 construction
gain 不总能变成 realized gain。此时再花主要精力换 target 是绕开最直接的 failure mode。RG-OPSD
正面利用其 `G_construct-G_fit` 分解，并用真实 `G_realized` 作决策变量。

---

## 5. 实验顺序与冻结的 go/no-go

### Phase 0：先收完正在跑的外部对照，不再重复提交

截至本记录：

- naive KL baseline `1036782 (crkl5)` 正在运行；
- own-run MOPD continuation `1036868 (mopdcont)` 正在运行；
- same-state Phase-1 `1036866` 因 eval graph 累积在 H200 上 OOM；Claude 已按“eval backward 释放
  graph”修补，并提交 `1036869`，当前等待资源。

本条不再提交重叠任务。先让 `1036869` 给出真实 active rate/certificate；该结果只决定是否把
`AP` 加入 portfolio，不决定 RG-OPSD 是否继续。

### Phase 1R：离线三候选 counterfactual selector probe（24 windows）

在与 `0025` 相同的 24 个 frozen windows 上，从同一 `(θ,opt_state)` 顺序生成 `A0/AD/AG`，
每个 microbatch 及时 backward/free graph，gate 全部 `inference_mode`；禁止同时保存多份计算图。

先于结果冻结：

| 判据 | go/no-go |
|---|---|
| `AD` 与 `AG` 在 ≥20% windows 上选择结果不同 | selector 有实际决策空间；否则关闭三臂版本 |
| ≥70% windows 选择 non-identity | 有足够训练进度；低于此值先加 `AL/AQ`，不跑长训 |
| protected constraints exact pass ≥90% | 进入 30-step；否则检查 gate variance/候选覆盖，不宣称安全 |
| selected median Pick gain ≥ `AG`，且 ≥50% 的 `AD` median gain | 最低效用门槛；否则 fixed `w=10` 更合理 |

这里的 checkpoint/window 是诊断样本，不当成独立训练 seed，也不报虚假的 t-test。

### Phase 2：30-step、三臂独立训练

```text
arm 1: AD / w=0 direct
arm 2: AG / frozen w=10 fixed hinge
arm 3: RG-OPSD selector (A0, AD, AG；必要时按 Phase 1R 预先决定加 AL/AQ)
```

相同 Student init、prompt stream、seed、optimizer-update opportunity；另外完整报告 GPU-hours。
主要成功门槛：

1. RG 的 OCR/GenEval 不差于 `AG` 超过各自独立重测 `2σ`；
2. 相对 `AG`，至少收回 `w=0` 与 `w=10` 之间 Pick gap 的 25%。当前 gap 约 `0.0143`，即先看
   是否有至少约 `+0.0036` 的 Pick 改善；
3. HPS 不得比 `AG` 进一步下降；若仍明显低于 Student，则在 paper-scale 前加入 `AQ + HPS gate`；
4. identity rate ≤30%，否则方法主要是在停止训练，不是更好的 update；
5. 不能用同一 trajectory 的多个 checkpoint 冒充 independent `n`。

### Phase 3：paper-scale

- 固定一个 checkpoint/训练长度，至少 3 个 independent training seeds；
- own-run baselines：Student、DiffusionOPSD-direct、MOPD-continue、naive KL、fixed `w=10`；
- Self-OPD 官方 mixed release 完整后 own-run；在此前只能标注 faithful reimplementation；
- 同时报告 task-image Pick/HPS 与 held-out general-prompt Pick/HPS/AES；
- matched protocol，不用论文表里的官方点估计替代自己 run 的 baseline；
- 报 score、置信区间、wall-clock、GPU-hours、identity/arm selection frequency。

---

## 6. 实现要点（给下一位实现者/复核者）

1. 只 snapshot LoRA trainable parameters 与 optimizer/scheduler state；base SD3.5 weights 共享。
2. candidate 顺序执行，每个 candidate 前恢复同一 snapshot 与 RNG；每个 microbatch 都 backward
   并释放图，不能重演 `1036866` 的 12×10 graph 累积。
3. gate 必须 `torch.inference_mode()`，reward gradient target construction 才允许建 reward graph。
4. 每个 candidate 保存 coherent optimizer state；选 identity 时也必须还原 moments/step/scheduler。
5. 若加入 parameter-space `AP`，明确 moments policy并做 ablation：
   `direct moments + projected params` 不是自动正确。
6. 所有 ranks all-reduce candidate metrics，rank 0 选择后 broadcast candidate id。
7. log 至少包含：每个 candidate 的 exact Pick gain、每个 `C_m before/after`、HPS change、feasible、
   chosen id、identity rate、额外 forwards/backwards、wall-clock。
8. gate prompt pool 与最终 test set 做 hash-level disjointness 检查；不能用 DrawBench test 做训练选择。

---

## 7. 最终 paper framing

最稳的 claim 不是“又一种 teacher-free OPD”，而是：

> **Proposal-agnostic post-fit self-distillation for continual alignment：将 DiffusionOPSD/Self-OPD
> 产生的 supervision proposal 经过真实 optimizer realization 后再选择，并以 teacher-relative
> capability budgets 与 black-box quality gates 对每次提交更新做 cached-state certificate。**

一句话区分：

```text
DiffusionOPSD branches in target space.
Self-OPD branches in transition space.
RG-OPSD branches and selects in optimizer-update space, after finite realization.
```

这条差异足够直观，也正好对应论文理论和我们目前最清楚的实验 failure mode。如果最小三臂
Phase 1R 都没有选择空间或经常只能 no-op，应尽早判负；如果能在守住 OCR/GE 的同时收回至少
25% Pick gap，再加入 HPS gate 后形成正向 Pareto，才值得 paper-scale。
