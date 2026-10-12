# 0036 从 DiffusionOPSD、Self-OPD 与 LLM OPSD 得出的统一结论与新方向

**日期：2026-08-30**  
**建议方向：Reliable Pareto Improvement OPSD（RPI-OPSD）**  
**状态：方法判断与 go/no-go；本条未提交新任务。**

## 0. 我还对 continuation 有信心吗？

需要分成两个答案：

- 对“继续用现有 direct/固定 penalty 训练，最终自然全面超过 DiffusionOPD Student”：**信心低**。
  现有五臂已表明，不加新机制只会落在“不动”或“新 reward 上升但 broad capability 下降”。
- 对“把 continuation 改成 selective、reliable、Pareto-feasible improvement，得到一个可发表的
  窄但真实 Pareto 改善”：**有中等信心，值得一次严格 gate**。

这里的信心不是因为 OPSD 这个名字听起来合理，而是因为四类证据已经拼成同一个 failure mode：

1. direct continuation 能强烈提高 Pick，证明 current Student 周围存在 reward-improving directions；
2. hinge 能守住 OCR/GE，证明保护旧能力并非不可能；
3. direct、hinge、naive KL 都掉 held-out HPS，说明当前 proposal set 缺少 broad-quality-safe direction；
4. LLM OPSD 的最新工作也发现：未经筛选的 dense self-teacher signal 会被 shortcut/outlier 主导，
   真正有效的是只蒸馏可靠且可迁移的局部纠正。

因此应停止问“哪个全局 weight 最好”，改问：

> **在当前 on-policy state 周围，是否存在一个经实际 rollout 验证的新 reward 改善、同时没有损害
> quality/capability 的局部 branch？只有存在时才训练；拟合后仍满足才提交。**

---

## 1. DiffusionOPSD 是如何让 OPSD work 的

DiffusionOPSD 的核心不是“用了 reward gradient”这么简单，而是一套 local policy-improvement →
dense regression 的转换：

1. **On-policy query**：frozen behavior rollout，监督建在当前策略真的会访问的低噪声 state，
   避免拿离线/forward-noised state 训练造成 distribution mismatch。
2. **Self anchor**：`y0=zq-σq v_old(s)`，所有修改都相对当前 behavior clean prediction。
3. **Local improvement oracle**：对可微 reward 做 normalized ascent/descent，并限制在
   `ρ||y0||` trust ball；因此 target 是局部、有限幅度的，而不是自由对抗 reward model。
4. **Dense detached target**：把 endpoint reward gradient 转成 intermediate clean-output target；
   reward/decoder graph 随后丢掉，训练变成稳定监督学习。
5. **Finite fitting + slow refresh**：只做有限 AdamW fitting，再更新 behavior EMA；每轮重新生成
   query/anchor/target，防止长期追逐陈旧 target。

它真正的理论贡献也指出自身缺口：

```text
G_realized = G_construct - G_fit
```

target construction gain 不等于 optimizer 后的 realized gain；论文自己观察到一步拟合的 ranking
reversal。因此它“work”的充分条件不只是 target 好，还要求 finite fitting 没有旋转/underfit/overshoot。

更重要的是，论文的主结果主要是 **reward-specific specialist**：每一列由针对该 reward 训练的
checkpoint 填入。“work”定义是 matched reward 提高，不要求同一模型同时保留 OCR/GE/HPS。

Primary sources：<https://arxiv.org/abs/2608.24646>，
<https://github.com/worldbench/DiffusionOPSD>。

---

## 2. Self-OPD 是如何让 OPSD work 的

Self-OPD 把 local improvement oracle 从“reward gradient”换成“随机探索 + rollout 比较”：

1. 当前 student 在 state `x_t` 给出 deterministic transition mean；
2. 围绕它采样 `K=8` 个轻量 SDE next-state branches；
3. 每个 branch 用 deterministic ODE rollout 到图像；
4. 同一 parent state 的 deterministic ODE rollout 是 self-reference/control variate；
5. `A_k=(r_k-r_ode)/(std(r_1...r_K)+eps)` 给出局部相对优势；
6. positive branches pull，negative branches 经 direction-aware attenuation 后 push；
7. transition-variance normalization 把不同 timestep 的回归统一到 per-step KL 尺度；
8. multi-reward 时先对每个 scorer z-normalize，再在 trajectory level 排序，并声称有 per-scorer
   veto；reward 不反传，所以可用 black-box scorer。

它“work”的关键不是 teacher-free 本身，而是：**候选 transition 是 student 当前分布附近真实可达
的具体样本，endpoint reward 已经验证过该样本，最后只回归一个具体 velocity target。**

但它仍有三个边界：

- `Σ λ_m z_m` 仍是有限 candidate set 上的 scalar ranking，不是真正 Pareto guarantee；
- branch endpoint 好，不保证 AdamW fitting 后的 policy 仍好；
- mixed Table 2 的 held-out Pick/HPS 仍低于 DiffusionOPD，说明 same-task-image reward fusion 没有
  自动解决 broad distribution retention。

Primary sources：<https://arxiv.org/abs/2608.26872>，
<https://github.com/Shiy-Zhang/Self-OPD>。

---

## 3. 两篇 diffusion/flow 方法的统一成功条件

它们表面上一个用 gradient、一个用 branches，本质上共享六条条件：

| 条件 | DiffusionOPSD | Self-OPD |
|---|---|---|
| 数据必须 on-policy | behavior query states | student parent states |
| 必须有当前 self-reference | clean anchor `y0` | deterministic ODE reward |
| sparse reward 要变成 dense target | reward-gradient clean target | branch velocity target |
| improvement 必须 local | bounded trust ball | local SDE neighborhood |
| target 要 detached/稳定拟合 | finite supervised fitting | transition regression |
| target 必须随策略刷新 | behavior EMA outer loop | current-policy branches |

再抽象一步：

```text
sample where the model currently fails
→ produce a local alternative
→ verify it is better than the model's own default
→ distill the verified correction densely
→ move conservatively and regenerate
```

所以 OPSD 并不是“自己教自己就会变强”。它依赖一个**非平凡 improvement operator**：

- DiffusionOPSD 的 improvement information 来自外部 differentiable reward；
- Self-OPD 的 improvement information 来自 stochastic search + external reward ranking；
- 若 reward 不完整、候选没有更好样本、或者 fitting 未实现 target，self-distillation 只会放大偏差。

---

## 4. 我们的 continuation 为什么没有按期待 work

### 4.1 起点已经很强，能移动的余量比从-base specialist 小

DiffusionOPSD/Self-OPD 原论文主要从 SD3.5-M base 开始，目标轴上有很大 headroom。我们的起点是
已融合 OCR/GE/preference 的 DiffusionOPD Student。提高一个轴更容易穿过已经形成的多能力
Pareto boundary，因此同样 reward update 的 opportunity cost 更高。

### 4.2 我们一直在过滤 parameter damage，没有先过滤 bad targets

Pick-only target 只证明“更像 PickScore 喜欢的图”，不证明：

- exact text/content 仍正确；
- strict composition 仍正确；
- HPS/CLIP 的 general preference 不下降；
- 该方向在其他 prompt distribution 上没有 parameter interference。

当前 hinge 是 target 已经构造后，在 parameter loss 上补救 OCR/GE。它可以阻止部分旧能力漂移，
但不能把一个 Pick-shortcut target 变成 general-quality-safe target。

### 4.3 我们的实测正好是 shortcut learning 签名

- Student continuation：Pick 大升，OCR/strict GE/HPS/CLIP 下降；
- from-base OPSD Pick specialist：OCR 下降、strict GE 下降而 loose accuracy 上升；
- 同一 task images 可能更讨 preference scorer，但 held-out general prompts 的 HPS 明显下降。

准确解释不是“continuation 不学习”，而是**它学习了 evaluator 最容易奖励、却不够 transferable 的
局部特征**。

### 4.4 optimizer realization 是第二个独立缺口

我们测得 actual AdamW update 与累积 raw gradient cosine 只有 `0.339`。真实 same-state QP probe
又显示一阶 constraint change 只解释 exact change 的约 `5%~12%`，QP 0/8 真正改动方向；反而
post-fit exact check + backtracking 8/8 能找到安全步长。

所以即使 target-level branch 是好的，也仍需要 post-fit commit gate。

---

## 5. LLM OPSD 给出的直接启发

### GKD：只在 student 自己生成的错误上给 dense feedback

Generalized Knowledge Distillation 用 student-generated sequences 缩小 train/inference mismatch；
核心是 teacher 在 student 真会犯错的 prefix 上监督，而不是离线复述 teacher 数据。

Source：<https://arxiv.org/abs/2306.13649>。

### Self-Distilled Reasoner / OPSD：privileged view 必须提供真正额外信息

同一 LLM 以 question-only student 和 question+verified-solution teacher 两种 context 工作，沿 student
rollout 做 per-token divergence。成功来自 privileged context，不是参数相同本身。其代码后续还加入
per-token KL clipping，因为 style tokens 的 KL 可比 math tokens 大 6–15 倍并主导训练。

Source：<https://arxiv.org/abs/2601.18734>，
<https://github.com/siyan-zhao/OPSD>。

### TrOPD：不可靠区域不要硬蒸馏

TrOPD 根据 teacher/student decoding agreement 定 trust region；可靠 token 用 reverse KL，outlier
mask/clip 或改用 top-k forward KL，并用 teacher-prefix off-policy guidance 把 student 带回可监督
区域。其结论是简单 full-signal OPD 会被大 mismatch 的 policy-gradient outlier 破坏。

Source：<https://arxiv.org/abs/2606.01249>。

### U-OPSD：监督集应当小而集中在“confidently wrong”

U-OPSD 对每个问题采多个 rollout，用 majority/self-consistency threshold 构造 pseudo solution，只在
与 consensus 不一致的 completions 上做 dense distillation。论文实测真正被蒸馏的集合不到 valid
rollouts 的 10%，即成功不依赖每个样本都有 loss，而依赖高置信度纠错。

Source：<https://arxiv.org/abs/2608.06296>。

### Purified OPSD：raw teacher signal 里可能主要是不可迁移 shortcut

Purified OPSD 发现 privileged reference 诱导的 rote-memorization component 在方向和幅度上支配
总更新，而真正 question-conditioned transferable residual 近乎正交甚至反向。它构造 reference-only
teacher，减去该 component，再把 residual 变成 PMI target，才保住 long-CoT 的自然 reasoning。

Source：<https://arxiv.org/abs/2607.02234>。

### OPSDL：用模型自身的 clean/strong view 监督 corrupted/weak view

OPSDL 用 relevant short context 下的强能力监督 full noisy long context 下的生成，说明 self-teacher
应来自模型已知更可靠的“clean view”，而不是让当前模型无条件追随自己的所有输出。

Source：<https://arxiv.org/abs/2604.17535>。

### 对 diffusion 的统一翻译

```text
LLM token trust region        → diffusion branch feasibility
LLM self-consistency vote     → branch score/ranking confidence
LLM privileged clean view     → frozen Student / capability teachers / self-reference
LLM shortcut purification     → reject reward-only branches that fail quality scorers
LLM token-level clipping      → target/advantage clipping and no negative repulsion outliers
```

---

## 6. 建议的新方法：Reliable Pareto Improvement OPSD（RPI-OPSD）

### 6.1 两层结构

```text
Layer A: target/transition level
        只构造并蒸馏 verified Pareto-feasible local correction

Layer B: optimizer/update level
        finite fitting 后在 disjoint capability buffers 上验收，再决定是否 commit
```

Layer A 解决 reward shortcut/bad target，Layer B 解决 AdamW realization 与跨 prompt parameter
interference。只做其中一层都不足：RG-OPSD 若只有 direct/guard/no-op 而所有非零 candidate 都掉
HPS，会一直 no-op；Self-OPD 若只挑 endpoint branch，又不能保证拟合后仍安全。

### 6.2 proposal set：gradient exploitation + stochastic exploration

在 frozen behavior 的 on-policy state `s` 上，构造：

```text
k=0: deterministic anchor / no-op
k=1: canonical DiffusionOPSD normalized reward-gradient proposal
k=2...K+1: Self-OPD-style local SDE branches (首个 probe K=4，过门后再 K=8)
```

这样做的理由：

- reward-gradient proposal 提供低方差、强 exploitation；
- stochastic branches 提供不同方向，增加找到“Pick 改善但 HPS/CLIP 不掉”的机会；
- anchor 使每个 state 都有安全 reference。

### 6.3 vector advantage，不做 weighted scalar ranking

所有 proposal 从同一 parent state 用 common suffix rollout，得到 scorer vector：

```text
A_pick(k) = normalized [Pick(k) - Pick(anchor)]
A_hps(k)  = normalized [HPS(k)  - HPS(anchor)]
A_clip(k) = normalized [CLIP(k) - CLIP(anchor)]
```

对 OCR/GE prompt 相应加入 `A_ocr/A_ge`。定义 hard feasible set：

```text
F(s) = {k:
          LCB(A_pick(k)) ≥ δ_pick,
          LCB(A_hps(k))  ≥ -τ_hps,
          LCB(A_clip(k)) ≥ -τ_clip,
          applicable task scorers ≥ their budgets }
```

`LCB` 初版用 paired common-random-number repeats 的 mean−margin，不能拿 final test set 校准。
与 Self-OPD 的 `Σ λz` 相比，这里新 reward 是 objective，其他 scorer 是 explicit constraints；一个
HPS 大跌的 branch 不能靠 Pick 大涨抵消后入选。

若 `F(s)` 为空，**这个 state 不产生 loss**。否则取 feasible 中 Pick 最大者。接受率与 skip rate
是方法的核心统计量，不是故障。

### 6.4 第一版只 pull，不做 negative push

对选中的 `k*` 做 transition/velocity regression：

```text
L_state = ||v_theta(s) - stopgrad(v_k*)||² / transition_variance
```

第一版不对 negative branches 施加负 MSE repulsion：

- DiffusionOPSD 小型 ablation 中去掉 negative branch 并未变差；
- Self-OPD 需要 direction attenuation 才避免 negative repulsion 抵消 best branch；
- continuation 起点已经很强，repel 一个 scorer 不喜欢的方向更可能误伤其他能力；
- LLM TrOPD/U-OPSD 的共同教训是 outlier/unreliable signal 应 mask，而不是强推。

如果 positive-only 通过，再把 bounded negative 作为单独 ablation，不能一开始混入。

### 6.5 protected/general replay：clean view 只在 at-risk region 发挥作用

每轮另取与 final tests hash-disjoint 的三个小 buffer：OCR、GE、general prompts。frozen released
Student/teachers 是 clean privileged view。只在 current policy 相对 clean view 超出容差时产生
mean-matching target，不做 naive always-on KL。

这相当于把 OPSDL/TrOPD 的“reliable clean view + trust region”搬到 diffusion：

- budget 内不干预，让新 reward 有空间；
- budget 外提供 dense recovery target；
- general prompt buffer 直接覆盖当前 hinge 没保护的 HPS/CLIP 分布。

### 6.6 post-fit exact commit gate

从同一 `(θ_t, AdamW_state_t)` 生成一个 RPI candidate，完成真实 clipping/AdamW 后，在未参与
gradient 的 gate buffers 上检查：

```text
realized Pick gain > δ_commit
OCR/GE violation 不增加
general HPS/CLIP 不低于 Student budget
```

通过才 commit parameters、optimizer state 与 behavior EMA；否则 backtrack learning rate 一次，仍
不通过则 identity/no-op。这里不用 QP/一阶 constraint prediction，因为真实 probe 已判负。

### 6.7 一个有趣但不放进 v1 的 Purified candidate

Purified OPSD 的图像类比是把 prompt-conditioned reward gradient 与 prompt-independent shortcut
分开。例如用 shuffled/empty prompt probe：

```text
g_transfer(y,c) = grad_y R(y,c) - grad_y R(y,c_shuffle)
```

再从 clean Student anchor 沿 `g_transfer` 构造 DiffusionOPSD target。它可能去掉 PickScore 的
prompt-independent/style exploitation，但目前没有证据证明这一 residual 与 OCR/HPS 保持相关。
因此只做离线 cosine/reward probe；不与 RPI v1 同时上线。

---

## 7. 为什么 RPI-OPSD 比继续做 RG 三臂更有希望

原 RG `A0/AD/AG` 只在 optimizer 后选择，但现有结果表明：`AD` 与 `AG` 都大幅掉 HPS；加 HPS
gate 后最可能频繁选择 `A0`。RPI 在 commit gate 之前先扩大并净化 proposal set，尝试真正生成
quality-safe direction。

它不是无根据堆模块：

- DiffusionOPSD 已证明 gradient proposal 的 target construction 很强；
- Self-OPD 说明 trajectory-level reward selection 可避免直接多-gradient conflict；
- 我们的 hinge 说明 OCR/GE budget 可守；
- same-state probe 说明 post-fit exact decision 比 pre-fit linear prediction可靠；
- 当前 Open3 训练早期三路 reward 同步上升，提示 multi-reward feasible direction 不是空集，但
  这仍只是训练日志，不是 held-out 结论；
- LLM 的 TrOPD/U-OPSD/Purified OPSD 都独立支持 selective reliable correction，而不是全量蒸馏。

---

## 8. 先于结果冻结的 kill gates

### Gate A：零训练 branch-feasibility probe

在 32 个 frozen on-policy states 上比较 anchor、1 个 gradient proposal、4 个 SDE branches；每个
proposal 用相同 suffix/seed scorer。进入训练的必要条件：

1. ≥25% states 存在 non-anchor feasible branch；
2. 选中 branch 的 median Pick improvement >0；
3. HPS/CLIP constraint pass ≥90%；
4. 换一组 suffix seed 后，≥70% accepted states 仍通过约束。

若第 1 条失败，说明当前 Student 附近根本找不到 local Pareto improvement，**停止 continuation
方法线**；不要靠放宽 HPS tolerance 让它通过。

### Gate B：20-update smoke

同一 Student 起点，比较：

```text
Student-0
OPSD-style direct continuation
fixed hinge w=10
RPI-OPSD
```

最低成功条件：

- Pick 相对 Student > `2σ`；
- OCR/GE/HPS/CLIP 均不低于 Student 超过各自 `2σ`；
- non-noop commit rate ≥30%；
- 同时报告 GPU-hours；不能只比较 update count。

若 Pick 上升但 HPS/CLIP 仍掉，或 commit rate <30%，停止 RPI 长训；说明 candidate generator
仍没有解决 broad quality，而 selector 只是延迟失败。

### Gate C：paper scale

只有 A/B 同时通过才做 ≥3 independent seeds，并补 own-run `Student→Self-OPD`。在官方 mixed
release 不完整时标为 faithful reimplementation。checkpoint 只画曲线，不当独立样本。

---

## 9. 当前最合理的执行顺序

1. 等正在跑的 DiffusionOPSD Open3、CLIP specialist、HPS specialist 给出结果；它们直接回答
   joint/matched reward 是否能提供比 Pick-only 更好的 proposal。
2. 不立即再提交长 continuation；先实现 Gate A 的 target/branch feasibility probe。
3. Gate A 过后再做 20-update RPI；不过则把项目转为“OPSD capability cost / reward hacking
   analysis + benchmark”，不继续堆 guard。

最终判断：**我仍愿意为一次 selective Pareto improvement gate 投入算力，但不再对普通
continuation 或更多 scalar penalty 有信心。**
