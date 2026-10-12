# 0006 — 最值得押注的论文方法：GuardedOPSD

Date: 2026-08-29  
Status: method recommendation; no new job submitted in this entry

## 一句话结论

最值得继续做的不是再调一组 CA / tau / top-k / 固定 loss weight，而是把项目重新定义为：

> **在一个已经整合 OCR、GenEval 等能力的 diffusion Student 上，用 OPSD 加入一个新 reward，同时用旧专家 teacher 约束每次更新，使已有能力不被遗忘。**

建议的方法名暂定为：

> **GuardedOPSD: Capability-Preserving On-Policy Self-Distillation via Feasible-Update Projection**

核心不是“把三个任务继续平均训练”，而是明确区分：

- PickScore 是当前真正要提高的 `new objective`；
- OCR、GenEval 是不能退化的 `capability constraints`；
- 只有预测会破坏旧能力时，保护约束才介入；
- 在不违反约束的方向里，保留尽可能多的原始 OPSD 更新。

这比对所有目标对称地做梯度平均更符合现在的真实问题，也比继续搜索固定权重更容易形成清楚的论文主张。

---

## 1. 为什么我现在最看好这个方向

### 1.1 它直接建立在项目已经验证较强的部分上

现有证据已经说明：

- released DiffusionOPD Student 是目前可信的多能力起点；
- 官方 OPSD direct-reward 路径是最明确的新 reward 改善引擎；
- 单独 direct training 有明显的新 reward 梯度，但会威胁 OCR 等原能力；
- 我们的 smoke test 已确认“冻结初始 Student + 外部 specialist teacher + 同一 on-policy state 的相对 KL”在代码和梯度层面可运行；
- retention 梯度和 direct 梯度相差约两个数量级，说明固定共享权重本来就不适合作为最终机制。

因此，这一方向不是再发明一个未经验证的生成方法，而是在一个能工作的增益引擎外面增加一个明确的 capability safety layer。

### 1.2 它解释了此前方法为什么混乱

此前 CA、CAGrad、POE、retain、top-k 等尝试混在一起，是因为没有先区分下面两个问题：

1. 是要从头找多目标 Pareto 解？
2. 还是已经有一个多能力模型，现在要添加一个新 reward 且不遗忘？

当前项目更自然的是第二个问题。这个定义下：

- moving self-snapshot retain 太弱，因为 reference 自己会跟着漂移；
- 固定 teacher loss 权重需要不断调尺度；
- CAGrad 把全部任务当成对称 objective，但这里旧任务其实只是 safety constraints；
- top-k 只是 teacher sample selection，并没有解决新 reward 更新导致的遗忘；
- Pareto scalarization 会给出一系列 trade-off，却不能保证“旧能力不低于预算”。

### 1.3 它更容易得到一种清楚的正结果

最有说服力的主结果不是“某个加权平均分提高了”，而是：

> Direct OPSD 提高 PickScore，但 OCR / GenEval 下降；GuardedOPSD 保留大部分 PickScore 增益，同时把 OCR / GenEval 控制在 released Student 的置信区间或预设 degradation budget 内。

这是一个一眼能看懂的结果，也有清楚的失败判据。

---

## 2. 必须诚实面对的 prior-art 边界

只做 primal-dual 或 adaptive lambda，不足以成为主创新。

| 已有方向 | 已经覆盖的内容 | 本项目不能只声称什么 |
|---|---|---|
| [DiffusionOPD](https://arxiv.org/abs/2605.15055) | 多 teacher、student on-policy rollout、逐步 KL distillation | 不能只声称“用多个 diffusion teacher” |
| [DiffusionOPSD](https://arxiv.org/abs/2608.24646) | 用 reward gradient 构造 on-policy self-distillation target | 不能只声称“直接优化 reward target” |
| [Composition and Alignment of Diffusion Models using Constrained Learning](https://arxiv.org/abs/2508.19104) | reward / model KL constraints、primal-dual constrained training | 不能只声称“把旧任务写成约束并学 lambda” |
| [ParetoSlider](https://arxiv.org/abs/2604.20816) | preference-conditioned Pareto front | 不能只声称“学一条多目标 Pareto 曲线” |
| PCGrad / CAGrad 等 | 一般多任务梯度冲突处理 | 不能只声称“投影冲突梯度” |

因此论文贡献必须是一个更具体的组合：

1. **continual diffusion alignment 场景**：已有 unified Student，增加新 reward，旧 reward 甚至可以不可微或不可再用；
2. **teacher-relative、same-query capability budget**：用冻结 specialist teacher 和冻结初始 Student 在同一个当前 on-policy query 上定义可达到的相对退化量；
3. **asymmetric feasible-update projection**：只把新 reward 当优化目标，把旧能力当不等式约束；
4. **post-step cached-state certificate**：候选更新后重新检查能力约束，弥补一阶投影与真实网络更新之间的误差。

如果最后只实现第 1、2 点和 adaptive lambda，仍然可能产生有用实验或 workshop paper，但主会 novelty 风险明显更高。

---

## 3. GuardedOPSD 的具体形式

### 3.1 初始化与三类 minibatch

从 released multi-capability Student 参数 `theta_0` 开始，并冻结：

- 初始 Student `theta_0`；
- OCR specialist teacher `T_ocr`；
- GenEval specialist teacher `T_ge`。

每个更新周期使用三种 prompt slot：

- PickScore prompts：构造官方 OPSD reward target；
- OCR prompts：只评估 OCR capability constraint；
- GenEval prompts：只评估 GenEval capability constraint。

这避免了把错误 teacher 用在不对应的 prompt/task 上。

### 3.2 新 reward 的原始方向

严格复用官方、已验证的 OPSD target construction，得到新 reward loss：

```text
L_new(theta) = L_OPSD-PickScore(theta)
g_new = grad_theta L_new
```

第一版不修改 OPSD reward target、SNR、query timestep 或 reward scaling。否则同时改变太多变量，无法知道正结果来自哪里。

### 3.3 旧能力约束

对旧能力 `m in {OCR, GE}`，在同一个当前 on-policy state `x_t` 上计算：

```text
D_cur,m  = KL[p_theta(. | x_t) || p_Tm(. | x_t)]
D_ref,m  = KL[p_theta0(. | x_t) || p_Tm(. | x_t)]
Delta_m  = D_cur,m - (1 + epsilon_m) * D_ref,m - delta_m
C_m      = mean(Delta_m)
```

要求：

```text
C_m <= 0
```

这里最重要的是 `D_cur` 和 `D_ref` 必须在**完全相同的当前 query state**上计算：

- teacher 不一定和 Student 架构或绝对 KL 尺度相同；
- `D_ref` 给出了 frozen Student 在这个 query 上本来就存在的 teacher mismatch；
- 相对预算只限制“相对初始 Student 新增的退化”，而不是强迫模型瞬间复制 teacher；
- 初始点天然可行，避免一开始就陷入无法满足的约束。

当均值约束已经跑通后，可把 `mean` 换成 worst-quantile / CVaR，测试是否能防止少数 OCR prompt 严重退化；这应作为后续 ablation，而不是第一版的额外复杂度。

### 3.4 非对称的可行更新投影

令：

```text
a_m = grad_theta C_m
```

模型执行 `theta <- theta - eta * d`。在一阶近似下：

```text
C_m(theta - eta*d) ~= C_m(theta) - eta * a_m^T d
```

因此选择离原始 reward 梯度最近、但下一步不跨过 constraint boundary 的方向：

```text
min_d  0.5 * ||d - g_new||^2
s.t.   a_m^T d >= (C_m + safety_margin_m) / eta,  for m in active constraints
```

只有两个旧能力约束时，这只是一个由 `2 x 2` gradient Gram matrix 决定的小 QP，不需要在全参数维度调用通用 QP solver。其解可以写成：

```text
d* = g_new + A^T lambda*
lambda* >= 0
```

`lambda*` 由二维 non-negative QP / active-set enumeration 求出。

这个机制和 CAGrad 的关键差别是：

- CAGrad 把所有任务对称地当优化目标；
- GuardedOPSD 有一个明确的主目标和若干硬约束；
- 约束安全时，它对 `g_new` 的影响严格为零；
- 约束将被突破时，它选择对原 OPSD 更新改动最小的可行方向；
- 不需要把相差约 100 倍的 gradient norm 手工压进一个固定 loss weight。

### 3.5 更新后的 capability certificate

一阶投影不能直接声称对整个 prompt distribution 有理论保证，尤其 AdamW、LoRA 和 on-policy state 都会带来非线性误差。因此每一步应增加一个诚实而实用的检查：

1. 保存更新前的 LoRA 参数；
2. 应用候选更新；
3. 在这一步缓存的 OCR / GE states 上重新计算真实 `C_m`；
4. 如果违反约束，对候选 parameter delta 做 backtracking；
5. 若缩到最小步长仍不安全，则拒绝该步并记录原因。

这个 certificate 只保证当前缓存的 on-policy constraint batch，不应被表述成全分布的形式化 safety guarantee。下一步刷新 rollout 后会重新检查新 states。

---

## 4. 最可能 work 的工程版与最可能发 paper 的完整版

### A. 最低风险的正结果 baseline

```text
official OPSD direct objective
+ frozen external-teacher relative constraints
+ adaptive primal-dual multipliers
```

优点：最容易实现，预期比固定权重稳定，适合作为第一条有效 constrained baseline。  
缺点：与已有 constrained diffusion prior art 重叠较大，不应当作最终主方法。

### B. 建议作为主方法的 paper version

```text
official OPSD direct objective
+ teacher-relative same-query capability budgets
+ asymmetric feasible-update projection
+ post-step cached-state certificate
```

优点：问题定义清楚；更新行为可解释；不会在 constraint inactive 时无谓拉回 teacher；能直接回答 fixed-weight、CAGrad 和 moving-retain 的缺陷。  
风险：投影经过 AdamW 后是否仍准确需要专门处理和验证，certificate 可能带来额外 forward cost。

### 推荐执行顺序

先实现 A 作为必须存在的 baseline，同时实现 B 的最小 Euclidean-gradient prototype。论文是否成立不应依赖 A；A 的作用是判断“约束本身是否能换来正结果”，B 的作用是判断“更有区分度的机制是否比 A 更好”。

---

## 5. 下一轮实验不应直接上 100 epochs

### Gate 0 — 先评估现有 direct engine

在任何新训练前，先确认现有 direct run：

- 真正提高了 PickScore；
- OCR / GenEval 的退化量和置信区间；
- 使用的是 raw trained state，而不是短 smoke 中尚未更新的 EMA export；
- evaluator、prompt split、seed 与 Student baseline 对齐。

如果 direct engine 本身没有可重复的 PickScore 增益，先修 direct engine，而不是继续给它加 constraint。

### Gate 1 — 20-step / 1-seed mechanism calibration

只跑三个 arm：

1. released Student（不训练）；
2. direct OPSD only；
3. GuardedOPSD projection + certificate。

每一步必须记录：

- `L_new`, `C_ocr`, `C_ge`；
- `||g_new||`, `||a_ocr||`, `||a_ge||`；
- constraint active rate；
- projection angle / `||d*-g_new||`；
- QP dual variables；
- certificate pass/fail；
- accepted backtracking scale；
- optimizer step、raw checkpoint、EMA checkpoint。

Gate 1 的目标不是得出最终指标，而是验证：

- direct-only 确实朝新 reward 方向移动；
- projection 不是永久 inactive；
- projection 也不是每步把更新压到接近零；
- certificate 偶尔能捕捉到一阶预测误差；
- raw train state 和 checkpoint 均真实改变。

### Gate 2 — 小规模 end-to-end 选择

比较：

- direct only；
- fixed-weight retention；
- adaptive primal-dual baseline；
- CAGrad / 已有最佳多任务 baseline；
- GuardedOPSD。

第一阶段成功判据建议预注册为：

```text
GuardedOPSD 获得 direct-only PickScore 增益的至少 70%–80%，
同时 OCR 和 GenEval 相对 released Student 的下降不超过预设 budget / 评估置信区间。
```

这里的 70%–80% 是实验 gate，不是需要写进方法定义的常数。最终应报告完整 Pareto/retention curve，避免只挑一个 threshold。

### Gate 3 — 扩成论文证据

至少需要：

- 2 个新 reward，而不是只有 PickScore；
- 2 个 base model / checkpoint setting，或至少明显不同的数据/teacher 组合；
- 3 个以上 seeds；
- 与 primal-dual constrained baseline 和 generic gradient surgery 的正面对比；
- wall-clock、teacher-forward cost、accepted-step rate；
- mean constraint 与 tail/CVaR constraint ablation；
- same-query frozen reference 与 moving snapshot ablation。

否则即使 SD3.5 上一条曲线好看，也更像一个项目技巧而不是一般性方法。

---

## 6. 哪些方向暂时不应继续消耗主线算力

- **继续扫 CA / tau**：已有实验没有给出稳定 final win，且论文故事不再清楚。
- **继续用 moving self-snapshot retain**：reference 会漂移，无法代表旧能力锚点。
- **只调固定 POE / loss weights**：可以做 baseline，不适合作为主贡献。
- **把 top-k 当主方法**：先修正并验证完整评估；它最多是 teacher sample selection component。
- **再换一种通用 gradient surgery**：除非能回答为何它适合 asymmetric continual alignment，否则 novelty 和结果风险都高。
- **直接跑很长的 constrained job**：在 direct engine、constraint activation 和 checkpoint export 没过 gate 前，只会制造昂贵但不可解释的曲线。

---

## 7. 最重要的 falsification 条件

以下任一情况出现，都应及时降低对该方向的投入，而不是继续调参掩盖：

1. Direct OPSD 在严格 evaluator 下没有稳定新 reward 增益；
2. 旧能力下降主要来自 evaluator noise，而不是训练干扰；
3. 可行投影几乎每一步都返回零更新，说明当前 Student 附近没有可用的局部 trade-off；
4. adaptive primal-dual baseline 已经同时达到相同 retention 和 reward，而 projection 没有更少调参、更低 violation 或更高收益；
5. same-query teacher KL 与真实 OCR / GenEval 变化无相关性，说明 proxy constraint 选错；
6. 方法只在单一 seed / 单一 reward / 单一模型成立。

其中第 5 点尤其关键：论文必须先证明 teacher-relative KL 是真实旧能力退化的有效 leading indicator。可以在 direct-only trajectory 的多个 checkpoints 上画：

```text
teacher-relative KL change  vs.  downstream capability change
```

如果相关性弱，应重新设计 constraint proxy，而不是继续优化 QP。

---

## 8. 当前最终判断

### 最可能先得到正结果

**官方 direct OPSD + frozen specialist-teacher relative constraints + adaptive dual baseline。**

理由是组件都已分别工作，且 adaptive dual 能自动处理现有约 100 倍 gradient-scale mismatch。

### 最有希望形成清楚论文贡献

**GuardedOPSD：same-query relative capability budget + asymmetric feasible-update projection + post-step certificate。**

最好的论文叙事不是“我们又找到一种多任务加权”，而是：

> **We can add a new reward to an already unified diffusion model without forgetting old, possibly non-differentiable capabilities, by filtering each on-policy self-distillation update through frozen teacher-defined capability constraints.**

这条主线既能利用 DiffusionOPD / DiffusionOPSD 已有基础，又明确解释了为什么此前那些对称多目标方法不够合适。

