# 0072 — Q 是否应成为 paper 主线，以及怎样真正增强 Q

日期：2026-09-04

本条是方向判断与方法设计，没有提交新任务。

---

## 1. 为什么要补 Q 的 step curve

补曲线不是因为已经决定把 Q 当主线，而是为了回答它有没有成为主线的资格。

当前观察是：

- Q-Core old/behavior adapter @ DPM2-10：相对 baseline `+0.0561`；
- Q-Core default/EMA adapter @ official flow-40：相对 baseline `-0.0141`；
- old adapter 的 Q-Core 在六个 sampler 上绝对分数很平。

因此至少有两种完全不同的故事都能解释现有数据：

1. **积极故事：**Q 把多步推理的一部分收益摊销进模型，构成 few-step acceleration；
2. **消极故事：**old behavior model 被训练到一个对 solver 不敏感但质量受限的平坦工作点，
   并没有形成可部署的加速模型。

default/EMA adapter 的精确 DPM2 step curve 能直接区分二者。如果 default 也满足：

- Q@10 显著超过 baseline@20；
- 在 24/28/32 附近存在稳定交点；
- 独立指标不下降；
- 多 seed 重复；

那么 Q 可以转成以 **value-gradient amortized sampling acceleration** 为中心的 paper 主线。
如果正结果只存在于 old adapter，或只存在于训练 reward，则 Q 不应做主线。

所以 curve 是 **go/no-go gate**，不是为了给已经选好的故事补图。

---

## 2. 当前 Q-Core 的三个核心瓶颈

### 2.1 Q 的定义与部署 sampler 不一致

当前：

```text
训练 state：DPM2-10 rollout
Q suffix：DPM2 原网格的剩余后缀
部署主评测：flow-40
```

Q-Core 精确优化了 `Q_DPM2(s,a)`，但论文主表需要的是 flow trajectory 的终点质量。
它在 matched DPM2 上大胜、flow 上转负，正是这个不一致的直接表现。

仅把 suffix helper 换成 flow 还不完整，因为 DPM2 state + flow suffix 是 hybrid trajectory。
更完整的定义应显式包含 solver：

```text
Q_S(s,a) = R(F_old^S(s,a)),  S ∈ {DPM2, flow, ...}
```

然后选择：

- deployment-aligned `Q_flow`；
- solver-conditioned `Q(s,a,S)`；
- 或 normalized solver ensemble
  `d = Σ_S w_S · unit(∇a Q_S)`。

在训练前必须用 state×suffix probe 判断 DPM-Q 与 flow-Q 是一致、互补还是冲突。

### 2.2 构造 target 用了 Q，但分支权重仍不是 Q-consistent

当前 Q-Core 只替换了 `y±` 的构造 oracle；dual loss 中的 `r1` 仍来自完整 rollout 的
group advantage。于是存在一个语义错位：

- proposal 方向回答：“在这个 state 改 action 后，frozen suffix 的 return 怎样变化？”
- 分支权重回答：“原始采样到的整条 trajectory 在同 prompt group 中排名怎样？”

两者不一定对同一个局部 action 给出一致判断。

因为 Q-Core 已经能计算 `Q(s,y+)` 与 `Q(s,y-)`，更一致的权重应来自 counterfactual
branch value，例如：

```text
A_Q+ = Q(s,y+) - Q(s,y0)
A_Q- = Q(s,y0) - Q(s,y-)
w+   = softmax([A_Q+, A_Q-] / tau)
```

也可保留原 group advantage 作 prompt-level baseline，但用 `A_Q` 决定本 state 的分支
可信度。该版本可叫 **Q-consistent advantage OPSD**。

开训前先测：

- 当前 `r1` 与 `Q(y+)-Q(y-)` 的相关性；
- 二者符号冲突率；
- 冲突样本是否贡献了 official flow-40 的亏损。

如果相关性已经很高，则不值得改；如果接近零或大量反号，这是比调步长更核心的切入点。

### 2.3 Q target 到参数更新的实现率太低

项目已有实测显示，构造出的 reward improvement 最终只有约 1–3% 被模型参数更新实现。
Q-Core 仍沿用“先构造 image-space target，再用 dual regression 去拟合”的长链条：

```text
terminal reward → action Q-gradient → y± target → regression loss → LoRA update
```

最保守的增强是给 Q-Core 加 inner2/更合适的学习率；它可能提高结果，但论文贡献较弱，
而且应先等 lr2x/inner4 把拟合机制识别清楚。

更第一性原理的版本是直接用 Q action gradient 形成 parameter surrogate：

```text
dQ = stopgrad(∇a Q_old(s,a_old))
L_DVG = - <unit(dQ), a_theta(s) - a_old(s)>
        + lambda * trust_region(a_theta, a_old)
```

其参数梯度直接是 `-J_theta^T dQ`，绕过固定 target 的二次回归、镜像 dual loss和
group-weight 稀释。这相当于 **truncated deterministic value-gradient distillation**：
suffix 与 reward model 只负责提供 action-space pseudo-gradient，student 只反传一次
query action 到 LoRA。

它比“Q-Core 再调 rho”更像真正的新方法，但也必须与 ReFL/直接 reward
backprop 比较，说明它通过 frozen short suffix 降低了全轨迹反传成本和方差。

---

## 3. 其他值得优化但优先级较低的部分

### 3.1 固定 query timestep

当前固定 `kq=8, sigma≈0.278`，Q 只学习一个晚期 state 的局部改进。若目标真是推理加速，
应检查多个 query sigma：早期负责全局布局，晚期负责细节/偏好。

不要直接随机 timestep 开训。先测每个 sigma 的：

- `cos(d_clean,d_Q)`；
- Q gain 与负增益率；
- 参数可达率；
- DPM-Q/flow-Q 一致性。

再按有效区间做 curriculum 或分层采样。

### 3.2 固定半径与梯度归一化

当前每个样本都把 Q gradient 归一化并走到相同 trust radius，梯度范数和局部曲率信息被
全部丢掉。可以用两步非镜像结构下的 branch-wise value line search：每一步分别验证
`Q(y+)` 上升、`Q(y-)` 下降，并保留逐轴安全条件。

早期 qopsd 的 backtracking 不能复用，因为它：

- 退化成一步严格镜像；
- 只检查 scalar positive branch；
- 不检查 negative branch 和逐轴条件。

### 3.3 多 reward 的 Q 组合

当前 DPM2-10 下三条 Open3 轴同时改善，说明 reward conflict 不是首要问题。只有当
flow-Q probe 显示不同 reward 的 suffix gradients 明显冲突时，才考虑：

```text
d = robust_combine(unit(∇Q_pick), unit(∇Q_clip), unit(∇Q_hps))
```

例如 max-min/common-descent 组合。现在先做会把 solver mismatch 与 reward geometry 混在一起。

---

## 4. 建议的主线决策树

### Gate A：Q 是否有可部署的 acceleration 信号

补 default/EMA step curve、24/28/32 精确点、墙钟、独立指标。

- **通过：**进入 Q paper track；
- **不通过：**Q 降为机制观察，主线回到 inner2/lr2x 拟合端。

### Gate B：Q 应该怎样跨 solver

做 `state source × Q suffix × evaluation suffix` 零训练矩阵。

- flow-Q 在 flow return 上显著胜：训练 qcore-flow；
- 两种 Q 各自在本 solver 胜、交叉失败：做 solver-conditioned/ensemble Q；
- 两种 Q 都没有稳定 counterfactual gain：停止 Q 训练。

### Gate C：当前 r1 是否与 Q branch value 一致

- 高相关、低冲突：保留原 advantage；
- 低相关/高冲突：实现 Q-consistent advantage；
- Q-consistent 权重仍被低实现率吞掉：转 direct value-gradient surrogate。

### Gate D：最后才组合拟合增强

只有 Q 方法本身在 official/default 口径转正后，再测试 Q+inner2 或 Q+合适 LR。否则组合臂
即使转正，也无法证明 Q 有贡献。

---

## 5. 哪个 Q 版本最可能更强

按“短期成功概率”排序：

1. **deployment-aligned Q-Core + 已验证的拟合增强**：最容易涨分，但方法新颖性一般，
   且必须先单独证明两部分；
2. **Q-consistent advantage OPSD**：改到 loss 的核心语义，理论闭环比当前 Q-Core 更完整，
   计算增量相对有限；
3. **solver-conditioned/ensemble Q**：最直接处理当前强烈的 protocol interaction；
4. **direct suffix value-gradient distillation**：最可能形成独立 paper 方法，也最需要新的
   稳定性与 trust-region 设计。

我当前最推荐的研究组合不是立刻把四项全堆起来，而是：

```text
先用 probe 选择 Q_flow 或 Q_ensemble
        ↓
测 r1 与 counterfactual A_Q 是否冲突
        ↓
若冲突，做 Q-consistent advantage
        ↓
若仍受 1–3% realization 限制，再升级到 direct value-gradient
```

这条路线每一步都有可证伪观测，也逐步触及 Q 方法真正的核心，而不是在现有 Q-Core 外围
继续加小旋钮。

---

## 6. 最终立场

Q 现在是**候选主线，而不是已确定主线**。它比 inner2 更有方法新颖性，但尚没有 official
正结果；inner2 更接近稳定涨分，但自身更像优化 recipe。

Q 若通过 default curve 和多 seed，其最佳 paper 叙事应是：

> 利用 frozen suffix action value 的梯度，把多步采样中的未来收益蒸馏到当前 action，
> 从而摊销推理步骤；进一步用 solver-conditioned value 与 counterfactual Q-advantage
> 解决离散求解器依赖。

如果这些 gate 不通过，就不应为了新颖性强行把 Q 写成主线。
