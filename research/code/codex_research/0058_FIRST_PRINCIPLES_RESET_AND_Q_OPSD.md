# 0058 第一性原理重置：为什么小改动都不 work，以及 Q-OPSD 方向

**日期：2026-08-31 23:19 EDT**  
**性质：独立复核 + 方法推导；本轮未提交新训练**  
**依据：Claude 0051–0057、`EXPERIMENT_LOG.md` 至 §4.58、原始 held-out JSON、
DiffusionOPSD 原论文/钉住代码、Self-OPD 原论文，以及最新近邻工作。**

---

## 0. 结论先行

用户的直觉**基本正确，但需要说得更精确**：

> 目前没有方法赢，不只是因为“改动小”，而是因为 N1/A1/N2/A3/Gate P/top-k
> 几乎都保留了 DiffusionOPSD 的同一个核心 improvement operator：
> **在单个 low-noise query 的 clean-output 坐标里，对局部 reward 做梯度上升，
> 得到 detached target，再用一个共享 LoRA 拟合。**

这些方法改变的是权重、mask、normalizer 或候选选择，却没有改变“什么量应该被优化”。
因此它们最多会：

- 几乎不触发（A1）；
- 沿原有 Pareto 面换位置（N1）；
- 不改变更新方向，只改标度（N2）；
- 丢掉样本、降低有效 batch（cert/A3）；
- 放大候选选择噪声（top-k / LSR 风险）。

但“必须想一个原理和实现都完全不像 DiffusionOPSD 的方法”也不是正确目标。
完全不同很可能只是重新发明 ReFL、FlowGRPO 或 Self-OPD。真正需要替换的是一个核心数学对象：

```text
DiffusionOPSD 当前优化：局部 clean proxy  R(D(y))
真正的序列决策对象： action value       Q_old(s,y)=R(D(F_old(s,y)))
```

其中 `F_old(s,y)` 表示：在 query state 强制采用 clean action `y`，再由冻结 behavior policy
走完**精确 suffix**得到终点。

本报告首推的新方向是：

> **Q-OPSD / Suffix-Value OPSD：不再对 `R(D(y))` 求梯度，而对
> `Q_old(s,y)=R(D(F_old(s,y)))` 求梯度；然后把这个真正考虑未来 rollout 的 action-value
> 梯度做成 detached target，再沿用 OPSD 的稳定拟合框架。**

它不是又一个 selector。它把 DiffusionOPSD 的“即时 clean proxy”换成真正的 return-to-go，
是 DiffusionOPSD 的可微 target construction 与 Self-OPD 的 terminal-return action evaluation
之间最自然的桥。

---

## 1. Claude 最新整理与原始结果：哪些结论可信

### 1.1 同种子 100-update A/B 主结果

| 臂 | 实际改变 | 结果 | 正确解释 |
|---|---|---|---|
| N1 | Open3 静态权重 | 2 胜 2 负；最低专家保持率 87.9%→75.6% | 换 Pareto 位置，不是整体改善 |
| A1 | clean-target 三轴 veto | 五轴全打平，`|t|≤0.35` | veto 仅触发 1.39%，干预基本没发生 |
| N2 | frozen/global normalizer | 五轴全打平，`|t|≤1.40` | 每步都生效但假设错误；独立旧跑还伤 Pick |
| A3 | exact-suffix binary cert | 已提交，尚未起跑 | 有触发率，但可能只是丢样本；不能先报正向 |

原始 JSON 与 0052/0056 的表一致：

```text
baseline: Pick 24.6588 / CLIP .3165 / HPS .3704 / Aes 5.8958 / ImgR 1.4331
A1:       Pick 24.6475 / CLIP .3173 / HPS .3703 / Aes 5.8949 / ImgR 1.4308
N2:       Pick 24.6598 / CLIP .3167 / HPS .3710 / Aes 5.9257 / ImgR 1.4330
```

N1 的大幅 trade-off、A1 的近乎恒等、N2 的近乎恒等，都不是模糊结果。

### 1.2 最重要的机制结果：三层而不是一个“实现率”

必须把三个量拆开：

1. **local construction gain**：`R(D(y+))-R(D(y0))`；
2. **fixed-suffix gain**：`R(D(F_old(s,y+)))-R(D(F_old(s,y0)))`；
3. **policy gain**：参数更新后，新 policy 完整生成的 reward 改变。

Claude 的 cert probe 给出：

| updates | local construction | fixed suffix | local→suffix retention |
|---|---:|---:|---:|
| 0–9 | 0.18394 | 0.11664 | 63.4% |
| 10–19 | 0.17110 | 0.07356 | 43.0% |

同时，指示性的 suffix→policy retention 从 23.9% 退到接近 0；后一个点噪声很大、且来自
跨跑拼接，**不能当精确点估计**，但说明第二段至少没有被解决。

两个重要细节：

- early cert 的**正号率**仍有 95.5%→87.7%，但连续 gain retention 已只有 63.4%→43.0%。
  所以 binary cert 只保证“符号不是负”，不保证方向有足够幅度。
- 上升幅度在固定 `R_old` 分层后下降 29–50%，下降幅度只下降约 0–13%。
  这不像模型整体变僵，更像**reward-up direction 与实际 dynamics 的匹配在恶化**。

### 1.3 时间结构否定了“再训久一点自然会好”

- baseline 总 `reward_avg` 增益的 **63.6% 在前 10 updates**；
- 单独 epoch 1 占总增益 25%；
- 后 50 updates 的斜率 `t=2.12<2.5`；
- baseline/A1/N2 在共享 seed 下逐 epoch 曲线几乎重合；大震荡主要由 prompt batch 驱动。

因此 A1/N2 不是“效果被噪声盖住”，而是几乎没有把训练轨迹推到另一条路上。

### 1.4 两个对新 idea 很不友好的负结果

1. old `wf100`：Pick 相对 canonical `−5.54×2σ, t=−2.90`；
2. old `topk100`：Pick/HPS/ImgR 分别 `−11.80/−8.00/−5.63×2σ`。

旧仓库、不同 seed、manifest 不完全可比是 caveat；但 top-k 效应很大，且 Self-OPD 自己的
ablation 也报告 Best-of-K 不稳定、弱于 all-branch advantage weighting。

因此“从更多候选里挑一个看起来最好的 target”目前是**负面先验**，不是首选。

---

## 2. 所有已试方法到底没有改变什么

令 query state 为

\[
s=(c,z_q,\sigma_q),\qquad y_0=z_q-\sigma_q v_{old}(s).
\]

canonical DiffusionOPSD 先构造

\[
d_{clean}\propto \nabla_y \widetilde R(y_0,c),\qquad
y_\pm=y_0\pm d_{clean},
\]

再用 detached MSE 让 `y_theta` 靠近正 target、远离负 target。

| 方法 | 改了什么 | 没改什么 |
|---|---|---|
| N1 | reward scalarization | 仍是 `∇ clean reward → detached y target → shared LoRA fit` |
| A1 | local target 的 accept mask | 同一个 oracle、同一个 target、同一个 fit |
| N2 | fit loss 的尺度/几何 | target direction、query、policy class 全不变 |
| A3/cert | suffix 后做 binary mask | 不改变错误方向，只丢弃它；不补 batch |
| Gate P/LSR | 在同一 reward-gradient 家族中选方向 | 仍是 isolated clean target；还有 selection bias |
| top-k | 从候选中挑最高分 | 同一信息族，且实测显著变差 |

用信息观点说：A1/N2/A3 是对同一份信息

```text
I = {on-policy state, clean anchor, local reward gradient, endpoint scalar advantage}
```

做筛选或缩放。它们不会凭空产生一个原 oracle 没有的 trajectory-consistent improvement direction。
N1/Gate P 能在已有 reward-gradient span 内重组，但仍不能看见 suffix dynamics。

这就是为什么目前“idea 看起来不同”，训练轨迹却几乎相同。

---

## 3. 第一性原理：DiffusionOPSD 把非终点 action 当成了终点 reward

扩散/flow 采样是序列决策过程。query 处的 clean prediction `y` 是一个 action 表示，
不是最终图片。正确的 behavior-policy action value 应为

\[
Q^{old}_q(s,y)
=
R\!\left(D\!\left(F^{old}_q(s,y)\right),c\right),
\]

其中 `F_old_q` 是“强制 query action 后，用 old policy 走完剩余 suffix”的 flow map。

DiffusionOPSD 实际使用的是

\[
\widetilde R(y,c)=R(D(y),c),
\]

它隐含假设：

\[
\nabla_y\widetilde R(y,c)
\approx
\nabla_y Q^{old}_q(s,y).
\]

训练初期这个近似可能尚可；cert probe 证明它会随训练恶化。更精确地，若终点 reward 梯度为
`g_R`、suffix flow map 的 action Jacobian 为 `J_F`，则正确方向是

\[
\nabla_y Q^{old}_q=J_F^\top g_R,
\]

canonical local clean direction没有 `J_F^T`。N1/A1/N2 都没有补上它；A3 只在事后问方向
有没有碰巧成功。

### 3.1 为什么这也解释参数拟合段

令 trainable model 在 query 的输出为 `y_theta`，其参数 Jacobian 为 `J_theta`。
若 target 取

\[
y_+=y_0+\epsilon\,\mathrm{normalize}(\nabla_y Q^{old}_q),
\]

则在每个 outer update 开始的 `theta≈theta_old, y_theta≈y0` 附近，target MSE 的梯度下降方向为

\[
-\nabla_\theta \|y_\theta-y_+\|^2
\propto
J_\theta^\top \nabla_y Q^{old}_q
=
\nabla_\theta Q^{old}_q(s,y_\theta).
\]

也就是说：**用 suffix-value gradient 做 detached target，第一步的 parameter gradient 与对
fixed-suffix return 直接反传是同方向的**（忽略 normalization、dual-branch scalar 和有限步误差）。

这比“把错误 target 拟合得更用力”（N2）更根本：它先把 per-sample 拟合方向变成正确的
policy-improvement direction。

仍需诚实：跨样本共享 LoRA、AdamW、batch interference 仍可能使实际更新偏离；但这已经把
错误从“目标对象不对”推进到可直接测量的 optimizer 层，而不是继续用 loss scalar 猜。

---

## 4. 推荐主方法：Q-OPSD（也可命名 Suffix-Value / Flow-Map OPSD）

### 4.1 一句话

> **对冻结 behavior policy 的 exact suffix action value 做 trust-region ascent/descent，
> 再把得到的 on-policy policy-improvement action 蒸馏回 student。**

### 4.2 算法

对每个 on-policy query：

1. 与 canonical 一样得到 `s=(c,z_q,sigma_q)`、`y0`、`prev_x0`；
2. 定义可微 exact suffix：
   `e(y)=F_old_q(z_q,y,prev_x0)`；且已有 null sanity：`e(y0)` 以约 `2e-6`
   相对误差复现原 endpoint；
3. 定义多 reward action value：

   \[
   Q(y)=\sum_m\lambda_m R_m(D(e(y)),c);
   \]

4. 通过 frozen old-policy suffix 反传，计算 `g_Q=∇_y Q(y0)`；
5. 在与 baseline **相同半径**的 trust region 内构造 `y+ / y-`；
6. 用 endpoint-Q 的 backtracking `alpha∈{1,.5,.25,.125}` 验证，而不是失败就直接丢样本：
   选择最大、满足 scalar gain 正且 rollout-level per-axis 不显著退化的步长；
7. 复用 canonical dual-branch OPSD loss 拟合 detached `y±`；
8. 更新/EMA behavior policy，下一 outer iteration 重建 Q-target。

第一版不要同时改 reward weights、normalizer、query sigma、LoRA rank。唯一核心差异是：

```text
∇_y R(D(y))   →   ∇_y R(D(F_old(s,y)))
```

### 4.3 它如何统一 DiffusionOPSD 与 Self-OPD

| 方法 | 如何评价 query action 的未来价值 |
|---|---|
| DiffusionOPSD | 对 clean prediction 直接打分并求梯度；不看未来 suffix |
| Self-OPD | 对 K 个 stochastic next actions 全部 rollout 到终点，用 terminal reward 作 advantage |
| **Q-OPSD** | 对 exact suffix return 求可微 action-value gradient |

Self-OPD 的关键不是“随机噪声”本身，而是它用 terminal rollout reward 判断 action；
Q-OPSD 保留这个正确的序列决策对象，同时利用 Open3 rewards 可微这一事实，不需要每个状态
固定做 K=8 个 stochastic branches。

### 4.4 它为什么不是 A3 / Gate P / ReFL 的改名

- **不是 A3**：A3 对 canonical 错方向作二值 accept/reject；Q-OPSD 用 suffix Jacobian
  直接旋转方向，并用 line search 缩步，不缩有效 batch。
- **不是 Gate P/LSR**：Gate P 在 7 个手工方向中离散选一个；Q-OPSD 给出 action-value 的
  连续一阶最优方向，不依赖 winner selection。
- **不是 Self-OPD**：Self-OPD 是 derivative-free branch Monte Carlo；Q-OPSD 是可微 Q-gradient。
- **不等同 ReFL**：ReFL 对 trainable sampling graph 直接做 endpoint reward update；Q-OPSD
  在 frozen behavior suffix 上构造 detached on-policy target，再用 OPSD finite-fit/EMA 实现 policy
  iteration。二者一阶有联系，实验上仍须把 ReFL 作为 baseline。
- **与 FMRG 有近邻关系**：Flow Map Reward Guidance 用 flow map 做 inference-time optimal-control
  guidance；Q-OPSD 把相同的 flow-map value gradient**在线蒸馏进 policy**。写论文时必须明确引用，
  不能宣称首次发现 flow-map reward gradient。

---

## 5. 为什么我不再首推 “低分样本 curriculum / refill” 作为主贡献

cert probe 确实发现：低 `R_old` states 的 rollout 可达性几乎不退化，而高分 states 退化明显。
这提示训练预算分配有问题，但不能直接推出“只训最低分”。

最新 AdaGRPO 已经系统研究 flow-GRPO 的 capability-aware prompt filtering：

- easiest prompts 会严重退化；
- hardest prompts 只略好于/接近 random；
- 位于当前 capability boundary 的中等难度 prompts 最有效；
- 它还做了 global-EMA capability tracking 和 cross-level advantage。

因此：

1. 单纯 curriculum/refill **novelty 已不足**；
2. raw low reward 不等于 learnable；
3. 我们真正独特的 observable 是 **counterfactual suffix gain / Q-gradient**，不是 reward 难度。

curriculum 可作为 Q-OPSD 后续正交组件或 ablation，但不应是当前主方法。

---

## 6. 最新 LLM OPD 给出的关键启发：不是 mask，而是反事实决策

2026-08 的 LLM 工作 *Not Every Divergence Should Be Suppressed* 已把
**counterfactual recoverability** 明确定义为：在预算匹配条件下，比较 continuation 与 rollback
的未来成功率，再决定 retain / rollback / ordinary supervision。

它直接提醒我们：

- divergence、local feasibility、reward 高低都只是 proxy；
- A3 的“失败就 mask”只比较 canonical action 与 no-op，没有比较替代 improvement operator；
- diffusion 里已经有 exact suffix replay，因此能做更精确、连续的 action-value 版本。

这篇 LLM 工作必须列为方法动机/最近邻；“counterfactual recoverability”这个概念不能据为己有。
我们的 diffusion-specific 空间是：

1. 把 query clean output 明确当成 action；
2. 用 exact differentiable flow suffix 定义连续 `Q_old(s,y)`；
3. 用 `∇_y Q_old` 构造 dense target；
4. 研究 local proxy、suffix value、parameter realization 三层 gap。

若单一 Q-gradient 仍有失败尾部，第二阶段才考虑三路 routing：

```text
Q-gradient oracle 成功   → Q-OPSD target
Q-gradient 失败但 branch 可恢复 → Self-OPD all-branch target
两者都无可靠增益        → identity / ordinary path
```

这里选的是**不同 improvement operator**，不是从同一 candidate pool 里 top-1；但它是高成本
第二阶段，不应在 Q-gradient 本身未过 gate 前就实现。

---

## 7. 在提交训练前必须做的两个零训练 Gate

### Gate Q1：方向层——Q-gradient 是否真的修复 local→suffix gap

在 frozen baseline checkpoints（建议 ck0/10/50/100）上，用同一批 state、同一半径比较：

```text
d_clean = normalize(∇ R(D(y0)))
d_Q     = normalize(∇ R(D(F_old(s,y0))))
d_rand  = matched-radius random control
```

全部强制走 exact suffix，报告：

- mean/median/p10 fixed-suffix gain；
- negative-gain rate；
- Open3 三个分轴的负 gain rate；
- `cos(d_clean,d_Q)` 随训练和 `R_old` 分层；
- gradient norm、NaN/OOM、wall-clock；
- 换 seed 后稳定性。

**建议在 launch 前预注册阈值**，而不是看到结果再定。一个合理起点：

- `d_Q` 的 negative-gain rate 至少比 `d_clean` 降低 50%；
- p10 从负/近零恢复为正；
- mean suffix gain 相对 `d_clean` 提高至少 15%；
- 三个目标轴无系统性显著下降；
- 第二 seed 复现方向。

若 Q1 不过，Q-OPSD 终止；不调 rho 复活。

### Gate Q2：参数层——正确 Q-direction 能否通过一个真实 optimizer step

Q1 通过后，在 selection/validation 分离的同状态数据上：

1. 用一半 states 分别形成 canonical 与 Q-target 的 candidate AdamW update；
2. 临时应用更新；
3. 在另一半 states 上重新生成 exact suffix；
4. 比较同 seed、同 batch、同 optimizer 的实际 fixed-suffix gain；
5. 回滚参数，不形成正式训练 checkpoint。

同时记录 per-sample LoRA gradient sketch/Gram，回答第二段损失到底来自：

- target direction 错；还是
- 不同 prompt/state 的 parameter gradients 相互抵消；还是
- AdamW/preconditioner 旋转。

**这一步尤其重要**：Gate A 的 `+0.715` 是 reward-gain correlation，绝不是 parameter-gradient
cosine。共享 LoRA 的梯度相容性至今没有在 clean official baseline 上被测过。

若 Q1 过、Q2 不过，才有依据转向 gradient-coherent batching、adapter experts 或 exact commit；
此时继续改 target selector 没意义。

---

## 8. 训练矩阵（只有 Q1/Q2 通过才启动）

第一轮只需要三臂，同 base、同 seed、同 100 updates：

| 臂 | 目的 |
|---|---|
| official DiffusionOPSD Open3 | 主 baseline |
| A3 `opa_cert=1` | “事后二值丢弃 local target”的官方对照 |
| **Q-OPSD Open3** | “直接优化 suffix action value”的 ours |

第二阶段才加：

- ReFL：端到端 differentiable reward baseline；
- Self-OPD mixed own implementation：derivative-free action-value baseline；
- Q-OPSD without line search / local-gradient with same line search：拆出 Q-gradient 与验证成本；
- 若需要，Q-gradient→Self-OPD branch fallback。

必须报告 GPU-hours/wall-clock，因为 Q-OPSD 的 suffix backward 明显比 canonical 贵；
只比更新数不公平。

---

## 9. 当前作业如何处理

截至 23:19：

- A3 `1038387/1038388/1038389`：仍 pending；它是干净的 same-seed official-feature 对照，
  **保留并让它完成**，但不预期它自动成为正结果；
- Gate P v2 与 certp-100：均 held；
- 本轮**未提交任何新 job**。

建议优先级调整为：

1. A3 完成，封存 binary cert baseline；
2. 先写并审 Q1 的预注册与 differentiable-suffix correctness tests；
3. Q1 过才做 Q2；
4. Q2 过才提交 Q-OPSD 100-update A/B；
5. Gate P v2 可留作诊断，但 **LSR training 不再是主线**；top-k 与 Self-OPD Best-of-K
   已共同给 selection 路线负面先验；
6. certp-100 只是机制加固，不应抢 Q1 的方法验证资源。

---

## 10. 对“为什么一直没有 idea work”的最终回答

### 已有证据支持

1. **不是 multi-reward local conflict**：98.6% clean targets 三轴同升；
2. **不是 target-distance normalizer**：N2 两次独立负结果；
3. **不是单纯训练步数不足**：增益极度前置、后半程无显著斜率；
4. **不是 hard mask 不够严格**：mask 只能丢信息，且作用最强时增益已停；
5. **不是候选不够多**：top-k 与 Self-OPD Best-of-K 都警告 winner selection；
6. **确有 sequential mismatch**：local clean gain 到 exact suffix gain 的 retention 明显下降；
7. **还有未解释的 parameter gap**：必须用 Q2/gradient Gram 正面测，不能用 N2 替代。

### 目前最合理、但仍属待证假设

> DiffusionOPSD 在 query 处优化的是一个越来越差的 immediate proxy；
> 它没有对 behavior-policy suffix 的真正 action value 求导。
> 在高 reward state 附近，local reward ascent 更容易走向 dynamics 不会实现的方向，
> 然后共享 LoRA 又必须把大量这种 isolated targets 一次性 amortize。

Q-OPSD 直接替换这个 proxy，并给出与 fixed-suffix policy gradient 一阶一致的 fit direction。
它是当前**最有理论闭环、最直接解释已有数据、同时没有被现有负结果击中的方向**。

这不保证一定赢。但相比再改一个 weight/mask/normalizer，它有明确的 causal target、
可证伪的零训练 gate、清楚的最近邻 baseline，以及失败后能定位到 parameter interference 的下一层。

---

## 11. Primary sources / nearest neighbors

- DiffusionOPSD: <https://arxiv.org/abs/2608.24646>
- Self-OPD: <https://arxiv.org/abs/2608.26872>
- AdaGRPO（capability-aware prompt curriculum）: <https://arxiv.org/abs/2606.06828>
- Counterfactual Recoverability in LLM OPD: <https://arxiv.org/abs/2608.04408>
- Flow Map Reward Guidance: <https://arxiv.org/abs/2604.27147>
- DDPO（denoising as MDP）: <https://arxiv.org/abs/2305.13301>
- TDPO-R（temporal mismatch / reward overoptimization）: <https://arxiv.org/abs/2402.08552>

