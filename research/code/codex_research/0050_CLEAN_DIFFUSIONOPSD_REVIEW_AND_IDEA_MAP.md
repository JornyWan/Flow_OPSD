# 0050 Claude 最新整理审计：clean DiffusionOPSD 主线与下一代 idea map

**日期：2026-08-31（审计时刻约 11:25 EDT）**  
**范围：** `EXPERIMENT_LOG.md`、`SUMMARY_20260831_morning.md`、0033–0049、pristine/fork diff、
baseline/A1/N1/N2 manifests、metrics 与 Slurm 状态。  
**状态：** 只读审计与 brainstorm；本条未修改训练代码、未提交新任务。

## 0. 总结判断

Claude 的最新重整方向是对的，而且相较早期项目已经干净很多：

```text
official DiffusionOPSD @ commit 9c4c48c
          ↓ pristine 只读固化
our fork（默认 flag 全关时 canonical）
          ↓ paired fixed-seed A/B
baseline vs 单变量干预
          ↓ 同一 held-out protocol
```

现在可以把当前工程称为**可信的 method-development harness**。但科学解释仍有三处需要降级：

1. Gate A 的 `+0.7149` 是跨 state 的 **reward-gain correlation**，不是 reward-gradient cosine/Gram；
2. N2 的齐次性推导正确，但它不只是“改变步长”，还把近似 robust/L1 fitting 改成了
   global-EMA-scaled MSE，重新分配了 sample/timestep/outlier 权重；
3. 现有 `1–3% realization rate` 是有价值的 diagnostic ratio，但 construction gain 与训练轨迹 gain
   不是同一个随机变量，不能表述成精确的“97–99% loss 被某一项吃掉”。

因此当前最准确的说法是：

> **target construction 在 Open3 三轴上通常同向，但轴间相对 headroom 不平衡；静态全局权重会沿
> trade-off surface 滑动；finite fitting 明显 under-realize target-level gain，但原因尚未被隔离。**

---

## 1. Baseline 是否足够干净

### 1.1 代码与协议：通过

- pristine 固定在 official commit `9c4c48cdfe5c3fa712d32957345d75fae9295696`；
- fork 的计算相关 source diff 只有：
  - `scripts/train_opsd_ri_sd3.py`：Gate A/A1/N2，均由 env flag 守卫；
  - `config/public.py`、`config/mixed.py`：固定 seed hook；
- baseline 明确 `RPI_A1=0`、`RPI_N2=0`、`RPI_GATEA_PROBE=0`、`OPSD_WF_MODE=adaptive`；
- 数据 manifest 固定为 Pick-a-Pic train 25,415 / test 1,000，md5 已检查；
- A/B baseline、A1、N1 均为 world size 4、seed 20260830、100 updates、无 restart；
- baseline/A1 甚至落在同一节点 `ai-gpu17`，是很好的 paired control。

### 1.2 数值复现：应分两层描述

**可以说：reward-specific baseline/protocol 复现可行。**

- Pick ck100：25.2133 vs paper 24.940；
- HPS ck100：0.3928 vs paper 0.390；
- CLIP ck50：0.3105 vs paper 0.3122±0.0050；
- base evaluation chain 对齐。

**暂时不能说：published Open3 已完整复现。** 论文 Open3 是 300 updates：
`25.51 / 0.333 / 0.389`；当前 official-style Open3 在 ck100 停止。100-update run 是完全合法的
controlled A/B baseline，但不是 300-update published-point reproduction。等某个改动在 100 updates 有明确
signal 后，再把 baseline/ours 都延伸到 300；现在无需先消耗这笔算力。

---

## 2. 最新各臂的真实状态

### 2.1 Paired baseline @ ck100

| Pick | CLIP | HPS | Aes | ImgR |
|---:|---:|---:|---:|---:|
| 24.6588 | 0.3165 | 0.3704 | 5.8958 | 1.4331 |

三 objective 相对 base 都大幅提升，且最低 local-specialist retention 为 HPS 87.9%。这说明 Open3
baseline 已经很强，ours 必须改善最弱轴或改善 Pareto balance，不能只提高平均和。

### 2.2 A1：训练完成，held-out 尚未完成；机制已经基本判空

审计末次刷新时，原评测 job `1038006` 在未开始运行的情况下被用户账号取消，尚无 A1 ck100
held-out JSON。需要之后重新提交/恢复评测，但不需要重跑训练。

- Gate A 预测 VFR：0.9935；
- 训练中实际可行率均值：约 0.9925；全量 probe：0.9861；
- 只 veto 约 1.39% state；正 loss 机械下降约 1.9%；
- paired `reward_avg` 只差约 0.06%，公共窗口实现斜率 t≈0.07；
- 30 updates 后 veto rate 无可信增长趋势。

所以 A1 若 held-out 打平，解释已经确定：**干预极少发生**，不是 baseline 饱和或训练不够长。
不应跑 A1 300 updates。

### 2.3 N1：明确负结果，但机制表述需要修正

| 轴 | baseline | N1 | 差 |
|---|---:|---:|---:|
| Pick | 24.6588 | 24.9864 | +0.3277 |
| CLIP | 0.3165 | 0.2996 | −0.0169 |
| HPS | 0.3704 | 0.3622 | −0.0082 |

最低 specialist retention `87.9% → 75.6%`，所以“静态尺度均衡权重”判负成立。

但下面这句当前证据**不支持**：

```text
hps–pick gradient correlation = +0.7149，因此应解 G w ∝ r0
```

Gate A 存的是 `ΔR_k(y_plus,y0)`，`+0.7149` 是这些 scalar gains 在样本间的相关性；它不是
`cos(g_hps,g_pick)`，更不是 parameter-space 或 image-space gradient Gram。N1 只证明：

> 一组由 diagonal response heuristic 得到的全局静态权重，把 policy 推向 Pick-favoring trade-off。

要讨论 `G`，必须在同一个 `y0` 上实际保存
`g_k=∂R_k/∂y` 的 norms/cosines，并区分 raw gradient 与 normalized target direction。

### 2.4 N2：正在运行，干预确实有“行程”，尚无结果

审计时 job `1037627` 在 `ai-gpu17` 运行，无 restart。epoch 0：

| diagnostic | positive | negative |
|---|---:|---:|
| `wf_cv` | 0.1350 | 0.1398 |
| `wf_drift` | 1.0167 | 1.0107 |

按 0049 预注册，CV 明显高于 0.05，故 N2 不是数值上的 no-op；但只有一个 epoch，不能读趋势。

---

## 3. 对 N2 的代码/理论审计

### 3.1 正确的部分

canonical：

```text
L(r) = mean(r²) / mean(|r|)
```

normalizer detached 时，对同一 residual 全局缩放 `r→cr`，`∇_r L` 为 0-homogeneous。N2 用跨 rank
平均、时间 EMA 的 scalar denominator 后，loss 对当前 residual 变 2-homogeneous、gradient 变
1-homogeneous。解析和数值检查都成立。

### 3.2 需要改写的解释

`opa_delta_res` 在 N1 权重变化后仍约 0.071，并不能证明 fitting normalizer 抹掉了 target distance。
DiffusionOPSD target constructor 本身就：

1. 把 aggregate reward gradient normalize；
2. 用固定 target step；
3. 投影进固定 trust ball。

因此权重首先改变**方向**，而 target norm 本来就应接近固定。这是 construction 的设计结果。

此外，N2 不只是纯 step-size intervention：

- canonical 每个样本用自己的 `mean|r_i|`，近似按 residual magnitude 做 robust normalization；
- N2 所有样本共享 EMA scalar，大 residual sample 获得更大梯度，小 residual sample 获得更小梯度；
- 不同 query/timestep/prompt 的相对权重随之改变。

若 N2 成功，应写成：

> global temporal residual calibration improves target fitting / realization

而不是直接写成：

> trust-region step magnitude restored, proving the realization gap was caused by scale cancellation

除非再做 per-sample multiplier、timestep 与 residual-quantile analysis。

### 3.3 与旧实验的关系

项目在 2026-08-28 已跑过 `OPSD_WF_MODE=frozen`：首 batch 常数 normalizer，单 Pick、100 updates。
它与 N2 共享“去掉即时 per-sample normalizer”的核心思想，所以 N2 不是概念上第一次出现。

但 N2 仍是必要的干净复核，因为它有四个关键差别：

- official pinned fork；
- paired seed；
- Open3 而非 single Pick；
- cross-rank global EMA，而非各 rank/首 batch 永久常数。

因此旧 null 不应提前杀掉 N2；但若 N2 成功，novelty 不能只靠“冻结 denominator”，必须提升为
有控制、有界的 realization-aware fitting framework。

### 3.4 一个 provenance 小洞

N2 的 Slurm 日志明确写了 `RPI_N2=1`，metrics 也出现 `n2_wf_*`，所以本次 run 身份无疑；但审计时
`outputs/ab_open3_n2_s20260830_u100/arm_provenance.json` 不存在。原因很可能是文件在 trainer 初始化
save directory 时被清掉。不要改正在运行的 fork；任务结束后应从 Slurm 日志回填，并让以后 provenance
在 trainer 初始化之后写入，或直接纳入一个不会被清理的 immutable launch manifest。

---

## 4. 最推荐的新 idea：Local Specialist-Retention Target Portfolio

简称暂定 **LSR-OPSD**。它完全从 official DiffusionOPSD target constructor 出发，不引入 Self-OPD
trajectory branches。

### 4.1 动机

现有证据同时说明：

- official Open3 direction 在约 99% state 上三轴都为正，binary veto 没有分辨率；
- 三轴 relative target gains 严重不均；
- 全局静态 reweighting 会把整个 run 推到另一个 trade-off 点；
- 问题应在每个 state 的局部 geometry 上解决，而不是猜一组全局 weights。

### 4.2 方法

对同一 on-policy query `y0`，分别计算：

```text
g_pick = ∂R_pick/∂y0
g_clip = ∂R_clip/∂y0
g_hps  = ∂R_hps /∂y0
```

构造一个很小的 direction portfolio：

```text
u0 = official normalize(g_pick + g_clip + g_hps)
u1,u2,u3 = normalize(g_pick), normalize(g_clip), normalize(g_hps)
u4,u5,u6 = three pairwise normalized sums
```

对每个 candidate `u_j` 构造同半径 target `y_j`，精确前向得到
`ΔR_{j,k}=R_k(y_j)-R_k(y0)`。

不要除以 raw `R_k(y0)`。用每一轴自己的 local specialist headroom：

```text
H_k = max_j ΔR_{j,k}          # 该 state 上该 reward 在 portfolio 内能达到的局部增益
retention_{j,k} = ΔR_{j,k} / (H_k + eps)
j* = argmax_j min_k retention_{j,k}
```

并要求 `ΔR_{j*,k} ≥ -τ_k`。这等价于：

> 每个 query 都选择保留最多“局部单轴 specialist improvement”的 Pareto direction。

它自然消除了 N1 的 reward 单位问题，不需要假装 `0.01 CLIP == 0.01 HPS == 0.01 Pick`；也比 A1
只判断正负强得多。

### 4.3 为什么它比 `G w ∝ r0` 更适合作为下一步

- 使用 target-level exact reward change，不把 gain correlation 误当 gradient Gram；
- state-adaptive，能处理 prompt/query-dependent geometry；
- candidate set 包含 official direction，最坏可回到 baseline；
- 可先做零训练 oracle probe，25–60 分钟即可判断是否真的有改进空间；
- 若 probe 有 signal，训练改动只发生在 target selection，canonical fitting loss先保持不变，归因清楚。

### 4.4 Gate P：训练前必须通过

在与 Gate A 同规模的 frozen states 上比较 official candidate `u0` 与 LSR selector：

1. 至少 20% states 选择非 `u0` candidate；
2. median worst-axis local retention 相对 `u0` 提高至少 15%；
3. mean scalar Open3 target gain 不下降超过 5%；
4. 三轴负 gain rate不高于 official；
5. 换一次 target-evaluation seed/precision 后，选择稳定率至少 70%。

Gate P 不过就不训练。阈值应在运行前正式预注册；以上只是本次 brainstorm 建议值。

### 4.5 必需 baseline/ablation

```text
B0 official Open3
B1 N1 static weights（已有负结果）
B2 portfolio + scalar-sum selection
B3 portfolio + local-specialist max-min selection（LSR-OPSD）
B4 B3 without pairwise candidates
```

这样能区分收益来自“更多 candidate”，还是来自 local specialist normalization/max-min rule。

---

## 5. 其他值得保留的 brainstorm（按优先级）

### Idea 2：Bounded Homogeneity-Controlled Fitting

如果 N2 有训练/held-out 正 signal，把 binary `adaptive vs global EMA` 推广为：

```text
denom_i(α) = gamma_i^α * EMA(gamma)^(1-α),  α∈[0,1]
```

- `α=1`：official adaptive normalizer；
- `α=0`：N2；
- 中间 α：保留部分 robust normalization，同时恢复部分 magnitude sensitivity。

再对 `gamma_i/EMA(gamma)` 做有界 clip，避免 MSE outlier domination。不要立即 sweep；先等 N2，若
N2 只提高 train reward、不提高 held-out，就不继续该线。

### Idea 3：Realization-aware step portfolio，而非 binary post-fit gate

A2 的 binary accept/reject 可能和 A1 一样几乎不触发，因为 aggregate realized gain 平均为正。更有
信息量的版本是对 actual optimizer candidate 评估 `α∈{0.5,1,2}` 三个 step scale，选择最大化
held-out mini-gate 的 worst normalized improvement；无安全 candidate 才 no-op。

这直接优化 finite realization，但需要复制 AdamW state/参数并避免 gate overfit，工程和计算都更重，
排在 LSR 后面。

### Idea 4：Realization-stratified multi-query OPSD

当前只在低噪声 `σ≈0.278` 拟合 target。模型必须把一个 query 的监督 amortize 到整条 flow，可能是
construction→realization gap 的来源之一。先做 probe：在多个 sigma 上测 same-query exact one-update
realization；若 realized gain 明显随 sigma 变化，再按可实现率分配 query，而不是盲目多 query。

旧 weighted-MQ run 只到约 12 updates，属于未检验，不是负结果；但应在 LSR/N2 后排序。

### Idea 5：Prompt-conditioned/purified reward residual

只在发现 Pick/HPS/CLIP 对 held-out ImgR 或多样性产生系统性损害时考虑：

```text
g_transfer(y,c) = ∇R(y,c) - ∇R(y,c_shuffled/unconditional)
```

它可去掉 prompt-independent shortcut，但当前 Open3 对 objective 三轴和 ImgR 总体都强，没有足够证据
把 reward hacking 设为首要瓶颈。

---

## 6. 立即行动建议

1. **不碰正在运行的 N2 fork。** 等其 100 updates + held-out；先按 0049 原判据判断。
2. **重新安排 A1 ck100 held-out，完成后封存 A1。** 预期打平；无需重跑训练，无论结果如何都不延长到 300。
3. **N2 结束后补 provenance。** 同时报告 multiplier/CV/drift，而不只报告 reward slope。
4. **不要现在提交 Gram-weight N3。** 先实际测 per-reward gradient norm/cosine；当前 `+0.7149` 不能支持 Gram 结论。
5. **下一项代码工作做 Gate P 零训练 probe。** 先验证 LSR portfolio 是否相对 official target 有局部
   Pareto headroom，再决定是否投 100-update run。
6. 任一新方法在 100 updates 明确胜出后，再做第二 seed 和 300-update official comparison；此前不扩张。

## 7. 最终优先级

```text
当前：等待 A1 held-out + N2 完整结果
  ↓
下一主 idea：LSR-OPSD（local specialist-retention target portfolio）
  ↓ 若 N2 正
bounded homogeneity-controlled fitting
  ↓ 若 target 好但 realization 仍差
actual optimizer step portfolio
  ↓ 后续分析
multi-query / purified reward
```

我目前对 **LSR-OPSD** 的信心高于 A1、N1 和单纯 Gram reweighting，因为它正面利用了已有的两个
最稳定事实：official direction通常三轴同升，但局部相对 headroom 不平衡；而这种不平衡是
state-dependent 的，不能用一组全局 scalar weights 修好。
