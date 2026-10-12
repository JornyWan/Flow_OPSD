# 0069 — Claude 的 Q-OPSD 实现与结果审计

日期：2026-09-02（EDT）

本轮只审计已有实现、运行产物和统计解释；没有修改训练代码，也没有提交新任务。

## 0. 结论先行

当前结果能支持的最严格结论是：

> Claude 实现的“单步、镜像双目标、按 Open3 标量和在线回溯”的 QOPSD，在官方
> DrawBench/flow-40/EMA-checkpoint 评测上显著差于复现出的 DiffusionOPSD baseline；
> `d_Q` 相对同框架的 canonical-gradient 对照只带来 `+0.0028` Open3 合计，低于当前
> 评测分辨率，因而没有证明 held-out 收益。

但这**还不是对 0058 中核心 Q-OPSD 命题的干净否定**，原因不是“再调几个超参数”，而是
实际训练实现和预注册算法之间有两个实质偏差，其中一个会改变 loss 的数学语义：

1. 训练版只按三轴**标量和**选 `alpha`，没有执行 Q1b 预注册的逐轴不退化约束；
2. `y+`/`y-` 被做成严格镜像，正好与 DiffusionOPSD 的 `y_pos`/`y_neg` 镜像结构相消，
   导致正负 loss 逐样本相等，group advantage 权重 `r1` 从梯度中完全消失。

此外，Claude 所称“4.4% 泛化率”和“训练 prompt 过拟合”把多个协议变化混为一个：
训练指标与正式评测同时改变了 prompt 域、sampler/步数、每 prompt 图数和 policy 权重版本。
现有结果更准确地说是：

> `d_Q` 在 Pick-a-Pic + DPM2-10 的训练分布/协议上产生强收益，但没有迁移到
> DrawBench + flow-40 的正式协议；究竟是 prompt-domain shift、sampler-specific Q、
> EMA/behavior 差异，还是 loss 退化，现有实验不能区分。

所以不建议继续跑当前代码的更多 epoch 或 rho/alpha sweep。应先做无训练的匹配评测，再决定
是否实现一个真正“只替换 local reward 为 suffix-Q、保留官方双步构造”的 Q-OPSD-Core。

---

## 1. 我检查了什么

### 1.1 代码与配置

- `CA-OPD/code/DiffusionOPSD-rpi/scripts/train_opsd_ri_sd3.py`
- `CA-OPD/code/DiffusionOPSD-rpi/src/diffusers_patch/solver.py`
- `slurm/ab_open3.slurm`
- `slurm/opsd_eval.slurm`
- `CA-OPD/code/DiffusionOPSD-rpi/scripts/cross_eval.py`
- baseline / qopsd / qbt 的 `run_config.json`、`arm_provenance.json`、`run_attempts.jsonl`
- 三臂的完整 `metrics.jsonl`

### 1.2 结果

- `outputs/TABLE_qopsd_vs_np3_ck100.txt`
- `outputs/TABLE_qbt_vs_np3_ck100.txt`
- `outputs/TABLE_qopsd_vs_qbt_ck100.txt`
- `outputs/TABLE_basenp3_vs_np4_ck100.txt`
- ck50/ck100 的原始 aggregate eval JSON
- Gate Q1/Q1b/Q2、noise-scale、reachability 的报告与 `EXPERIMENT_LOG.md` 4.106–4.144

---

## 2. 实际实现到底做了什么

三臂共享 SD3.5-M、Open3 reward、Pick-a-Pic train prompts、10-step deterministic DPM2 rollout、
`sigma_q ~= 0.278`、`rho=0.1`、seed 20260830、world size 3、每 outer epoch 1152 张图。

### 2.1 baseline

在 query state 上定义

```text
y0 = zq - sigma_q * v_old(zq)
```

然后分别做两步 ascent/descent；每步长度为 `budget/2`，第二步在新位置重新计算
`grad_y R(D(y))`。因此有限步长下 `y+` 和 `y-` 一般不是严格镜像。

### 2.2 qopsd

代码中的 Q 梯度路径本身结构是合理的：

```text
Q_old(s,y) = R(D(F_old(s,y)))
d_Q = grad_y Q_old(s,y0)
```

`F_old` 强制 query action，使用真实 `prev_x0` 初始化 DPM2 二阶状态，再让 old adapter
走完 suffix；可微版本只保留 `forced_x0` 和后续 old-model state transition 的计算图。
当前 `kq=8/10`，所以图中只有一个后续 transformer evaluation。

随后训练实现没有保留 baseline 的两步构造，而是：

```text
候选 alpha = 1, 0.5, 0.25, 0.125
选第一个使 Open3 scalar sum(Q(y+)) > Q(y0) 的 alpha
y+ = y0 + alpha * budget * unit(d_Q)
y- = y0 - alpha * budget * unit(d_Q)
```

若所有 alpha 都不满足，就令两者均为 `y0`。

### 2.3 qbt

qbt 与 qopsd 使用相同的一步、回溯和镜像构造，只把 `d_Q` 换成
`grad_y R(D(y0))`。因此 **qopsd vs qbt 是当前“一步镜像框架内”的干净方向消融**；
但 qbt vs baseline 同时改变了两步/一步、固定步长/回溯以及非镜像/镜像，不能把其差异单独
归因于“回溯”。

---

## 3. 哪些工程工作是正确的

1. **Q gradient 确实穿过 suffix。** `_dpm2_continue_from_grad` 没有在 forced action 或后续
   old-policy transition 上 detach，求得的是 action-value gradient，不是换名字的 clean gradient。
2. **behavior policy 与 target graph 没有被 optimizer 更新。** 虽然 old-adapter 参数会出现在
   autograd graph 中，但 `autograd.grad(..., _yq)` 只取对 action 的导数，没有把 old 参数梯度写入
   optimizer；target 最终也 detach。
3. **world-size 控制补对了。** 最终 baseline_np3、qopsd、qbt 均为 nproc=3、同 seed、同数据、
   同 batch 总量，且都没有 resume/restart。
4. **baseline_np3 是有效对照。** 它相对原 np4 baseline 五轴全部打平，排除了最终 qopsd/qbt
   结果仅由 world-size 引起的解释。
5. **首版 DDP rank-divergent early break 被正确修复。** 所有 rank 用 `all_reduce(MIN)` 同步后
   决定退出，最终 job 1038580 完整跑到 100。
6. **干预不是 no-op。** qopsd 最终累计 alpha 分布约为 1.0/0.5/0.25/0.125/no-op =
   57.8%/32.9%/7.4%/1.1%/0.8%；qbt 为 54.3%/30.6%/10.2%/2.4%/2.5%。

因此，正式评测上的负结果是真实的，不应以“代码根本没跑到 Q 分支”来回避。

---

## 4. 已有结果的可靠读法

### 4.1 构造层 Gate Q1/Q1b

固定半径、同 state 的 probe 中：

| 方向 | suffix reward mean gain | p10 | negative rate |
|---|---:|---:|---:|
| clean | +0.06092 | -0.01178 | 13.3% |
| Q | **+0.07953** | -0.00317 | **10.5%** |
| random | -0.04550 | -0.09721 | 90.5% |

所以 `d_Q` 在 exact training suffix 上明显包含真信息：均值比 clean 高 30.6%，三个 Open3
分量都显著更高，且与 clean 方向的中位 cosine 只有 0.0846。

离线 Q1b 再用逐轴约束选择 alpha 后，报告的 negative rate 为 0%、p10 为 +0.050、
no-op 1.2%。但这只证明该批 training-suffix state 上能选出更高 reward 的局部 target，
没有证明蒸馏后会泛化。

### 4.2 训练曲线

全 100 epoch、按 Claude 的 lag-1 autocorrelation 修正：

| 对比 | mean delta reward_avg | corrected t |
|---|---:|---:|
| qopsd - baseline_np3 | **+0.05140** | +23.49 |
| qbt - baseline_np3 | **-0.01202** | -4.22 |
| qopsd - qbt | **+0.06342** | +11.71 |

qopsd 相对 qbt 的三个训练 reward 分量全部为正：Pick +0.02573、CLIP +0.01657、
HPS +0.02112。也就是说，训练协议内的强 gain 不是某一轴换来的。

### 4.3 正式 DrawBench/flow-40 评测

ck100：

| arm | Pick | CLIP | HPS | Aes | ImgR | Open3 sum |
|---|---:|---:|---:|---:|---:|---:|
| baseline_np3 | 24.6617 | 0.3160 | 0.3702 | 5.9021 | 1.4329 | **1.634711** |
| qopsd | 24.3482 | 0.3117 | 0.3561 | 5.9052 | 1.3745 | **1.604228** |
| qbt | 24.3218 | 0.3074 | 0.3586 | 5.9980 | 1.3694 | **1.601453** |

- qopsd - baseline：Open3 sum **-0.03048**；Pick、HPS 显著更差；CLIP 也为负但未同时越过
  项目设定的双阈值。
- qbt - baseline：Open3 sum **-0.03326**；Pick、CLIP、HPS 显著更差，Aes 显著更好。
- qopsd - qbt：Open3 sum **+0.00278**，低于现有 MDE 0.0101；四轴打平，Aes 更差。

ck50 已出现相同方向：

| 对比 | ck50 Open3 sum delta | ck100 delta |
|---|---:|---:|
| qopsd - baseline | -0.01397 | **-0.03048** |
| qbt - baseline | -0.01813 | **-0.03326** |
| qopsd - qbt | +0.00416 | +0.00278 |

所以不是 ck100 的单点偶然翻转；两种一步镜像方法相对 baseline 的 gap 随训练扩大，而
`d_Q` 对 qbt 的 DrawBench 净贡献始终很小。

---

## 5. 关键实现问题一：严格镜像使 advantage 从 loss 中消失

这是本次审计最重要的发现。

令当前 policy 相对 old policy 的 query clean-output 改变量为 `delta`。代码定义：

```text
y_pos = y0 + beta * delta
y_neg = y0 - beta * delta
```

Claude 的一步构造严格定义：

```text
y_plus  = y0 + a*d
y_minus = y0 - a*d
```

于是：

```text
y_pos - y_plus   =  beta*delta - a*d
y_neg - y_minus  = -beta*delta + a*d = -(beta*delta - a*d)
```

平方误差相同，逐像素绝对值 normalizer 也相同，所以不仅两个 loss 数值相同，它们对参数的
梯度也相同：

```text
L_pos(theta) == L_neg(theta)
r1*L_pos + (1-r1)*L_neg == L_pos
```

这使来自每 prompt 24 个 branch return 的 group advantage `r1` **完全不起作用**。
当前方法已从“one-policy self-distillation”退化为对所有 sampled state 做无条件 reward-ascent
direction regression。

日志直接验证了代数：

| arm | mean abs(L_pos-L_neg) | max abs(L_pos-L_neg) |
|---|---:|---:|
| baseline_np3 | 1.44e-3 | 3.19e-3 |
| qopsd | **3.50e-7** | **1.15e-6** |
| qbt | **2.48e-7** | **1.07e-6** |

baseline 的两步会在第二步分别于 `y0+h*g` 和 `y0-h*g` 重算梯度，reward curvature 打破严格
镜像，因此没有精确退化。qopsd/qbt 的共同负结果首先指向**一步严格镜像**，而不是足以单独
证明“online backtracking 有害”。

这也解释了为什么 qopsd/qbt 的 `opa_pos_loss` 和 `opa_neg_loss` 在 metrics 中几乎逐位重合，
而 Claude 此前只比较其量级，没有检查二者差值是否退化为零。

---

## 6. 关键实现问题二：训练版不是通过 Q1b 的那个回溯

0058/0061 的定义是：

```text
scalar gain > 0
AND 每个 rollout-level reward axis gain >= 0
```

实际训练代码则只做：

```python
_r0q = _reward_of_latents_grad(...)
_rj  = _reward_of_latents_grad(...)
_ok  = (~_done) & (_rj > _r0q)
```

这里 `_r0q/_rj` 已经是 Pick/26 + CLIP + HPS 的标量和；而 `_NEEDS_AXIS_SCORES` 也没有包含
`_QOPSD_ON`。因此训练允许某一轴下降、由另两轴补偿。首 epoch alpha=1 接受率从 Q1b 的约
69% 升到在线版的约 91%，Claude 自己在 4.109 中也记录了这是“逐轴 vs 聚合”口径差异。

更进一步：同一个 `alpha` 只用正候选 `y+` 的 scalar improvement 选出，随后直接镜像生成
`y-`；代码没有验证 `Q(y-) < Q(y0)`，更没有验证三个轴在负方向上的行为。也就是说：

- 正 target 不满足预注册的 Pareto/no-regression 条件；
- 负 target 是否真是 suffix-level worse action 完全未测；
- Q1b 的 0% negative rate 和 1.2% no-op 不能直接移植为训练实现的证据。

这不是小注释错误，而是“通过 gate 的 target constructor”和“真正训练的 constructor”不同。

---

## 7. Claude 对泛化的解释过强

Claude 把训练 `reward_avg` 与 eval Open3 sum 视为“同一个统计量”，然后把
`0.0028/0.0634=4.4%` 叫作 `d_Q` 泛化比例。两边确实使用相同的 reward scalarization，
但生成分布和被评估权重不同：

| 维度 | 训练 reward_avg | 正式评测 |
|---|---|---|
| prompts | Pick-a-Pic train | DrawBench |
| rollout | deterministic DPM2, 10 steps | flow sampler, 40 steps |
| images/prompt | 24 | 1 |
| weights | outer-loop `old` behavior adapter | checkpoint root 的 EMA/default adapter |
| aggregation | 每 epoch 48 prompts、1152 images | 固定 1000 prompts、1000 images |

Q 的特殊之处正是包含**特定 suffix solver 的 Jacobian**。因此 DPM2-10 上更正确的 `d_Q`
不一定对 flow-40 仍是正确方向；把全部 gap 叫作 prompt overfit，反而忽略了 Q 方法最需要检查的
sampler transfer。

“memorize 训练 prompt”本身也与已有 sampling 事实不吻合。按固定 seed 和 sampler 精确重建：

- 100 epoch 共抽 4800 个 prompt slot；
- 其中 4394 个是不同 prompt，只有 406 次重复；
- 25415 条 Pick-a-Pic train prompts 中有 **21021 条从未用于训练**。

每个 epoch 的 reward 又是在该 epoch optimizer update 之前记录，所以 qopsd 在训练曲线上的强
优势大部分来自对新的 Pick-a-Pic prompt 的**域内迁移**，不能简单称为逐 prompt 记忆。

正确表述应是：

> qopsd 的收益在 Pick-a-Pic/DPM2-10/behavior-policy 协议内稳定出现，但没有迁移到
> DrawBench/flow-40/EMA 协议。现有数据不能分解 domain、sampler 和 adapter 三项。

---

## 8. 统计与 provenance 的次要限制

这些问题不会抹掉 Pick/HPS 的明显负效应，但会影响精确 t 值和“显著”的措辞。

### 8.1 当前不是 paired per-prompt 检验

`cross_eval.py` 已支持 `--dump_per_image`，但 `opsd_eval.slurm` 没有打开。现有 JSON 只有各臂
均值和单臂 SE。`ab_compare.py` 使用 `sqrt(SE_base^2+SE_arm^2)`，并没有利用相同 prompt、
相同 latent seed 的逐 prompt 差值；文档中的“paired SE”说法不准确。

### 8.2 Open3 sum 的 SE 忽略轴间 covariance

`make_report.py` 用三个 SE 的平方和开根号构造 sum SE，没有 Pick/CLIP/HPS 的同图 covariance。
所以 Open3 sum 的 t 不是完整估计。下一次保存 per-image scores 后，应直接对
`pick_i/26+clip_i+hps_i` 的逐图 paired difference 做 bootstrap/t-test。

### 8.3 只有一个训练 seed

np3-vs-np4 和 certp100 控制说明当前 pipeline 很稳定，但它们不是独立训练 seed 的方法复现。
效应很大的 qopsd-vs-baseline DrawBench 负差大概率真实；`+0.0028` 这种小的 qopsd-vs-qbt
差值则不能当作稳定正效应。

### 8.4 代码版本没有被完整冻结

`run_config.json` 中 `git_commit="unknown"`，qopsd 的 `code_variant` 仍写
`diffusionnft_baseline`；环境式 Q flags 不进 run_config。较新的 `arm_provenance.json` 能记录
`RPI_QOPSD_DIR`，但 qopsd 运行时的旧格式没有该字段，而且 `RPI_QOPSD_ALPHAS` 从未被 preflight
检查或落盘。日志足以确认本次 qopsd 使用默认 Q direction 和四个 alpha，但论文级复现还需要
source SHA256 + 完整 flag manifest。

---

## 9. 现在最有信息量的下一步：先不训练，做匹配评测

### 9.1 第一关：同协议 ID held-out

从 21021 条从未训练过的 Pick-a-Pic prompts 中按冻结规则选 1000 条，评估
baseline_np3/qbt/qopsd 的 ck100 **old adapter**，使用与训练完全一致的 DPM2-10、noise 0.7、
固定 paired latent，并打开 `--dump_per_image`。

这一步回答：训练曲线的 +0.0634 是否能在“未见 Pick-a-Pic prompt + 同 suffix”上复现。

### 9.2 若第一关为正，再做两个单因素迁移

| prompts | sampler | 回答的问题 |
|---|---|---|
| unseen Pick-a-Pic | DPM2-10 | 完全匹配的 ID generalization |
| DrawBench | DPM2-10 | 只换 prompt domain |
| unseen Pick-a-Pic | flow-40 | 只换 sampler/steps |
| DrawBench | flow-40 | 当前正式结果；两者都换 |

在 old adapter 上先完成 2x2；随后只需在关键 cell 比较 root EMA/default adapter，量化
behavior-to-checkpoint gap。判读：

- 只在 Pick-a-Pic/DPM2-10 赢：domain 与 sampler 都可能限制，当前 Q 太 protocol-specific；
- 两个 prompt 集的 DPM2-10 都赢、flow-40 都不赢：主要是 suffix/sampler mismatch；
- Pick-a-Pic 的两个 sampler 都赢、DrawBench 都不赢：主要是 prompt-domain shift；
- 完全匹配 cell 也不赢：训练 `reward_avg` 的 ensemble/adapter/timing 解释仍有遗漏；
- 四个 cell 都赢但 EMA 不赢：checkpoint/EMA 与 behavior policy 的不一致是主因。

这比再跑一条 100-epoch 训练更便宜，也直接决定 Q-OPSD 下一版该修什么。

### 9.3 评测必须同步修正的记录

1. 每张图保存 prompt index、三个 Open3 axis 和两个额外轴；
2. 对相同 prompt/latent 做 paired difference；
3. Open3 sum 直接逐图相加，保留 covariance；
4. 至少报 mean、median、bootstrap CI、win rate；
5. prompt manifest、source hash、sampler、adapter name、checkpoint hash 全落盘。

---

## 10. 若匹配评测支持 d_Q：下一版应是 Q-OPSD-Core，不是修补当前回溯

最干净的训练实验不是把当前 scalar backtracking 改成 per-axis 后直接长跑，而是只替换
DiffusionOPSD 的核心 oracle，保留原有 constructor：

```text
baseline:
    两步 ascent/descent，每步重新计算 grad R(D(y))

Q-OPSD-Core:
    两步 ascent/descent，每步重新计算 grad R(D(F_old(s,y)))

其他全部相同：
    同 step_len=budget/2、同 trust ball、无 backtracking、同 dual loss、同 advantage、同 seed
```

这样唯一的因果差异才是 0058 的核心：

```text
grad_y R(D(y))  ->  grad_y R(D(F_old(s,y)))
```

而且第二步会分别在正、负分支的新位置重算 Q gradient，reward curvature 会打破严格镜像，避免
当前 `L_pos == L_neg` 的精确退化。

在任何正式训练前先做一个 1-epoch implementation gate：

- `mean/max abs(L_pos-L_neg)` 必须明显大于 fp16 数值误差；
- shuffle `r1` 后总梯度必须改变，否则 advantage 仍未进入训练；
- 分别记录 `Q(y+)-Q(y0)` 与 `Q(y-)-Q(y0)`，负支确实应为负；
- 记录三个 axis 的 gain，但**只诊断、不用同 batch winner selection 改训练路径**；
- QCore vs baseline 的代码 diff 必须只包含 Q evaluator replacement 与必需的 suffix state。

若 QCore 在 matched ID 评测赢、但 flow-40 不赢，下一篇真正有方法价值的方向不是调 alpha，
而是 **sampler-robust / multi-suffix Q-OPSD**：训练时对多个 suffix discretization 的 action value
求期望或随机化 solver，使蒸馏方向不绑定 DPM2-10。这个问题是 Q 方法独有且第一性原理上明确，
比“再加一个 veto/normalizer”更像可发表的方法贡献。

---

## 11. 最终判断

### 已被否定的

- 当前“一步严格镜像 + scalar online backtracking”的实现不能作为论文主方法；
- 它在正式 DrawBench/flow-40 上比 own-run DiffusionOPSD baseline 更差；
- 当前框架下，`d_Q` 相对 qbt 没有可检测的 held-out 改善。

### 尚未被否定的

- exact suffix action-value gradient 比 local clean gradient 更符合序列决策对象；
- Gate Q1 已证明它在 training suffix 上显著旋转并改善方向；
- 保留官方两步/advantage 语义的 Q-OPSD-Core 尚未被实际实现或训练；
- 当前正式负结果究竟来自 sampler mismatch 还是 prompt-domain shift 尚未测。

### 推荐顺序

1. **先做无训练的 matched 2x2 evaluation**；
2. matched ID 若不赢，停止 Q 主线；
3. matched ID 若赢，按诊断结果决定做 eval-sampler-consistent Q，还是实现两步的 Q-OPSD-Core；
4. 不再为当前版本追加 epoch、rho、alpha 或更多 binary gates。

当前正在运行的 `base200_np3` 与 `inner2` 是 baseline convergence / fitting-side 实验，继续完成
有价值，但它们不会自动修复本报告指出的 Q target/loss 语义问题。
