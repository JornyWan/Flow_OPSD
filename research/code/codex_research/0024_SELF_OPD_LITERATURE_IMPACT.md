# 0024 Self-OPD 新论文复核与对本项目的影响（2026-08-30）

## 结论先行

用户提到的论文准确名称是 **Self-OPD: On-Policy Distillation for Flow Matching Models
without Teacher**，不是 Self-OPSD。arXiv:2608.26872v1，2026-08-27 发布，作者来自
清华、浙大与阿里巴巴；公开仓库为 <https://github.com/Shiy-Zhang/Self-OPD>。

它与本项目**高度相关，而且从现在起必须是主要 baseline/related work**。它已经覆盖了以下
可能被我们误当成新贡献的表述：

- teacher-free dense step-wise supervision；
- 从当前 student 的 on-policy trajectory 自探索出正/负 target；
- reward-level multi-objective fusion；
- 避免多 teacher 的 field/parameter-gradient conflict。

但它**没有覆盖我们真正仍有空间的核心问题**：从一个已经具备多能力的发布 Student 出发，
顺序加入一个新目标时，如何对指定旧能力施加显式 retention budget，并对实际 optimizer update
进行 certificate。Self-OPD 是 joint reward maximization；当前 Guarded 路线是 continual
specialization under capability constraints。这两者相近但不等价。

它自己的结果还暴露了和我们相同的现象：mixed Self-OPD 在 OCR/GenEval task images 上很强，
但在 held-out preference prompts 上仍低于 DiffusionOPD。因此“优化到的联合 reward”与
“广义旧能力/分布外质量保持”之间仍有缺口。这正是我们最值得保留的切入口。

---

## 1. 方法到底做了什么

在当前 on-policy latent `x_j` 上：

1. student 给出确定性下一状态均值 `μ_θ`；
2. 围绕 `μ_θ` 采样 `K=8` 个局部 SDE branches；
3. 每个 branch 都用当前 student 的 ODE sampler rollout 到图像；
4. 同一个 parent state 还生成一条 deterministic ODE self-reference；
5. 用最终 reward 形成
   `A_k = (r_k - r_ode) / std(r_1,...,r_K)`；
6. `A_k>0` 的 branch 用 MSE 拉近 student prediction，`A_k<0` 的 branch 用负 MSE 推远；
7. negative branch 若与 best branch 方向接近，就用 direction-aware coefficient 衰减 repulsion；
8. loss 除以 SDE transition variance，并偏重 early/mid timestep。

因此它本质上是**局部 derivative-free reward search + advantage-weighted transition regression**。
“distillation”target 不是外部 teacher 生成的，而是 student 周围采样出的具体高/低 reward
transition。

### 与 DiffusionOPSD 的区别

- DiffusionOPSD：需要 reward 对 clean prediction 可微；用 reward gradient 显式构造 bounded
  positive/negative clean-output targets。
- Self-OPD：reward 可以是 black-box；用 `K` 个随机 SDE branch 的 rollout reward 形成 target，
  不反传 reward gradient。

### 与 DiffusionOPD/MOPD 的区别

- DiffusionOPD：先训练各能力 specialist teacher，再把 teacher field 蒸馏到 student。
- Self-OPD：没有 specialist teacher LoRA；但训练时要反复运行任务 reward models，并承担
  `K` 个 branch 加 self-reference 的 rollout 成本。

它是 teacher-free，不是 supervision-free，也不是 compute-free。

---

## 2. Multi-objective 部分为什么与我们尤其相关

论文不是分别反传多个 reward loss，而是先把每个 branch 的各 reward 做标准化，再得到复合
scalar score，用这个 score 对完整 trajectory branch 排序。其核心直觉是：

- field-level fusion 在参数空间混合冲突方向；
- reward-level fusion 在 trajectory space 里寻找一个同时满足多个 reward 的具体样本；
- 最终回归的仍是一个 branch transition，而不是多个 reward gradient 的和。

论文设置中，OCR task 使用 `OCR:PickScore:HPSv2 = 3:1:1`，GenEval task 同理把 OCR 换成
GenEval；PickScore/HPS 相当于 quality guards。这个设计与我们观察到 `crg` 提高 Pick/OCR/GE、
却损失 HPS 的问题直接相关：**Self-OPD 已经把 quality reward 放到了 task image 本身上。**

但它仍是 weighted scalarization，论文另提 per-scorer veto；没有给出相对初始 Student 的硬
budget，也没有 exact post-update certificate。

---

## 3. 论文主结果：强，但并非所有维度都支配

SD3.5-M mixed-reward Table 2：

| 方法 | GE strict | GE cont. | OCR | same-image Pick | same-image HPS | held-out Pick | held-out HPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flow-OPD | 0.8594 | 0.9203 | 0.9392 | 23.43 | 0.3042 | 23.13 | 0.3334 |
| DiffusionOPD | 0.9150 | 0.9479 | 0.9464 | 22.72 | 0.2676 | **23.95** | **0.3729** |
| DiffusionNFT | 0.8888 | 0.9277 | 0.9229 | 23.67 | 0.3206 | 23.18 | 0.3482 |
| **Self-OPD** | **0.9521** | **0.9691** | **0.9597** | **23.87** | **0.3214** | 23.39 | 0.3415 |

正确读法：

- Self-OPD 在目标任务与同一批 task images 的综合质量上非常强，明显胜过 DiffusionOPD；
- 但在 held-out DrawBench/Pick-a-Pic 上，DiffusionOPD 的 Pick/HPS 分别高 `+0.56/+0.0314`；
- Self-OPD 也不是在所有轴上支配 DiffusionNFT（held-out HPS 低 0.0067）；
- 所以它解决的是 joint targeted alignment，不足以证明 broad retention。

论文只报 point estimates，主表没有独立 seed 的 error bars。我们应把其数值视为重要报告结果，
但自己的主表仍必须 own-run、same-protocol、multi-seed。

---

## 4. 截至 2026-08-30 的公开复现状态

审计的 GitHub HEAD：

```text
e635befaf41192d6b191bf981f239cb259470436
commit date: 2026-08-28 17:01:39 +08:00
```

公开 release 当前不够复现 Table 2：

1. `README.md` 明确说目前只提供 OCR 作为第一个 public reward example，其他 reward config
   和 checkpoints 后续逐步发布。
2. `configs/self_opd.py` 只有一个 text-file dataset 和 `reward_fn={"ocr":1.0}`。
3. `self_opd/rewards.py` 只内置注册了 OCR；PickScore/HPS/GenEval 训练 reward module 未发布。
4. 论文 Eq. 18 描述逐 scorer z-normalization，并称有 per-scorer veto；当前
   `select_all_branches()` 只是对 raw scorer values 固定加权，然后只对合成 reward 做 branch
   std normalization，没有看到逐 scorer z-score 或 veto。
5. 当前 Hugging Face model repo API 只列出 `.gitattributes` 与 `README.md`，没有 LoRA 权重，
   与 GitHub README 的下载示例暂不一致。
6. 当前 DataLoader 使用 `num_workers=1`；本项目已经实测 PaddleOCR 与 forked DataLoader 在集群
   上不安全，迁移时必须固定为 0。
7. v1 正文与 Table 1 的 single-reward 数字有内部不一致：正文称 PickScore 24.79、HPSv2
   0.3665，而表中对应 best specialist 为 24.47、0.4099。正式引用必须以表格/后续修订为准。

理论部分也应保守理解：reward-level fusion 避免的是“直接相加多个可微 reward gradient”，
并没有消灭多目标之间真实的 Pareto 冲突；若有限 `K` 个 branch 中不存在同时改善所有 reward
的样本，weighted score 仍然会选择折衷。其 KL 推导为 loss 提供了动机，但 signed advantage、
direction attenuation 与 clipping 后不再是对 reward-tilted distribution 的严格无偏重要性采样。

因此现在可以审计和移植核心单 reward 算法，但**不能声称官方 mixed Self-OPD 已复现**；如果
我们补齐缺失部分，必须标注为 own reimplementation，并冻结与论文公式的对应关系。

---

## 5. 与我们当前路线的逐项关系

| 维度 | Self-OPD | 当前 CR / Guarded 路线 |
|---|---|---|
| 起点 | SD3.5 base；也讨论 specialist warm-start | 发布的 DiffusionOPD multi-capability Student |
| 问题 | 从头/联合提高多个 reward | 顺序加入新 reward 时保持旧能力 |
| target | student 周围的 stochastic branches | DiffusionOPSD reward target / optimizer proposal |
| 旧能力信号 | 每一步重新运行全部 reward 并加权/否决 | frozen initial Student 与旧 teacher 的 relative budget |
| 多目标机制 | reward-level scalar ranking | per-capability constraint，fixed hinge；计划 projection/certificate |
| reward 要求 | 可 black-box，但每个 branch 都要评 | direct reward 可微；retention 用 field/KL proxy |
| 保证 | 无硬 retention guarantee | 目标是 explicit budget + exact same-state certificate |
| 主要风险 | branch rollout 昂贵、weight/veto 仍需调、held-out 退化 | fixed hinge 牺牲 reward/HPS；projection 尚未验证 |

### 对 novelty 的影响

以下 framing 应立即停止：

- “我们首次用 self-generated targets 做 flow-model self-distillation”；
- “我们首次用正负方向 target”；
- “我们首次避免多 reward gradient conflict”；
- “无需 specialist teachers 的多目标 OPD”。

仍可能成立的 framing：

> Proposal-agnostic capability-safe continual alignment：无论 proposal 来自 DiffusionOPSD
> reward-gradient target 还是 Self-OPD branch search，都用 explicit teacher-relative capability
> budgets 约束真实 optimizer update，并通过 same-state exact certificate 防止遗忘。

这比把方法限定为“更好的 DiffusionOPSD penalty”更稳，也与 Self-OPD 的目标互补。

---

## 6. 这篇论文如何改变下一步优先级

### 第一优先：补一个不需要训练的 same-image evaluation

Self-OPD 的关键论点是 task images 上的 preference 与 held-out DrawBench/Pick-a-Pic 是两种不同
问题。我们现在只知道 `crg` 的 DrawBench HPS 明显下降，却没有测：

- 在 OCR eval images 上的 PickScore/HPS；
- 在 GenEval eval images 上的 PickScore/HPS。

现有 `outputs/eval/*_{geneval,ocr}` 只保留 JSON、没有保留 images，因此不能事后补 scorer。
应以固定 prompt/seed 为每个模型重新生成一次，并在**同一批生成图像**上同时计算任务指标、
PickScore 与 HPS，避免每个指标各采一批图。这个评估不涉及新训练，能区分：

- `crg` 只在 held-out prompt distribution 退化；还是
- 连它正在保护的 task images 也变差。

这个诊断比继续研究 `G_fit` 更重要。

### 第二优先：把 Self-OPD 加为最近邻 baseline，但分两道 gate

**Gate A：public-code fidelity smoke**

- vendor/记录上述固定 commit；
- `num_workers=0`；
- 用现有 OCR reward 做 1–2 step smoke；
- 检查 K branches、ODE reference、signed loss、EMA、checkpoint 与显存/吞吐；
- 用合成 scorer 单测 pull/push 符号、direction attenuation 和 variance normalization。

**Gate B：20-step Student continuation**

- 相同起点：发布 DiffusionOPD Student；
- 先做 Self-OPD-Pick 单 reward，回答它作为 proposal 是否比 DiffusionOPSD-direct 更有效；
- 再做 own Self-OPD-joint：Pick 作为新目标，OCR/GenEval/HPS 作为 quality/capability rewards；
- 与 `crd`、最佳 fixed hinge 采用 matched images、matched optimizer updates，同时报告 wall-clock
  与 GPU-hours。

在官方 mixed configs 未发布前，不直接提交 paper-scale 复现。

### 第三优先：只有在两个独立条件满足后才组合方法

如果：

1. 0023 的 same-state optimizer-aware projection gate 通过；并且
2. Self-OPD proposal 在 20-step continuation 中比 DiffusionOPSD-direct 更强；

则把 Self-OPD 的 actual AdamW proposal 送入我们的 per-capability projection/certificate，形成
**Guarded Self-OPD**。关键比较为：

```text
Student anchor
  ├─ DiffusionOPSD-direct
  ├─ Self-OPD-Pick
  ├─ Self-OPD-joint weighted/veto
  ├─ Self-OPD + fixed retention hinge
  └─ Self-OPD + projected/certified retention
```

只有最后一臂在相同 OCR/GenEval retention 下提高 Pick/HPS Pareto，才值得 full-scale。

如果 Self-OPD-joint 已经完全支配我们的 guard，就应停止当前方法论文，转为复现/分析工作或寻找
held-out retention 的新约束；不能为了保留旧 idea 而忽略新 baseline。

---

## 7. 当前决定

- **是**：这篇工作和我们非常接近，且比 MOPD-continue 更应作为最新近邻方法。
- **不是**：它没有完全做掉“从强 Student 顺序适配并显式守住旧能力”的问题。
- **立即调整**：paper framing 从 multi-objective/self-distillation 转为 continual capability
  retention / certified safe update。
- **立即新增评估**：same-task-image PickScore/HPS。
- **暂不提交长训练**：公开 mixed code/checkpoint 不完整；先做代码 fidelity smoke 和 20-step gate。
- **保留 0023 的 projection gate**，但以后应把它设计成 proposal-agnostic，并允许接收
  Self-OPD proposal，而不是只绑定 DiffusionOPSD gradient。

## 8. Primary sources

- Paper: <https://arxiv.org/abs/2608.26872>
- Code: <https://github.com/Shiy-Zhang/Self-OPD>
- Advertised model repository: <https://huggingface.co/ShiyiZhang/Self-OPD>
- Closest existing local method, DiffusionOPSD: <https://arxiv.org/abs/2608.24646>
