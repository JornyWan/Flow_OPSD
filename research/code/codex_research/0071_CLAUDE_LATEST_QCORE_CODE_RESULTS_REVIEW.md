# 0071 — Claude 最新 Q-Core 代码、实验与 `0070` 论证复核

日期：2026-09-04

本轮只做代码和结果审计，没有提交新训练/评测任务。复核对象：

- `codex_research/0070_Q_VS_BASELINE_FULL.md`；
- `EXPERIMENT_LOG.md` §4.145–4.160；
- `CA-OPD/code/DiffusionOPSD-rpi/scripts/train_opsd_ri_sd3.py`；
- Q-Core 训练产物、matched 2×2、六格 sampler grid；
- 0070 写完后完成的 CFG 全量评测。

---

## 1. 结论先行

### 可以保留的结论

1. **Q-Core 代码的核心数学路径实现正确。** 它不是只在第一步使用 suffix-Q：
   第一梯度在 `y0` 共享，第二步在正负两支各自当前位置重新计算
   `∇y R(D(F_old(s,y)))`。因此它确实保留了 DiffusionOPSD 的两步构造并打破
   qopsd/qbt 的严格镜像退化。
2. **matched DPM2-10 上的正结果是真实且很大。** old/behavior adapter、unseen
   Pick-a-Pic、DPM2-10 下，Q-Core 相对 baseline 的 Open3 和为
   `+0.05613 ± 0.00140 SE`，paired `t=40.21`，win rate `91.1%`。
3. **官方部署口径仍是负结果。** default/EMA adapter、DrawBench、flow-40 下
   `Δ=-0.0141`，`t=-3.53`。因此现在不能写成“Q-Core 改善 DiffusionOPSD”。
4. sampler grid 揭示了一个值得继续追的现象：**Q-Core 把模型对采样深度/solver 的
   绝对响应压平了**。这很像把一部分迭代求解收益摊销进模型，但尚不能叫
   “sampler-independent”。

### 不能照收的结论

1. `0070` 的“Q 近似 sampler 无关”表述过强。Q-Core 的绝对分数范围虽只有
   `0.0204`，但它对 baseline 的相对优势从 `+0.0561` 变到 `-0.0454`，跨度
   `0.1016` 且跨零。正确名称应是 **sampler-response flattening / saturation**，
   不是已经证明的 sampler robustness。
2. “Q 继承 DPM2 天花板”目前只是合理假说。所谓 `0.0237` 的“精确分解”是选择
   DPM2-40 作中间点后的代数恒等式，不是因果识别；40 步也没有被证明是渐近值，
   Q-Core 在已测六点中的最大值也不能被称为天花板。
3. “flow suffix 预计拿回 57%（+0.0136）”没有数据依据。换 suffix 会同时改变
   action-value Jacobian、训练目标和 off-distribution state 行为，baseline 的
   flow-vs-DPM 差不能直接当作可回收增益。
4. “2.8×”可以保留为**单 seed、old adapter、Open3 reward、log-step 插值得到的初步
   估计**，不能当作已经测得的数字。当前直接观测只支持：相同质量的 baseline
   DPM2 步数落在 20–40 之间，即 `2×–4×` 区间。

---

## 2. Q-Core 代码审计

### 2.1 核心路径确实是“两步都用 Q”

代码路径为：

1. `_dpm2_continue_from_grad` 保留 `forced_x0` 到 suffix endpoint 的梯度，并让梯度
   穿过后续 frozen-old-policy velocity；
2. 在 `y0` 处先计算 `_qc_g0`；
3. `_opa_tr_step(..., first_grad=_qc_g0, q_ctx=_qc_ctx)` 的第一步使用共享的 Q 梯度；
4. 第二步因 `_i>0` 进入 `q_ctx` 分支，在正、负两支当前 `x` 上分别重算 suffix-Q。

所以实际 proposal 是：

```text
y+ = y0 + h·unit(dQ(y0)) + h·unit(dQ(y0+h·unit(dQ(y0))))
y- = y0 - h·unit(dQ(y0)) - h·unit(dQ(y0-h·unit(dQ(y0))))
```

而不是早期 qopsd 的 `y±=y0±a·dQ(y0)`。后一种严格镜像使
`pos_loss == neg_loss`，Q-Core 已经修掉这个结构问题。

已有数值验证也足够支撑微分实现：可微/不可微 suffix endpoint 相同、action
gradient 非零、有限差分相对误差 `1.69e-5`。训练中 `mean|pos_loss-neg_loss|`
从退化版约 `1e-7` 恢复到前五 epoch 均值 `1.32e-2`，epoch 99 为 `2.99e-2`。

### 2.2 frozen old policy 与 detached target 没有串进 optimizer

`vold_g` 虽然让 old-adapter 参与 autograd graph，但调用只取
`autograd.grad(Q, y)`；Q proposal 最后 detach，optimizer 只持有 default-adapter
参数。epoch 末才按原算法把 default 更新进 old。这个实现语义成立。

### 2.3 `old` 与 `default` 必须分清

checkpoint 根目录的 default adapter 在保存前被 EMA 权重覆盖，是官方评测/部署对象；
`lora/old/` 是训练时的滞后 behavior adapter。当前六格 sampler grid 全部评的是
**old adapter**，因此它适合诊断训练工作点，却不足以支持部署模型的推理加速主张。

### 2.4 正确性不受影响、但必须补的工程问题

1. **provenance 漏项。** `RPI_QCORE` 没进 `preflight_check.py` 的
   `DANGEROUS_ENV`，也没进 `ab_open3.slurm` 的 `_AE` 和
   `arm_provenance.json`。stdout 确实打印了 `RPI_QCORE=1`，所以这次 run 可以追认；
   但 manifest 自身无法区分 qcore 和 baseline。
2. `run_config.json` 的 `git_commit="unknown"`，也没保存训练脚本 hash。当前脚本
   SHA-256 是 `b0893618...`，但不能从 checkpoint 自证训练时一定是该版本。
3. Q-Core 第一次 job `1040211` 在 80GB GPU 的 epoch 0 构造 Q 梯度时 OOM；正式 job
   `1040308` 在 H200 上从头开始、无 resume 跑完。两条 attempt 都留在
   `run_attempts.jsonl`，最终 checkpoint 没有混入失败 run 的 optimizer state。
4. 同一 H200、np3 下，baseline 用时约 `6:01`，Q-Core 约 `8:23`，训练墙钟增加
   约 **39%**；当前实现还无谓地计算了一次随后不用的 canonical `opa_g0`。这不改结果，
   但论文若强调效率必须报告训练成本并清掉该冗余。

---

## 3. 结果复算

目标合计均按逐图 `PickScore/26 + CLIPScore + HPSv2.1` 重算，同 index、prompt 配对，
`n=1000`。

### 3.1 matched 2×2

| protocol | baseline | Q-Core | Δ | paired t | win |
|---|---:|---:|---:|---:|---:|
| unseen PP + DPM2-10 | 1.58205 | 1.63818 | **+0.05613** | **+40.21** | 91.1% |
| DrawBench + DPM2-10 | 1.5619 | 1.6247 | **+0.06275** | +38.20 | 90.5% |
| unseen PP + flow-40 | 1.66183 | 1.63131 | **-0.03052** | -25.15 | 18.2% |
| DrawBench + flow-40 | 1.6382 | 1.6179 | **-0.02029** | -13.43 | 29.6% |

这证明的是一个**强 protocol interaction**，而不是一般化提升。

### 3.2 六格 sampler grid

| arm | DPM2-10 | DPM2-20 | DPM2-40 | flow-10 | flow-20 | flow-40 | range |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1.58205 | 1.62840 | 1.64824 | 1.66324 | 1.66369 | 1.66183 | 0.08163 |
| qopsd | 1.64125 | 1.63891 | 1.63091 | 1.59952 | 1.62267 | 1.62863 | 0.04173 |
| Q-Core | 1.63818 | 1.63543 | 1.62959 | 1.61781 | 1.62932 | 1.63131 | 0.02037 |

Q-Core 相对 baseline 的六点 Δ：

```text
+0.05613, +0.00703, -0.01865, -0.04543, -0.03437, -0.03052
```

因此两个事实要同时说：

- Q-Core 自身输出对 sampler/步数不敏感；
- Q-Core **相对 baseline 的效果高度依赖 sampler/步数**。

低 absolute range 可能表示摊销了迭代收益，也可能表示模型无法利用更好的 solver；仅凭
这张表无法区分。

### 3.3 `0070` 四因子分解用了错误的 arm

`0070` §5 给出的 prompt `-0.0004`、sampler `-0.0924`、adapter `+0.0009`
实际对应早期 **qopsd**，不是表格加粗所讨论的 Q-Core。

Q-Core 从 matched cell 出发的实际变化为：

- prompt：`+0.06275 - +0.05613 = +0.00662`；
- sampler：`-0.03052 - +0.05613 = -0.08665`；
- prompt×sampler interaction：约 `+0.00360`；
- old→default adapter（DrawBench flow-40）：约 `+0.0062`，不是 `+0.0009`。

sampler 仍然显然是最大因素，但“99%”不是 Q-Core 数据支持的数字，而且带交互项时不应
把这些有符号差异硬解释为可加百分比。

### 3.4 “70%”混用了 solver 家族

`0.0797` 是 `baseline flow-40 - baseline DPM2-10`，不是同一 DPM2 家族的
10→40 差。Q-Core 的 `+0.05613` 占它约 70%。若严格在 DPM2 家族内，baseline
10→40 的增益是 `0.06619`，Q-Core 补回的是约 **84.8%**。两种说法都可作描述，
但不能把前者放在纯 DPM2 的 2.8× 论证中而不注明 solver 也变了。

---

## 4. 对 `0070` 三个核心问题的回答

### 4.1 天花板归因是否成立？

**尚未成立，只能列为 H1。** 原因：

1. `A-B=(A-C)+(C-B)` 的数值吻合必然成立，不能证明 C 是机制中介；
2. DPM2 从 20→40 仍增加 `+0.01984`，没有证据说 40 已渐近；
3. Q-Core 的“天花板”只是单 checkpoint、六个离散设置里的最大值；
4. baseline ck100 已知未收敛，Q-Core ck200 未测；
5. 只有一个训练 seed。

可以写成：**“Q-Core 的 sampler-response flattening 与 solver-conditioned
amortization 假说一致。”** 不能写成已经证明“继承 solver ceiling”。

### 4.2 2.8× 能否成立？

当前最严谨的写法是：

> 在 single-seed、old behavior adapter、unseen Pick-a-Pic、Open3 reward 的
> CFG-free DPM2 评测中，Q-Core@10 显著优于 baseline@20、显著差于
> baseline@40；按 20/40 两点在 log2(step) 上插值，等质量交点约 28 steps，
> 对应 **estimated 2.8×** step reduction。

另一个代码层面的更正：这里的 DPM-Solver++ 2M 是 **multistep** solver，采样循环每步
只调用一次 `v_pred_fn`。在 `gs=1` 下，10/20/40 steps 就是 10/20/40 次 transformer
evaluation；所以 `0070` 的“二阶故 steps≠NFE”担心在本实现里并不成立。仍应实测墙钟，
但 NFE 不需要乘 2。

要把 estimated 2.8× 升级成 paper claim，至少还缺：

- baseline 的 24/28/32-step 直接观测，而不是两点插值；
- default/EMA adapter 的完整曲线；
- ≥3 个训练 seeds；
- 实际生成墙钟/吞吐；
- 不只用训练时同一组 reward models，还要有 GenEval/OCR、独立偏好模型/人评与多样性；
- 与真正的 few-step/distillation acceleration baseline 比较。

### 4.3 是否直接做 flow-suffix Q？

**可以作为候选，但不能依据“57% 可回收”直接开正式训练。**

若仍从 DPM2 rollout state 出发，只把 suffix 换成 flow，得到的是
`DPM-prefix + flow-suffix` 的 hybrid value；它不等于完整 flow rollout 上的 action value。
state distribution 与 suffix operator 两个因素仍混在一起。

应先做一个无需训练的 `state × oracle suffix × evaluation suffix` probe：

| 维度 | 水平 |
|---|---|
| query state 来源 | DPM2 rollout / flow rollout |
| 构造方向 | local clean / DPM2-Q / flow-Q |
| target 评估 suffix | DPM2 / flow |

所有方向同 state、同半径，报告逐轴 gain、negative rate、方向间 cosine。只有当
`flow-Q` 在 flow suffix 上稳定优于 DPM2-Q，且不是完全牺牲 DPM2，才值得训练 qflow。
若 state×suffix 交互很强，则正确方法应是 solver-conditioned 或 solver-ensemble Q，
而不是只替换一个 helper。

---

## 5. 0070 之后完成的 CFG 全量结果

job `1041233` 已完成 n=1000、逐图配对。对象是 baseline DiffusionOPSD ck100，
不是 Q-Core。

### trained gs=4.5 相对 trained gs=1.0

| 指标 | Δ | paired t |
|---|---:|---:|
| Open3 sum | **-0.05472** | **-32.57** |
| Pick | -0.7824 | -35.96 |
| CLIP | -0.00449 | -5.19 |
| HPS | -0.02013 | -29.47 |
| Aesthetic | -0.1257 | -10.71 |
| ImageReward | **+0.0390** | **+3.76** |

所以小样本的“额外 CFG 会让训练目标过冲”得到全量确认；但不能说所有质量指标都变差，
因为 ImageReward 显著上升。更准确的解释是：**额外 CFG 与被优化的 Open3 偏好方向不兼容，
不是普适画质崩坏。**

训练后 gs=1 相对未训练 base gs=4.5 仍是 `Open3 +0.21299, t=86.69`；训练后即使
开 gs=4.5，相对 base gs=4.5 也有 `+0.15827, t=64.33`。这支持 DiffusionOPSD 已把
大量 CFG 带来的 target-reward 增益内化，但 ImageReward 的反向变化说明不能把机制
简化成“所有意义上的 CFG 饱和”。

---

## 6. 我建议的顺序

### P0：先封住口径，不开新 Q 训练

1. 给 `RPI_QCORE` 补 preflight allow-list、dangerous-env、arm provenance；保存代码 hash。
2. 用 default/EMA adapter 补 Q-Core 与 baseline 的 DPM2 10/20/24/28/32/40 和
   flow 10/20/40；同时记录真实墙钟。
3. 把 `0070` 的术语改为 `sampler-response flattening`，撤掉“99%”“ceiling 已归因”
   和“flow 可回收 57%”的确定语气。

### P1：零训练机制 probe

做上面的 `state × oracle suffix × evaluation suffix` 矩阵。它直接决定下一版是：

- flow-Q；
- DPM/flow normalized gradient ensemble；
- solver-conditioned Q；
- 或者 Q 路线停止。

### P2：之后才决定训练

若 probe 支持 flow-Q，做同 seed 的 `baseline / qcore-DPM / qcore-flow` 三臂，其他配置完全
一致，先以 official default-adapter flow-40 为主判据。同时 Q-Core 要与 baseline
做同为 ck200 的收敛预算对比，不能继续只比未收敛的 ck100。

### 与拟合端路线的关系

当前 official default flow-40 上：

- Q-Core：`-0.0141`；
- 简单的 inner2：`+0.0145`（paired `t=12.27`）。

所以如果目标是最稳妥地拿主表正结果，现阶段拟合强度路线比 Q 更领先；`lr2x` 正在跑，
`inner4` 与 `inner2_200` 尚未完成。若目标是方法新颖性，Q 路线的最佳定位不是“已经提高
官方 DiffusionOPSD”，而是 **solver-conditioned value-gradient amortization / few-step
acceleration**。两条路线应分开叙述，暂时不要把 inner2 与 Q 合并，否则无法归因。

---

## 7. 最终判断

Claude 这轮最大的贡献是修复了真正的 Q-Core，并通过 paired grid 发现了一个此前被误读的
现象；这部分值得保留。最大的问题是从“曲线变平”跳到了“sampler-independent”，再从一个
代数分解跳到了“solver ceiling 已归因”和“flow suffix 可恢复 57%”。

我对当前证据的最终表述是：

> **Q-Core 在其训练 solver/步数下具有非常强的正效果，并显著压平了模型随采样步数增加的
> reward 变化；这提示它可能在摊销迭代推理，但该效果尚未在部署 adapter、跨 seed、独立
> 质量指标或精确 step/NFE 曲线上成立，而且 official flow-40 结果仍显著落后。**

这足以支持一个很清晰的下一轮机制实验，但还不足以支持 `0070` 当前版本的 paper-level
因果结论。
