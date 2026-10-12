# 0023 Codex 复核 Claude 新实验：当前结论与下一步决策（2026-08-30）

## 一句话结论

正式的真实数据实验已经给出了一个**可信但边界很窄的正向结果**：从发布的
DiffusionOPD Student 出发继续优化 PickScore 时，`w=10` 的 one-sided same-state
retention hinge 能大幅减轻 direct continuation 对 OCR/GenEval 的破坏，同时仍让
PickScore 高于起点；但它会损失 HPS/CLIP/Aesthetic，而且还没有自己的
MOPD-continue、朴素 Student-KL 等同预算外部基线。因此它现在是一个好的实验现象，
还不是足够完整的 paper result。

Claude 据现有一阶探针直接关闭 QP 路线的结论也过早。现有探针没有在同一个缓存状态/
同一个 batch 上比较预测和真实变化，并且拿第一 microbatch 的约束梯度解释 12 个
microbatch 累积后的 AdamW 更新。它证明了“当前 raw-gradient probe 不可用”，不能证明
“optimizer-aware projection + exact certificate 不可用”。下一步应只做一次便宜、严格的
same-state finite-step gate；通过才进入短训练，失败则永久关闭 QP。

本次没有提交新训练。`crm(w=3)` 训练已完成，step 121/151/181 的三个评测作业仍在队列中；
先让它们完成，不应在只有前三个点时选择权重。

---

## 1. 本次实际核查的材料

- 正式实验与预注册：`0015_CR_PROD_RUN.md`、`0018_CR_PROD_PREREG.md`
- 几何/QP 判断：`0016_GEOM_PROBE.md`、`code/0016_gradient_geometry_probe.patch`、
  `code/0017_qp_solver.py`
- 一阶/AdamW 探针：`code/0019_first_order_validity.patch`、
  `code/0020_adamw_rotation_decomp.patch`
- Claude 的最终定位：`0021_OPEN_QUESTION_fit_loss.md`、`0022_POSITION_vs_BASELINE.md`
- 正式训练日志：`outputs/slurm_crdirect_1036470.out`、
  `outputs/slurm_crguard_1036474.out`、`outputs/slurm_crmid_1036569.out`
- 所有现有 `crd_*`、`crg_*`、`crm_*` 与 `Student_*` 本地评测 JSON
- 独立重跑：`compare_arms.py` 与 `overall_score.py`

日志确认正式实验确实使用了三个真实、分流正确的数据集：PickScore 25,432、OCR
19,653、GenEval 50,000；`crd` 与 `crg` 都正常训练到 200 epoch。因此这次结果不再有
0010 的 smoke prompt 混用问题。

---

## 2. 哪些结果成立

### 2.1 `w=10` 相对 direct continuation 的保持作用很强

在预先指定的六个共同 checkpoint `{31,61,91,121,151,181}` 上，独立重算得到：

| 轴 | `crg(w=10)` | `crd(w=0)` | 均值差 |
|---|---:|---:|---:|
| GenEval `accuracy` | 0.9382 | 0.8581 | **+0.0801** |
| OCR | 0.9430 | 0.7916 | **+0.1513** |
| PickScore | 0.9298 | 0.9441 | **−0.0143** |
| CLIPScore | 0.2878 | 0.2810 | +0.0069 |
| Aesthetic | 6.0960 | 6.2670 | **−0.1710** |
| HPSv2 | 0.3180 | 0.3175 | +0.0005 |

如果使用论文表格对应的 `geneval` 字段而不是 `accuracy`，六点均值差反而更大：
`0.9479 − 0.7197 = +0.2282`。所以“守卫显著减轻 direct continuation 的能力遗忘”
这个定性结论不依赖两个 GenEval 字段的选择。

训练代理也方向一致：正式 run 中 OCR violation 约下降 98%，GenEval violation 约下降
93%，同时 direct objective 的进展变慢约 7%。这说明 fixed hinge 不是无效实验；它给出了
真实的 retention–specialization trade-off。

### 2.2 相对 Student 起点，结果不只是“少破坏了一点”

`crg_181` 与本集群用同一评测代码重跑的发布 Student 权重：

| 轴 | Student | `crg_181` | 变化 | 当前读法 |
|---|---:|---:|---:|---|
| PickScore | 0.9222 | **0.9369** | **+0.0146** | 明确提高目标 reward |
| OCR | 0.9273 | **0.9360** | +0.0087 | 保持/略升 |
| GenEval `accuracy` | 0.9376 | 0.9336 | −0.0040 | 保持 |
| GenEval paper metric | 0.9615 | 0.9375 | −0.0240 | 在现有 2σ 重测带内 |
| CLIPScore | 0.2958 | 0.2874 | −0.0084 | 退化 |
| Aesthetic | 6.1594 | 6.1246 | −0.0348 | 退化 |
| HPSv2 | 0.3427 | 0.3132 | **−0.0295** | 明显退化 |

因此 `0022` 里“只把自身损害从 −10.9% 压到 −2.0%”不是最好的科学表述。
更准确的表述是：**模型取得了新的 PickScore specialization，并保持了预先指定的
OCR/GenEval 能力，但没有保持所有一般图像质量指标。** 这个结果有价值，但 claim 必须限定在
被显式保护的能力上，不能写成 broad capability preservation。

### 2.3 `w=3` 尚未定稿

- 训练 job `1036569`：`COMPLETED`, exit `0:0`, elapsed `03:50:04`。
- step 31/61/91 已评完；step 121/151/181 对应 job `1036772/1036775/1036778`
  仍为 `PENDING (Priority)`。
- 前三点 `crm` 对 `crg` 六个轴均未形成可靠差异。不能据此称 `w=3` 更好，也不能选择
  `w=10`；等六点齐全后只把它们作为 trajectory/Pareto 证据。

---

## 3. 当前分析中必须修正的四个问题

### 3.1 六个 checkpoint 不是六个独立实验重复

`compare_arms.py` 的配对均值适合画同进度的 trade-off 轨迹，但六点来自同一条训练轨迹，
高度相关。lag-1 修正也不能把它们变成独立 seed。现在的巨大、单调效应足以支持定性判断，
但论文中的置信区间和显著性必须以**独立训练 seed**为样本单位。

规则应改为：

- checkpoints 用于曲线、Pareto front 和确定候选 checkpoint；
- 选定方法、权重和 checkpoint 后，至少跑 3 个独立 seed；
- 不再把同一 trajectory 的 checkpoint t-test 写成 paper-level significance。

### 3.2 两个分析脚本使用了不同的 GenEval 定义

- `compare_arms.py` 明确读取 `accuracy`，但输出轴名写成 `geneval`；
- `overall_score.py::_axis_of` 读取 JSON 中真正名为 `geneval` 的字段；
- `0022` 的逐轴表又使用 `accuracy`，但其 Ratio 排名来自 `overall_score.py`。

这会让同一节里的表格和聚合分数不是同一个指标。发布前必须冻结口径：论文主表使用
`geneval`（它与原论文约 0.96 的口径一致），`accuracy` 只作为额外诊断并明确命名。
修正后重新生成所有表，不要手工混用。

### 3.3 现有 probe 不能证明“一阶线性化失效”

当前 `fo_pred_dC=<a,u>` 与“下一 epoch 的 violation 变化”比较时，同时改变了：

- prompt、timestep、噪声和 on-policy latent；
- 当前 student state 与 frozen-reference threshold；
- 梯度估计 batch（`a` 主要来自第一 microbatch，而 `u` 是 12 个 microbatch 的累积更新）；
- SGD/AdamW 的参数更新映射。

所以符号约 50%、幅度只解释 8–10% 也可能主要来自 resampling/state-distribution drift，
不是 Taylor approximation 本身失效。`cos(actual AdamW update, -accumulated gradient)=0.339`
确实说明 raw-gradient QP 不合适，但它反而给出了正确的改法：在**实际 optimizer proposal 的
参数增量空间**投影，再做 exact same-state certificate。

### 3.4 已验证的 QP solver 还不是 0006 所需的问题

`code/0017_qp_solver.py` 当前解的是类似 `a_m^T d >= 0` 的“不再恶化”问题。若当前缓存状态
已经有 `C_m > 0` 的违反，真正的一阶可行域需要非零 repair RHS：

```text
min_Δ  1/2 ||Δ - Δ_adam||²
s.t.   C_m(θ) + a_mᵀ Δ <= 0,  for each protected capability m
```

而且这里的 `Δ_adam` 应是复制 optimizer state 后得到的真实 AdamW proposal，不是 `-g`。
所以 solver 的合成测试通过是好事，但不能把现有组件直接接入训练并称为完整 GuardedOPSD。

---

## 4. 下一步：只做一个有严格 go/no-go 的低成本实验

### Phase 0：先收齐 `crm`，不新增训练

等 3 个 pending eval 完成后，用全部六点画 `w=0/3/10` 的逐轴 trajectory 和
PickScore–OCR、PickScore–GenEval、PickScore–HPS Pareto 图。这里不做 checkpoint-as-replicate
显著性宣称，也不按综合 Ratio 单值选模型。

### Phase 1：same-state optimizer-aware finite-step probe

在真实 Pick/OCR/GenEval batch 上做 16–32 个固定 probe；先不跑 200 epoch：

1. 缓存完整 accumulation window 的 prompt、timestep、noise、latent 和 frozen-reference 输出。
2. 在旧参数上计算 `C_before` 与每个约束梯度 `a_m`，约束梯度也必须覆盖同一 accumulation
   window，不能只取第一个 microbatch。
3. 复制参数和 optimizer state，产生只含 reward objective 的真实 AdamW proposal
   `Δ_adam = θ_candidate − θ`。
4. 在**同一批缓存状态**上计算 direct candidate 的真实 `C_after_direct − C_before`，与
   `a_m^T Δ_adam` 比较。
5. 解带 repair RHS 的 update-space projection，得到 `Δ_proj`；在同一缓存状态上精确重算
   `C_after_proj`。
6. 若 full step 不可行，按 `{1, 1/2, 1/4, 1/8}` backtrack；同时记录保留的 reward update
   比例、direct loss 的真实变化、每个约束是否满足。

预先冻结 go/no-go：

- 如果 direct proposal 基本从不破坏缓存约束，则 projection 没有要解决的问题，关闭路线；
- 如果 projected candidate 在至少 90% 的 active probe 中能经至多 3 次 backtrack 通过 exact
  certificate，且 median reward-progress retention ≥ 0.5，则进入短训练；
- 否则关闭 QP，保留 fixed hinge 作为结果，不再追加几何探针。

这个 probe 回答的是算法是否真的能产生“少牺牲 reward、仍满足约束”的**有限步更新**，比继续
研究高维 cosine 或 `G_fit` 更直接。

### Phase 2：probe 通过后才跑短三臂训练

只跑 20–30 optimizer steps，三臂使用完全相同的 batch schedule：

1. OPSD-direct / `w=0`；
2. 当前最佳 fixed hinge (`w=3` 或 `w=10`，等六点评完后冻结)；
3. optimizer-aware projection + exact certificate。

成功标准不是“QP 可运行”，而是第三臂在 OCR/GenEval 同等 retention 下，PickScore/direct-loss
进展严格优于 fixed hinge。若没有 Pareto 改善，就不进行 full run。

---

## 5. 进入 paper-scale 前不可缺的基线

所有主表数字都应在本集群、同一 checkpoint schedule、同一 seed 和同一评测代码下自己跑；
官方论文数字只作 sanity reference。比较关系应固定为：

1. 发布的 DiffusionOPD Student：初始化/anchor，本地重评；
2. **Student → DiffusionOPSD-direct**：当前 `crd`，证明不加保护时会怎样；
3. **Student → MOPD/DiffusionOPD-continue**：同计算预算的论文方法基线，目前缺失；
4. Student → naive frozen-Student KL：最必要、最低成本的 regularization baseline；
5. 当前 fixed one-sided same-state hinge；
6. 只有 Phase 1/2 通过时才加入 optimizer-aware projected variant；
7. 若篇幅/算力允许，再加 adaptive primal-dual baseline。

其中 2 是机制 control，3 和 4 才是审稿人会要求的 method baseline。不能只拿官方表格中的
MOPD 数字与自己的 continuation 数字横比。

---

## 6. 最可能形成论文的路线

### 如果 projection gate 通过

主张应是：**在已蒸馏的多能力 diffusion student 上进行新 reward specialization 时，用
optimizer-aware feasible update 与 exact certificate 保持指定旧能力，并比 fixed KL/hinge
获得更好的 retention–reward Pareto trade-off。**

这比“再调一个 penalty weight”更有明确方法贡献，也直接对应现有实验已经暴露的代价。

### 如果 projection gate 失败

不要再花时间在 QP/GN。把工作收束为：

- one-sided same-state retention 的扎实 empirical study；
- 与 direct、MOPD-continue、naive KL、primal-dual 的公平比较；
- 解释为什么只保护 OCR/GenEval 会让 HPS/CLIP/AES 下降；
- 如基线后仍有优势，再考虑加入一个 general frozen-Student KL/budget 保护总体质量。

但在外部基线尚未跑完之前，不应立即增加第三个 general constraint；否则方法和实验会继续
变 messy，也无法知道收益来自哪个组件。

---

## 7. 本次决定与作业状态

- 接受：正式 `w=10` fixed hinge 对 direct 的 retention 正向结果。
- 修正：不能称为 broad preservation；HPS/CLIP/Aesthetic 的损失必须显式报告。
- 暂缓：`w=3` 权重选择，等待 step 121/151/181。
- 推翻：仅凭现有 0019/0020 probe 永久关闭 QP。
- 新闸：只允许一次 same-state、optimizer-aware、finite-step certificate probe。
- 基线：下一轮 full-scale 优先补自己的 MOPD-continue 和 naive Student-KL。
- 统计：后续 paper claim 用独立 seed，不用 checkpoint 充当 n。
- 本次新提交作业：**无**。现有 pending eval 已覆盖 `crm` 的缺失点，避免重复提交。

## 8. 本次独立复核命令

```bash
python opd_related/compare_arms.py crg crd
python opd_related/compare_arms.py crm crg
python opd_related/overall_score.py
sacct -j 1036569,1036772,1036775,1036778 \
  --format=JobID,JobName%18,State,ExitCode,Elapsed,Start,End -X
```

后续 Claude/Codex 的实现与作业必须继续按新 ID 追加，不覆盖本记录；实现 probe 前先把 probe
口径、样本数和 go/no-go 写入下一份预注册文件。
