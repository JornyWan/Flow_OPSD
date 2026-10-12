# 0007 — Claude 对 0002–0004 的审计(按 0005 行动项 #4)

Date: 2026-08-29
Status: audit; no job submitted
Auditor: Claude (受 0005 §"Immediate operational actions" #4 指派)

## 审计范围
0005 指定四项:mask routing / frozen-reference timing / branch normalization /
EMA-raw checkpoint behavior。逐项核对 `code/0002_constrained_retention.patch`
与 `0004` 的记录。

## 结果

| 检查项 | 结论 | 证据 |
|---|---|---|
| mask routing | **通过** | 逐样本 `_tidx = sample["teacher_idx"]`;`_ret_mask` 按 retention slot 累加,`_direct_mask` 取 `_tidx == cr_reward_teacher_idx`;**且有显式互斥检查**(`if cr_reward_teacher_idx in cr_retention_teacher_indices: raise`)与 slot 范围检查 |
| frozen-reference timing | **通过** | `cr_reference_snapshot` 在 **resume 状态恢复之前**取(注释与代码一致),故恒为 released Student;`_swap_params` + `torch.no_grad()` 重放;`D_ref` 与 `D_cur` 用**同一 epoch 的同一 rollout state 与同一 j**;**epoch-0 自检** `max_rel > 1e-2` 即 raise,smoke 实测 12/12 批全为 `0.000000e+00` |
| branch normalization | **通过** | 两分支均为各自掩码子集的均值:retention `(_ret_mask*_ret_per_sample).sum()/_ret_mask.sum().clamp(min=1)`;direct `(_direct_mask*_direct_per_sample).sum()/_direct_mask.sum().clamp(min=1)` |
| EMA/raw checkpoint | **通过(且已由 0004 自查)** | 导出的 LoRA 是 EMA 权重、`EMAModuleWrapper` 每 8 步更新,3 步 smoke 导出值等于初始 Student;raw 参数确已更新(Adam step=3,380/382 张量一阶矩非零) |

## ⚠️ 发现一处会导致标定偏差 10 倍的不对称(不在 0005 列出的四项内)

### 事实
- `_direct_mask` 含 `& (sample["rtc_jq"].long() == j)` => **direct loss 只在 RTC 查询点那一个 j 上非零**;
- retention 的 `per_step_kl_scalar` 在**训练时间步循环的每个 j** 上都参与,每个 j 各做一次 backward;
- 实测 `num_train_timesteps = num_steps * timestep_fraction`,`num_steps = 10`。

### 后果
```
0004 测得(单 batch 探针):  ||g_direct|| / ||g_retention|| ~= 105
epoch 累积后:              direct x 1 步   vs   retention x ~10 步
实际有效比 ~= 105 / 10 ~= 10.5
```
0004 自己已注明该探针是 "one local batch per rank and epoch",
**但 0004 §"What this changes" 与 0005 §Phase B 都以 105x 作为标定依据**。
若据此设固定 `cr_retention_weight` 或初始化 primal-dual 的 lambda,
**会把 retention 项高估约 10 倍** —— 约束过强会压掉 direct 增益,
且**不会崩、不会报错**,只会被读成"约束方法没用"。

### 建议(与 0005 Phase B 已列的修正项一致,但需加一条)
1. 标定必须用 **epoch 累积**的分支梯度范数(0005 已列);
2. **新增**:日志里同时记录两个分支各自的**生效步数/生效样本数**
   (如 `cr_direct_steps`、`cr_retention_steps`),使比值可直接读出而非靠推导;
3. primal-dual 的 lambda 初始化用上述 epoch 级比值,而非单 batch 探针值。

## 对 0005 决策门的补充意见

0005 Phase A 的 go/no-go 用的是 **held-out PickScore**,这一点正确且重要。
补充一个已观测到的事实供 Phase A 判读:
`1035879`(top-k)的**训练** reward 相对 canonical 是 **显著更差**的
(n=24 配对:−0.0087, t=−3.48, 19/24 为负, 符号检验 z=−2.86)。
按 0005 的 no-go 判据("training reward rises but held-out PickScore does not"的对偶情形),
**训练 reward 本身就没有上升**,故 top-k 很可能直接落入 no-go。
但仍应按 0005 要求跑完 held-out 评测再判 —— 训练 reward 与 held-out 不必然同向,
且该评测同时提供 OCR/GenEval 的退化基线,是 Phase B 必需的输入。
(`evtopk` 与基线补点 `evrepro10` 已挂后台,在 `1035879` 结束时自动提交。)
