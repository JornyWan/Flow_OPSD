# Terminal-verified local expert distillation v1

本轮实现主线一：**教师整体更强，不意味着每个学生状态上的教师修正都值得模仿。先验证一次局部介入的最终结果，再决定是否蒸馏。** 方向二（无教师候选自改进）和方向三（保留 optimizer state 的 shadow update 选目标）尚未实现。

## 一个训练周期

1. 将当前 raw 学生 LoRA 复制为冻结 `behavior`。采集全部查询和完成 fitting 期间不刷新它。
2. 用原 prompt 和 behavior 生成到查询状态。v1 只在确定性 flow/Euler 上做局部介入；不存在被遗漏的多步 solver history。
3. 查询每个专家一次，候选为 `v_b + alpha * (v_teacher - v_b)`，另有 `noop = v_b`。默认 OCR、GenEval 两个专家、alpha 0.5/1.0，共五条分支。
4. 每个候选只替换这一步 velocity，余下后缀全部由同一个 behavior 继续。初始状态、prompt、CFG 和网格相同；确定性后缀没有分支噪声差异。
5. 对**同一张最终图像**逐项验证 OCR、计数、颜色、关系。分别保留布尔满足标记 `b_j` 和 `[0,1]` 的进度 `r_j`。
6. `U = eta * all(b_j) + (1-eta) * mean(r_j)`。候选必须保留 noop 终点已满足的每个约束，且 U 严格超过 noop + margin；可选 aesthetic floor 和允许下降幅度。最高 U 胜出，同分选离 noop 最近的动作，最后按固定候选顺序打破平局。无合格候选就拒绝该查询。
7. 在原查询状态上拟合 detached velocity target：

   `loss = 0.5 * mean(((sigma_next - sigma) * (v_student - target)) ** 2)`

   这对应原版确定性 flow OPD 的单教师 transition-mean loss，**不是**去掉 dt 权重的 velocity MSE。数值测试执行了从新拉取上游 AST 提取的 loss 代码。

8. 拒绝查询 loss 为零，分母仍是全部查询数，防止接受率低时隐式放大学习率。一个全局 accumulation group 全被拒绝时不执行 AdamW/EMA，避免零监督下 momentum/weight decay 仍更新模型。
9. 保留 AdamW 和 EMA 状态，下一 outer epoch 才刷新 behavior。EMA 更新约定显式记录，所有局部实验臂使用相同实现。

## 对照的准确含义

| 入口/方法 | 实际含义 |
|---|---|
| `native-prepare diffusionopd` | 固定版本原始 MOPD/SOPD 训练算法，原数据、路由、loss、累积、EMA；单独的生命周期补丁 |
| `native-prepare diffusionopsd` | 项目页链接的公开 OPSD 实现，原在线查询、reward gradient target、fitting 流程 |
| `single_query` / `fixed` | 同一局部引擎中固定教师、固定 alpha 的单状态监督；两者是别名，**不是**两个不同算法，也不是完整 DanceOPD 复现 |
| `round_robin` | 按全局查询序号轮换教师，固定 alpha；不读取终端分数 |
| `random` | 以固定概率接受同一个固定教师动作；不读取 verifier 决定是否接受 |
| `disagreement` | 固定教师与 behavior 的 velocity 平均平方差超过阈值时接受；不读取 verifier |
| `verified` | 候选间终端验证和逐项约束保护 |

所有局部实验臂都会评估全部候选以记录诊断和匹配查询成本；固定/随机/差异臂的**选择决策不使用这些终端分数**。不同接受率可能带来不同实际 optimizer steps，记录中分别报告，不能只按 step 宣称计算公平。可分别设置 `fixed_teacher` 和 `fixed_alpha` 跑所有固定策略，再比较动态选择是否优于最佳固定策略。不能在测试集选择最佳固定策略。

原版 MOPD 与这里的单状态引擎在训练 prompt、查询数、时间采样、有效 batch 等方面有区别，因此“新方法减去原版 MOPD”的差不能单独归因于筛选。**验证筛选机制应比较同一引擎中匹配配置的控制臂**；原版用于独立的算法复现参照。demo 配置不是论文优化过的最终 recipe。

## Verifier 定义

数据中每个 `geneval` 条目是一个需要保留的语义子句。关系子句可包含作为参照的对象。使用固定版本 OPD GenEval 的 strict correctness；进度负值裁剪到零。计数、颜色和位置沿用上游对应 tag 的 detector 阈值，可能存在不同 tag 的检测差异，不宣称它们是真实场景的完美判断。

v1 明确拒绝 `exclude` 子句：上游 strict reward 并未实现这一部分检查，不能接受一项实际上被忽略的约束。上游位置进度也存在失败时仍累计部分分数的行为；保留它作为已命名的代理进度，约束是否满足始终读取 strict boolean。

OCR 使用 PaddleOCR 全部识别文字的 NFKC、casefold、合并空白后的**整串 exact match**，进度是归一化编辑距离。它与上游原版 OCR 的“去空格后子串即成功”不是同一个指标。原版复现保留上游定义，统一评测使用 `joint_constraints_v1`，禁止跨两者比较数值。

加载依赖/权重失败直接停止。逐图 verifier 异常记为 invalid，不作为负标签；任何 rank 的候选集合出现 invalid 时，训练在拟合前全局停止，避免隐式改变候选池。冻结诊断保留完整错误证据并标记失败；统一评测出现 invalid 结果，不允许继续生成正式 paired comparison。

默认 quality guard 未启用；可以显式配置 `quality_floor`、`quality_drop_tolerance`，启用上游 aesthetic scorer。阈值需要在训练/验证集校准，不能从测试结果反推。即使启用也只保护该代理指标。

## 三层收益必须分开

- `queries-rank-*.jsonl`：同一冻结轨迹上的候选收益、接受结果、是否破坏已有约束；`analyze` 给出教师局部 harm rate 和合格候选 oracle headroom。
- `postfit-rank-*.jsonl`：拟合后的同状态单次介入收益，以及**训练 prompt** 上完整更新策略 rollout 的分数。前者继续用旧 behavior 后缀；二者均不能冒充 held-out 效果。
- `eval`：独立 validation/test prompts，正常完整采样，无教师和候选选择。可用 flow/DPM2 的 10/16/24/32/40 步分别评价；**没有实现或宣称跨 sampler 稳定提升**。

每个阶段累计模型 forward 次数、CFG 计算样本数、teacher 调用、decode 图数、各 verifier 调用和训练 backward 查询数。`launches.jsonl` 的 wall time 包含加载、保存与诊断；GPU-hours = 该次 launch wall time × GPU 数。训练日志内的阶段累计时间不能替代完整 launch 成本。CPU toy 的 GPU-hours 为零。

## 本轮边界

已实现 SD3.5 + LoRA、单机多卡同步训练（torchrun）、完整 outer-epoch checkpoint/resume、局部筛选控制臂和统一评测。局部引擎尚未实现多节点训练、stochastic/multistep 分支介入、不可微自提案方向、shadow-update 目标选择、独立 NFT/TreeGRPO 训练封装。native 上游入口已有多节点启动封装，尚未进行真实跨节点 GPU 验证。所有分支最终收益保证都只针对当前冻结策略的实际验证结果，不保证拟合后策略或未知 sampler 的收益。

第二阶段新增均衡的 `training_manifest`、只筛选固定动作的 `verified_fixed`，以及通过 `protect_satisfied=False` 关闭保护的消融。默认 v1 行为保持；参见 [当前 pipeline](pipeline.md)。
