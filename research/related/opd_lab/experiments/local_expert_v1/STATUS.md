# Local expert v1 — 实验状态

更新：2026-09-10。第一阶段的工程验证、冻结诊断、图片复核和拟合步长检查已完成，本轮提交作业均已结束。以下区分工程验证、冻结局部干预和训练效果；不将提交作业当成完成实验。机器可读摘要和产物哈希见 [results.json](results.json)。

## 已有证据

- 固定官方/项目页上游 commit 的独立快照；未复制旧 Q-Core/CA-OPD 训练代码。
- 显式 prompt × seed × query-index 清单、跨位置共享初始噪声、所有候选 PNG/评分/文字识别、输入和证据哈希。
- `round_robin` 教师轮换控制臂；invalid 候选阻止后续拟合；完整 checkpoint/resume；raw/EMA 分开导出。
- H100 CUDA、实际矩阵运算和 MMCV GPU NMS 通过：`runs/le1_gpu_probe_002/probe.json`。
- 40 项 CPU/真实小型 SD3+PEFT/恢复与协议测试通过；不能替代真实 GPU 实验。报告分组统计修改后，相关 5 项测试另行通过。
- 实际 PaddleOCR + joint verifier 的五个已知答案测试全部通过：正确文字、空白、额外文字、错误词、大小写归一化。见 `runs/le1_ocr_controls_001/result.json`。这不证明复杂生成图上的 OCR 准确率。
- **H200 端到端 smoke 完成**：4 prompts × 1 seed × 3 positions × 5 actions = 60 张候选图，0 invalid；同轨迹 noop 跨三个位置完全一致；0 optimizer updates。Slurm 1047321，182 秒，约 0.0506 GPU-hours。
- **新拉取原版 MOPD 的训练 smoke 完成**：`le1_native_opd_smoke_003` / 1047349，H100 上 2 outer epochs、2 optimizer updates；两个 checkpoint 均有 raw/EMA LoRA、optimizer/scaler/EMA state 和 RNG；总分配 138 秒，约 0.0383 GPU-hours。单卡、各任务每 split 4 条、eval 10 步，明确不等于完整论文复现。
- **新拉取 OPSD 的训练 smoke 完成**：`le1_native_opsd_smoke_007` / 1047361，H100 上 2 updates；上游导出的 EMA `lora`、old adapter、optimizer/scaler/EMA/RNG 均保存。81 秒，0.0225 GPU-hours；4 条数据、group=2/batch=2，属于明确标记的单卡工程变体。
- **两个 native checkpoint 重新加载并完成原版评测**：`le1_native_evaluation_job_001` / 1047369，MOPD OCR 和 OPSD DrawBench/open3 各 4 条、10 步，77 秒；这些 native 分数不能与本地 joint verifier 混比。
- **本地两臂训练、续训和统一评测完成**：round_robin/verified 各 2 次更新，真实 stop-after-1/resume，raw/EMA 分别评价；4 条 validation × 5 种权重 × flow/DPM2 10 步，共 40 张图。评测见 `runs/le1_evaluation_smoke_job_002/checks/result.json`，Slurm 1047365，406 秒。

两次更新的工程小样本均未显示相对 backbone 的正增益。verified EMA 的 utility delta 为 flow10 −0.21875、DPM2-10 −0.29365；round_robin EMA 为 −0.20288、−0.46240。仅 4 条 validation 和两次更新，**既不能证明有效，也不足以判断方法失败或泛化规律**。

逐次拟合审计发现：verified 第一轮接受目标的 transition MSE 总和从 0.00034919 升至 0.00118091（3.38 倍），第二轮为 0.919 倍；round_robin 两轮分别降至 0.476、0.697 倍。见两训练 run 的 `postfit_audit.json`。因此追加三档学习率的一次更新检查，计划见 [fit_scale_plan.json](fit_scale_plan.json)，Slurm 1047375；该实验不使用 validation/test。**检查已完成，三臂初始候选图片、分数、查询和选择完全一致**：

| 学习率 | 更新后 / 更新前的接受目标拟合误差 |
|---|---:|
| 3e-4 | 3.379 |
| 1e-4 | 0.672 |
| 3e-5 | 0.426 |

这支持“当前小目标的一步更新过大”这一局部解释，说明较小步长能在该批次拟合目标；不证明长期最优学习率，也不证明 held-out 完整采样有效。

Smoke 报告：[统计](../../runs/le1_smoke_001/reports/v1/report.md)、[候选图复核页](../../runs/le1_smoke_001/reports/v1/review.html)、[20 图定性检查](../../runs/le1_smoke_001/reports/v1/assistant_visual_audit.json)、[核对教师身份后的分析](../../runs/le1_smoke_001/reports/v1/post_key_review.json)。这只是实施者的定性审查，不是独立人工评测。

## Smoke 的实际发现

1. 计数例子中，OCR alpha 0.5/1 都写对了 MINT，却删掉杯子；逐约束保护拒绝这两个动作。GenEval alpha 1 同时保留杯子并写对文字，被接受。
2. 红杯例子中，OCR alpha 1 保留颜色并修正文字。
3. 关系例子中，GenEval alpha 1 补出了左杯右瓶，却把 MINT 变为 INT；OCR 保护拒绝它。
4. 存在疑似 OCR 假成功：额外 I/竖笔画被识别器漏读，同分 tie-break 可能选到视觉上较差的候选。该问题需要在更大诊断和独立图像复核中量化。

因此可以继续冻结诊断，**尚不能宣布方法训练有效、优于 baseline 或跨 sampler 稳定增益**。Smoke 的 selected local gain 是用同一个 verifier 筛选出来的描述量，不是独立证据。

## 本轮作业

| 阶段 | run / job | 状态 |
|---|---|---|
| 64 prompt × 2 seed × 3 positions，1920 张候选 | `le1_diagnostic_001` / 1047336 | 完成；实际设备 L40S，0 invalid，全部图片/记录哈希通过；与 H200 smoke 不合并统计 |
| 新拉取 MOPD 的 2 outer epochs，三教师各 4 条 train/test | `le1_native_opd_smoke_003` / 1047349 | 完成，已核验两个 checkpoint |
| 新拉取 OPSD open3 的 2 updates，4 条数据 | `le1_native_opsd_smoke_007` / 1047361 | 完成并通过重载评测 |
| 本地 round_robin/verified 各 2 epochs，真实续训 | `le1_training_smoke_job_001` / 1047351 | 两臂训练均完成；父作业在随后评测的缓存漂移检查失败，未重训或覆盖记录 |
| 两臂 raw/EMA 的 flow/DPM2 10 步统一评测 | `le1_evaluation_smoke_job_002` / 1047365 | 完成；所有协议比较通过 |
| 三档学习率、相同初始候选的一次更新 | `le1_fit_scale_job_001` / 1047375 | 完成；较小步长使接受目标的拟合误差下降 |

作业的实时状态以 Slurm 和 [ledger.jsonl](ledger.jsonl) 为准。本轮总分配约 **1.1897 GPU-hours**，包含失败尝试和模型加载；H100/H200/L40 类设备混合，不能当成硬件归一化的效率比较。运行产物保存在 `runs/`，代码、冻结实验设计、结果摘要和事件账本由本目录的本地 Git 管理。

## 384 查询的冻结诊断结果

64 个 development prompts、128 条噪声轨迹、3 个介入位置、1920 张候选；0 optimizer updates、0 invalid。所有 noop 终点跨位置逐像素一致，完成摘要与逐查询记录 digest 匹配，图片和输入哈希全部核验。分配 2559 秒，0.7108 GPU-hours；实测峰值 allocated memory 约 23.0 GiB。完整 [统计报告](../../runs/le1_diagnostic_001/reports/v1/report.md)、[逐组统计 JSON](../../runs/le1_diagnostic_001/reports/v1/summary.json)、[位置效应图](../../runs/le1_diagnostic_001/reports/v1/figures/fixed_actions.png)、[全部候选复核页](../../runs/le1_diagnostic_001/reports/v1/review.html)。

| 固定动作（在所有位置平均） | 平均 utility 增益 | 已满足约束破坏率 |
|---|---:|---:|
| OCR alpha=0.5 | +0.01949 | 7.81% |
| OCR alpha=1 | +0.05689 | 6.51% |
| GenEval alpha=0.5 | +0.02674 | 7.55% |
| GenEval alpha=1 | +0.07056 | 9.64% |

早期 index=1 / sigma≈0.960：OCR alpha1、GenEval alpha1 平均增益约 +0.140、+0.144，但已满足约束破坏率分别为 14.1%、20.3%。晚期 index=7 / sigma≈0.465：两者平均增益仅 +0.00251、+0.00392，区间均包含零。这支持调查“哪些局部动作值得模仿”，也说明固定的早期策略是必须认真比较的控制臂。

逐状态选择的平均局部 gain 为 +0.14522、接受率 36.20%；它使用同一个 verifier 挑赢家，**不能用来证明筛选法的训练增益**。统计区间按 64 个 prompt 聚类，而不是把 384 个查询当成独立 prompts；完整明细和探索性区间在报告中。

## 完整诊断的定性复核

在汇总结果产生前，按固定 hash 从每个任务组 × 介入位置选 1 个查询，共 12 查询/60 张候选；选择不读取 reward。先看隐藏身份的拼图，写下观察，再解开映射；两张疑似误判另看原始 512×512 图。见 [复核记录](../../runs/le1_diagnostic_001/qualitative_v1/assistant_visual_audit.json) 和 [核对评分后的分析](../../runs/le1_diagnostic_001/qualitative_v1/post_key_review.json)。实施者看过先前 smoke，可能认出重复构图，**这不是独立人工研究，也不据此估计总体误判率**。

- 确认一个清楚的 OCR 假阴性：图中 LUNA 被识别成 LONA；同一查询的 5 个近似候选均如此，不能算 5 个独立 prompt。
- 疑似假阳性：额外 I/竖笔画被忽略，识别为 MINT；该候选在这次复核例子中并未被选中。
- 有可信的局部修正：MINT 红杯例子修正文字并保留红杯；CLEAR 黄书例子修正文字，但 detector 未报告物体/颜色成功。
- 有明显的任务取舍：DAWN 例子中 OCR 修好文字仍只有一个苹果；GenEval 得到两个苹果但文字错误。它们都不是完整的联合成功。
- “完全可见”、白色标牌、黑色粗体等自然语言要求没有全部编码为显式评分子句。不能把 modeled joint success 宣称成完整 prompt 的逐字满足。

## 数据与比较纪律

`data/joint_pilot_v1/records.jsonl` 是构造的开发集，train/validation/test 为 64/32/64，平衡 OCR+计数、颜色、关系、计数+颜色。文本词表和 prompt 模板跨 split 分开，物体类别共享；不是公开 benchmark。

诊断的 64 条均为 train/development，不作为最终测试。正式 test 尚未用于本轮模型选择。工程训练使用该集合的 4 条 train 和 4 条 validation 子集；两次更新只检查路径正确，不能判断有效性。

候选生成、候选筛选、拟合后的同状态收益、held-out 完整策略收益分别报告。固定动作的收益/损害按实际 sigma 和 query index 分开；置信区间按 prompt 聚类，不把 seed/时间位置当独立 prompt。正式训练和 flow/DPM2 10/20/40 泛化矩阵仍待诊断与 verifier 复核后决定。

## 已记录并处理的工程问题

- H200 限定请求排队过久：取消未运行请求，改为允许可用 GPU，保留两次提交记录。
- NFS 临时目录导致 GenEval 子进程清理警告：后续 Slurm 使用每个作业独立的节点本地 `/tmp`。首轮 60 图有效性检查通过。
- NFS 目录元数据传播延迟：首次读取最后一个已完成 query 暂不可见；等待传播后再次核验完整 60 图，未跳过缺失证据。
- OPSD 的 48×24 默认分组在单卡配置构造时即失败：仅工程 smoke 允许显式按 8-GPU recipe 构造后，覆盖所有 batch/group 字段，再单卡运行。完整复现不使用此选项。
- OPSD 原入口要求每个 prompt 至少两张图；group=1 在模型加载后触发断言。prepare 已增加提前检查，下一 smoke 使用 group=2/batch=2。
- 原版 OPD 的 Accelerate unwrap 会导入环境中已安装的 DeepSpeed；缺失 CUDA_HOME 导致训练前停止。绑定本地现有 CUDA 12.6 工具链，不更改上游算法；失败 run 保留。
- OpenCLIP 的 URL 权重缓存与 Transformers 的 HF 缓存不同：新增独立资产绑定，旧 smoke 补充完整 SHA256 审计。
- OPSD 新 Transformers 与旧 timm 不兼容：仅在 `.venv-opsd` 安装 timm 1.0.19；原 OPD 环境不变。
- Hugging Face 的缺失 chat-template 缓存标记导致资产漂移检查停止评测：审计确认旧文件均未变化，仅新增两个 `.no_exist` 标记。新实验固定到实际 snapshot payload；旧实验仅对这类追加标记允许显式审计通过，权重变化仍拒绝。
- Diffusers 的 `_use_default_values` 无序列表导致相同 scheduler 被误判：仅规范化这个元数据列表，实际参数仍严格比较。失败评测保留，重新在同一 L40 作业内跑完全部比较，不混合 H200/L40 结果。

native launcher 已补齐两节点启动、rank/master 解析、资源计划核对和逐节点退出记录。8 GPU/2 node 的完整数据计划已准备且 Slurm test-only 通过，见 [计划](native_two_node_plans.json)。**未提交完整训练，实际跨节点 GPU 执行尚未验证**；单卡 smoke 不能冒充完整复现。

## 下一决策

本轮已收齐证据：局部修正确实有机会，也存在损害；较小步长可改善当前批次拟合，但 verifier 的生成图误判仍未解决。下一阶段先完成 verifier 校准和保守步长的匹配短训，再扩展固定教师/早期位置、随机接受、差异筛选等对照。**尚未启动 100-epoch 正式训练或 8-GPU 完整原版复现**，也未消耗保留 test。
