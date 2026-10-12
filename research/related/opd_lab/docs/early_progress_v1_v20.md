# 早期阶段进度记录（Phase1–20）

此页保存原README的历史进度文字；其中“当前”指当时状态。最新状态见[项目概览](project_overview.md)和[实验入口](../experiments/CURRENT.md)。历史助手读图记录不代表现在的评测流程；现在默认助手直接读图0次。

当前[第二十阶段已完成，筛选组主检验失败](../experiments/local_expert_v20/STATUS.md)：三组各128更新，224张新seed图全部冻结复核。主要12个新seed单步完整数为初始6、全教师9、筛选6、随机6；筛选完整生成8/12但与全教师持平，全部32个单步仍有2例损伤。122项测试及统计重建通过，总分配0.894167 GPU-hours。原训练末尾计数断言失败已核验恢复，0重训、原FAILED保留。[恢复说明](../experiments/local_expert_v20/RECOVERY.md)、[全部配对图](../experiments/local_expert_v20/learning_cases.html)、[结果图](../experiments/local_expert_v20/learning_outcomes.png)、[下一步损失消融](../experiments/local_expert_v20/NEXT.md)。当前无该阶段作业运行。

当前[第十九阶段已完成](../experiments/local_expert_v19/STATUS.md)：第十八阶段增加可靠状态后，训练目标拟合改善，但主要新seed仍为5/12，全部新seed有4例已知损伤。随后在同一12个状态上补做教师配对诊断：教师在2例能安全完成、学生单步仍失败；安全教师oracle为7/12，学生仍5/12。原始GenEval单步6/12且有1例损伤，仍不能无条件模仿。所有标签先冻结后揭盲，113项测试与档案验证通过。见[配对证据](../experiments/local_expert_v19/teacher_student_gap.png)、[第十八阶段](../experiments/local_expert_v18/STATUS.md)、[下一步](../experiments/local_expert_v19/NEXT.md)。

此前[第十七阶段候选覆盖采集已完成](../experiments/local_expert_v17/STATUS.md)：12张旧参考精确重现、156张新图身份隐藏复核，72次教师查询、0次更新。固定index3候选覆盖1/12个确定安全修复状态，扩展时间/强度后3/12，新增两例来自index1 GenEval；主差值95%区间包含零。基线另有4例已完整、4例未知、1例仅裁切，不能都计作专家失败。标签辅助选择的完整终点4/12→7/12尚非学生收益。已核对跨阶段11个监督状态，其中9个固定index3；102项测试通过，成本0.068333 GPU-hours。见[结果图](../experiments/local_expert_v17/candidate_coverage.png)、[下一步多状态学习](../experiments/local_expert_v17/NEXT.md)、[执行记录](phase17_runbook.md)。

此前[第十六阶段单步动作迁移诊断已完成](../experiments/local_expert_v16/STATUS.md)：42张旧参考全部精确重现，42张新图身份隐藏复核，24次新教师查询、0次更新。新seed完整场景：初始化、完整学生、单步学生均2/12；教师只在2/12状态有确定安全修复，另4/12未知。学生相对同类专家的动作MSE中位数由旧状态0.05075升至新状态0.87996。存在一个旧seed的轨迹干扰案例，但单步隔离没有解决新seed收益。97项测试通过，成本0.043889 GPU-hours。见[结果图](../experiments/local_expert_v16/action_transfer.png)、[下一步多状态候选采集](../experiments/local_expert_v16/NEXT.md)、[执行记录](phase16_runbook.md)。

此前[第十五阶段标签辅助学习实验已完成](../experiments/local_expert_v15/STATUS.md)：使用全部 6 个已标注安全动作和 6 个保留状态，完成标注筛选、匹配随机、不筛选教师三组各 128 次更新，以及 152 张配对完整生成复核。筛选组目标相对 MSE 中位数降至 0.05075，拟合 seed 完整场景 0/6→5/6；同提示词新 seed 仍为 2/12，随机对照 3/12，全部新 seed 有 5 次确定损伤，未过扩展门槛。93 项测试通过，成本 0.476667 GPU-hours。见[结果图](../experiments/local_expert_v15/annotation_probe.png)、[同提示词 seed 反转](../experiments/local_expert_v15/fitting_vs_fresh.png)、[下一步机制诊断](../experiments/local_expert_v15/NEXT.md)、[执行记录](phase15_runbook.md)。本轮是助手标签辅助的固定状态实验，未宣称自动方法、原版 OPD 对比或 sampler 鲁棒性成功。

此前[第十四阶段成对证据审查已完成](../experiments/local_expert_v14/STATUS.md)：192次无目标词观察、128个专家候选对的双向审查256次、64次同图控制，共512次Qwen请求。冻结主协议因外层JSON容器不匹配而全部拒绝；单独兼容数组的事后诊断最终接受3个，2安全、1未知，安全状态召回2/6，仍未过门槛。总体顺序/同图控制通过仍不能保证修复判定正确，小字和裁切漏检持续存在。无新生成或训练，86项完整测试+2项兼容测试通过，成本0.591667 GPU-hours。见[结果图](../experiments/local_expert_v14/paired_audit.png)、[下一步：可靠监督与机制上限](../experiments/local_expert_v14/NEXT.md)、[执行记录](phase14_runbook.md)。未启动新seed确认或自动训练。

此前[第十三阶段测量开发](../experiments/local_expert_v13/STATUS.md)：在原192张图上实现并实测独立版本的多视图 verifier。原v1接受20个（6安全、3损伤）；冻结v2接受6个（3安全、0损伤），精度50%、召回50%；事后保留未知基线的逻辑找回一个安全目标，4/7仍不足。原[结果图](../experiments/local_expert_v13/evidence_audit.png)与[执行记录](phase13_runbook.md)保留。

此前[第十二阶段前瞻候选审计](../experiments/local_expert_v12/STATUS.md)：全部64个训练提示词的新seed、192张图盲审。接受20个动作中6个安全修复、3个确定损伤、6个无确认修复、5个未知；完整场景21/64→25/64。四个受损原正确子句均被基线 verifier 判为 false，未受保护。原[结果图](../experiments/local_expert_v12/selection_audit.png)与[执行记录](phase12_runbook.md)保留。

此前[第十一阶段迁移评测](../experiments/local_expert_v11/STATUS.md)：两个 sparse checkpoint 在五个拟合提示词的新 seed 上将完整场景 2/10 提到 4/10，32 个既有验证提示词 9/32 提到 11/32、12/32。仅有部分跨 seed 迁移，存在原正确约束回归，区间跨零。原[结果图](../experiments/local_expert_v11/transfer.png)与[执行记录](phase11_runbook.md)保留。

此前[第十阶段](../experiments/local_expert_v10/STATUS.md)在五个选定训练案例上验证了稀疏监督后的完整生成修复；相同反向查询预算的完整轨迹组漂移更小，但生成了两个碗。原[图片对照](../experiments/local_expert_v10/apple_endpoints.png)与[执行记录](phase10_runbook.md)保持原样。

此前[第八阶段轨迹诊断](../experiments/local_expert_v8/STATUS.md)和[第九阶段专家复查](../experiments/local_expert_v9/STATUS.md)定位了旧目标在新状态上的失效，并找到新修正；[对照图](../experiments/local_expert_v9/diagnostic_chain.png)和[执行记录](phase8_9_runbook.md)保持原样。

此前[第七阶段](../experiments/local_expert_v7/STATUS.md)完成原版 SOPD-OCR 正向学习对照与固定好目标的充分拟合：联合拟合保留 4/5 个局部修复、3/5 个完整生成修复。研究方向调整见 [重新审视原始 idea](research_reassessment_20260912.md)。整体设计见 [实验 pipeline](pipeline.md)，历史结果保留在 [phase 6](../experiments/local_expert_v6/STATUS.md)、[phase 5](../experiments/local_expert_v5/STATUS.md)、[phase 4](../experiments/local_expert_v4/STATUS.md)、[phase 3](../experiments/local_expert_v3/STATUS.md) 和 [phase 2](../experiments/local_expert_v2/STATUS.md)。

