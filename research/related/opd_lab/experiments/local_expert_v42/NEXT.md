# 下一步：运行开发端480张候选图

冒烟测试已通过；不再重做权重下载、重复旧图补评或用四例效果调整规则。

1. 使用只读`data/joint_correction_v1`及其frozen.json、bindings.json。只读取validation/48提示词用于开发，每题两seed。test/192提示词继续不生成，直到开发结果选定T*。
2. 将已测`trace_flow/correct_flow`接入一个受限采集runner：每个case只生成一条behavior参考轨迹，再共享前缀生成start{1,3}×length{1,2}四条OCR路线。保存真实窗口监督、后缀与所有图SHA，关闭smoke额外重跑；第一例执行一次绑定/状态契约检查，其成本另记。不能把所有case当作独立prompt。
3. 用已测A/B评测适配器评分全480图；A逐约束修复＋保护＋unknown弃权，B原图OCR exact与GenEval strict及分项。全量保留证据，不改OCR模型/阈值。五项质量指标复用现有runner，不挑选方法有利的列。
4. B联合成功率最高的非noop路线为T*；并列先短、后早。在T*与noop中按开发分数决定固定对照，并列noop。写出冻结选择记录及源结果SHA。开发结果只是选择/机会诊断，不是独立效益确认。
5. 先检查开发实际成本、错误、候选机会和预算，更新`experiments/joint_correction_v1/budget.json`。共享预算剩余10698秒；每个作业walltime及失败尝试都受剩余总额限制。确认生成必须预留完整评分/质量成本。
6. 符合既定继续条件且资源足够时，执行确认384个case×2分支；按照预定主要比较与未知敏感性分析，再决定是否进入短程学习。未收回人类盲评前，不能把代理提升叫独立视觉确认。

当前smoke已经完成，开发采集尚未提交，尚无新训练或新方法效果结论。完整研究顺序继续遵循docs/next_experiment.md与docs/paper_plan.md。
