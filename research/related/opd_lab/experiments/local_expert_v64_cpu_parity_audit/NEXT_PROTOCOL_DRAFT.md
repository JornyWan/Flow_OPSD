# 下一步：仅重放评分路径，不重做训练或生成

状态：CPU草案，未提交/未预留GPU；不回收Phase64失败预算。建议单L40至多900秒，
新预算独立记录。旧24dev图已存在，额外图像生成0、optimizer更新0。

问题：完整A证据不等是否来自trace包装或PyTorch执行模式？是否影响bool/None和修复判断？
固定原第一查询的noop/teacher两张训练候选、原expected A、24个dev终点和全部hash。

1. 同一冻结scorer代码和模型，在独立子进程分别采用训练默认执行模式与原评测强制
   deterministic模式。每种模式比较直接EvidenceVerifier和TaskScorers带trace路径，
   保留expected/actual、标签、检测/识别数值、运行flag、模型/config/source指纹。
2. 先保存完整差异再执行原严格门槛。若只有浮点差也不追认Phase64通过；若存在标签
   或accept/reject变化，则先修评分一致性，不新增训练。不要按开发分数选运行模式。
3. 后续对已存24图做固定B任务和五质量，不重新生成、不调阈值、不选择checkpoint。
   评分恢复必须是另一个清楚标注的记录；原失败保留。只有经诊断确认的契约路径才能
   用作后续正式比较，不能把此8题结果当作统计显著性或主表。
4. 一次技术失败停止，无自动重试；新提交前再检查四partition和sablab跨项目≤6。

完成评分后，正式四臂比较的CPU准备见../formal_online_comparison_20260929/README.md。
它仍未授权或提交额外GPU；正式512train日程需版本化接口/测试、全资产指纹与实际成本上限。
