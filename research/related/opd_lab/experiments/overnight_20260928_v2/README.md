# Phase55 每小时续看 · 第二次授权

用户2026-09-28再次要求夜间自动查看和推进，本次设置每小时一次、最多六次；不扩大Phase55已有三个池/16200 L40秒的预算。

- 本地guard PID：1795046；timer PID：1795059，启动于16:24 UTC。
- 计划UTC时间：17:24、18:24、19:24、20:24、21:24、22:24；到期22:34 UTC。
- Phase55变为complete/failed_stopped/failed/cancelled即写STOP，停止剩余周期提醒；完成审计由现有manager的一次完成回调接手。状态交接时可能各投递一条，接收方先查已有ack与审计，不重复工作。
- 实际周期唤醒：**启动时0次**。未来以[scheduler.jsonl](scheduler.jsonl)的queued和`acknowledgements.jsonl`的接收确认分别核对；CLI返回0不等于对话已执行。
- [计划](plan.json)、[续看指令](CONTINUE.md)、[guard状态](guard_latest.json)、[timer状态](scheduler_latest.json)、[启动记录](launch.json)、[6项检查](verification.json)。

实验manager负责Slurm与串行池推进。定时器只调用现有codex queue，没有提交、取消、重试GPU的代码；限定精简数值审计、CPU分析/文档，确认/最终集封存，零读图。完整结果后先写清科学问题、比较和预算，本定时器不授权下一轮GPU。

这是本地有界脚本，不是产品原生Scheduled任务。官方文档说明CLI/IDE不提供Scheduled管理界面：[Scheduled tasks](https://learn.chatgpt.com/docs/automations?surface=app)。本机制依赖服务器和已有Codex服务保持可用；原Phase54同线程queue已实际投递并收到确认，新定时器仍须看各次ack。旧overnight_20260928到期且0次周期唤醒的档案不改写。

验证覆盖缺少/写入中进度文件、终态识别、已完成时不启动、运行中完成时停止子进程，以及真实六次计划有界检查。测试使用假timer，真实queue尝试0次。

人工停止：在本目录创建STOP文件（不影响GPU实验manager）。不要重启旧定时器或同时再建第二份计划。
