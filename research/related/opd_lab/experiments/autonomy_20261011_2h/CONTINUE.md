这是用户2026-10-11明确授权“先完成干净实现，再针对性尝试/debug；每2小时检查并决定推进”的同一研究任务。先读取opd_related/opd_lab/AGENTS.md、experiments/CURRENT.md，以及experiments/autonomy_20261011_2h/PLAN.md、plan.json、budget.json、HANDOFF.md；在acks/本次编号.json记录实际UTC接收。

全程零图片/截图/远程视觉/子agent读图，人工仅辅助；不创建新agent/timer/另一会话。先检查真实Slurm、submission/intent/watch、管理器和紧凑统计，避免重复。Phase79为1077741（preempt_gpu，单L40S，16200秒，无requeue）；完整256更新两步闭环学习，正式主比较comparison/result.json为最终EMA−Phase77原生EMA49/128，不是只比底座。128dev固定重复使用，确认512/final768封存。两步增加teacher queries、学生计算和j9覆盖，不能称同算力/同信息；终端验证筛选尚未加入。Phase77原生S0/EMA/RAW joint17/49/45，不属于创新收益。

运行中只监控和有必要的独立CPU工作，不改冻结输入/配方/门槛。完成后审计256真实更新、全部prompt/实际LR与Phase77匹配、EMA/权重链、GPU梯度合同、S0跨轮图像/任务精确一致和质量参考容差、384端点/1920主要质量、全部joint/OCR/结构/CER/unknown、五质量、配对区间、家族与新增/损失及真实账务。不可只看Slurm完成或within-run EMA−S0结果。CPU错误在独立修复目录处理；已完成档案不改；技术失败不自动重试。

用户提醒：单次失败不等于idea错误；也不能通过不断试验/挑结果制造显著性。按PLAN.md先区分实现、优化拟合、部署、任务取舍、统计不确定性。有正向信号优先同目标/同覆盖的停止中间梯度控制与计算匹配原生对照；未定或负向先分析目标尺度、梯度/拟合、EMA/RAW及部署，再仅在证据支持时冻结一个能区分假说的单变量试验。新种子须匹配对照，不把单臂新seed和旧seed比较当方法证据。小幅CLIP下降不自动否决，但全部五质量和反例必报。不能同时改LR/筛选/时步/EMA，不能扫参数/挑checkpoint/改评测器，不将描述性开发CI当独立验证，不要求用满预算。

本campaign到plan.json expires_utc为止（最后一次仅收尾）；总请求上限48600 L40S秒，含Phase79既有16200；最多再2新GPU作业、各≤16200秒，不退款/结转失败或未用请求。每次先写独立协议、实现和CPU检查，明确起点/匹配对照/唯一变量/主指标及所有质量/算力/停止规则，再用本目录runner.py reserve --name local_expert_vNN_arm --seconds N原子预留；逐个作业记录submission并更新watch。reserve拒绝就停止GPU推进，不自动延长。两臂比较必须预算足够完整完成。技术修复后新作业须独立说明新证据与修复，并占新额度，不盲重试。

每次提交前probe minilab-gpu/sablab-gpu/scu-gpu/preempt_gpu；sablab跨项目allocated+pending≤6（含low），保持L40S协议，无requeue/自动重试，不动其他项目。到期不新增提交或定时器，已提交作业按自身硬时限结束。无新外部变化不重复同分析测试。每次写checkins/编号.md、HANDOFF和CURRENT；有新证据再同步README/项目概览/论文计划。简洁汇报新证据、反证、实现状态、预算及下一问题；不保证正向结果。
