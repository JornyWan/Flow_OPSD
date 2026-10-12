这是本对话用户2026-09-29明确要求每小时查看并继续推进的一次夜间续看，同一研究任务，不是实验结果。读取opd_related/opd_lab/AGENTS.md、experiments/CURRENT.md、experiments/overnight_20260929/PLAN.md和budget.json。先在本目录acks/对应编号.json记录真实接收时间。全程零图片、截图、远程视觉或子agent读图；人工标签仅辅助。

先查真实Slurm、管理器进程、已有submission/intent/账务和精简数值，避免重复提交。当前Phase58作业1064447，local_expert_v58，manager PID600772，日志/tmp/opdlab_phase58_manager_20260929.log；已关闭Slurm自动Requeue。若仍运行则只监控和做独立CPU研究工作。完成后核对manifest/报告/全部五质量、任务joint、原接受/拒绝子集、三训练池一致性，解释CFG1相对4.5。技术失败不重试；CPU分析错误另立修复目录。完成档案不可修改。

本次用户的新授权允许根据结果继续必要的匹配实验，但不是无限GPU预算：本夜代理主动设上限10800 L40 GPU秒（包含Phase58预留1800），最多另两个新作业、每个至多4500秒，按请求上限累计，不回收旧余额或失败预算。不保证或挑选正向结果，不改已冻结门槛。下一步先写独立协议和测试，固定研究问题、原始起点、对照、唯一变量、全部任务与五质量、采样配置、GPU上限与可证伪停止规则，再通过本目录runner.py reserve --name local_expert_vNN --seconds N预留，记录每个submission与watch.json。单轮可有多个臂但每个新GPU作业单独记账；reserve拒绝时停止GPU推进。截止2026-09-29 09:05 America/New_York停止新增提交和定时排队，已提交作业按自身时限结束。若队列估计不能在夜间合理完成，先准备协议/CPU工作，不盲目占队列。

Phase58门槛通过也只说明候选值得另立学习比较：可固定CFG1对比验证/同量随机，用相同起点/训练池/16步和已有CFG4.5对照，主看已有128dev的joint和五质量，不因候选变好就宣称学生变好。门槛失败则不切换默认CFG、不扫CFG网格；根据完整证据考虑已存research_reset_20260929/NEXT_PROTOCOL_DRAFT.md的“明确损伤保持 vs同量同分层随机保持”，固定原正例和训练配置，并先说明为何此对照能区别负例信息与保持强度。若现有证据不足以设计有辨别力的实验，做CPU分解、原生OPD与本地改造的配方核对和论文实验矩阵，明确缺口；不要为用满预算而提交。不要更换评测器或开confirmation/final，不把同128题的多池当更多独立prompt，不将诊断机制臂冒称原生OPD。

每次GPU提交前检查minilab-gpu、sablab-gpu、scu-gpu、preempt_gpu，sablab跨项目并发及pending预留不超6，保持协议GPU兼容性/无自动重试。不要取消其他项目作业，不复制已运行任务，不另开定时器或agent。更新README、docs/project_overview.md、docs/paper_plan.md与CURRENT，写本次checkins/编号.md和MORNING.md：新证据、反证、预算、下一步和论文最缺什么。少读原始长日志，多读紧凑统计；需要用户知道的结论在本对话简洁汇报。
