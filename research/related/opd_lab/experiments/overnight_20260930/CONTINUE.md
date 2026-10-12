这是用户2026-09-30睡前明确授权的夜间每小时监控推进，同一研究任务，不是新实验结果。先读取opd_related/opd_lab/AGENTS.md、experiments/CURRENT.md和experiments/overnight_20260930/PLAN.md及budget.json，在该目录acks/对应编号.json写真实UTC接收确认。全程零图片/截图/远程视觉/子agent读图，人工标签仅辅助；不新建agent或timer。

先查真实Slurm/已有submission与intent/管理器，Phase68 verified1065020、random1065024。不重复提交，不改变冻结配方/门槛/已完成档案，技术失败不自动重试，CPU报告错误另立修复目录。Phase68已占两作业各7200 L40S秒；只最终16轮RAW，128相同dev，全部joint/OCR/结构/CER/unknown和五质量。随机依赖验证成功且新两臂逐批实际学习率匹配，不能将与旧native不同LR的差归为筛选优势。

若Phase68仍运行，仅监控和做有必要的独立CPU工作。若两臂均COMPLETED 0:0，先运行冻结report_sparse_update_study.py输出到独立local_expert_v68_cpu_report，核对全部输入/账务/门槛；不因恢复初始能力就宣称超过baseline。当前可准备的实际接受状态诊断在scripts/prepare_accepted_state_mechanism.py与python/opdlab/accepted_state_mechanism.py，16项CPU测试已通过但尚未GPU验收，提交前再检查实现、冻结并测试。它比较保存的behavior/teacher/刚更新后局部/最终局部/最终后缀，最多64真实选中状态/臂，按prompt聚类，不能跨臂当同状态对照；正式提交前必须实际完成来源、核对整个权重链、冻结来源/规则/预算和四分区probe。不要冒称它已执行。

本夜额外上限2新作业，每个至多4500 L40S秒，总含Phase68原14400为23400请求秒，不回收余额或失败额度。新GPU必须先通过本目录runner.py reserve --name local_expert_vNN --seconds N预留，保存submission并更新watch。其中优先一次有辨别力的真实接受状态前向诊断；第二次仅在完整证据足以支持单变量、同起点/同实际LR/匹配控制的有限实验时使用。不得为用满预算提交，不扫LR或CFG网格，不通过放宽门槛/挑checkpoint/少报质量制造正结果。若缺乏可区分假设的设计就CPU复盘/写协议。

每次提交前probe minilab-gpu、sablab-gpu、scu-gpu、preempt_gpu；实际设备L40S，sablab跨项目allocated+pending承诺不超过6，不动其他项目。无requeue/自动重试。截止2026-09-30 09:05 America/New_York停止新增GPU或定时排队，提交的作业按自身时限结束。confirmation512/final768封存。每次写checkins/编号.md并更新MORNING.md和CURRENT；新证据产生时同步README/项目概览/论文计划，明确完成/失败/待验收、预算和论文缺口。少读长日志、只读精简数值。不要保证或挑选正向结果。
