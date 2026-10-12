这是用户2026-10-01明确授权“先研究透OPD为何有效及本地偏离，再使方法有效；自动推进12小时，每20分钟一次”的同一研究任务续看，不是新的实验结果。先读取opd_related/opd_lab/AGENTS.md、experiments/CURRENT.md及experiments/autonomy_20261001_12h/PLAN.md、budget.json、RESEARCH.md。先在acks/对应编号.json写真实UTC接收时间。

零图片、截图、远程视觉或子agent读图，人工仅辅助；不创建新agent/timer，不启动另一Codex会话。检查真实Slurm、submission/intent、管理器、watch.json及紧凑数值，防重复。运行中只监控及做独立CPU研究/实现；技术失败不自动重试，CPU报告错误独立修复，完成档案不可改。无外部变化时不重跑相同测试/分析，不制造研究进展。

主线：不能把少量教师动作直接介入的成功当训练成功。先核对官方固定commit的状态覆盖、外部专家、时间权重、实际LR、batch、数据与训练量、EMA及完整部署；区分理论充分条件、实测与假说。当前Phase68完整dev验证8/128、随机9/128，Phase70自身训练状态teacher1→33但最终局部1，五质量已完整；随机自身教师也有收益。严格正例仅51/1024、更新16次，不等于原生OPD；Phase54更多拟合恢复训练修复但dev输随机是反例。不要默认为加fit就是答案。

已完成record_fit_blocks与独立审计（118CPU测试、102真实保存状态检查），尚未把4fit接入完整GPU机制/报告。新4fit研究合同见repeated_fit_implementation_20260930/LEARNING_PROTOCOL.md。若继续这个候选，必须先完成模式/权重/来源/评分cohort/全部报告的集成验证；唯一算法变化为每批fit1→4，保留实际LR规则、同量分层随机、最终16轮RAW、完整128dev及全部joint/OCR/结构/CER/unknown和五质量。post_fit_block不能称第1步后；跨1/4fit在线目标集合会改变。不能把旧native不同LR的差归为筛选增益。

若研究证据支持改变优先级（如保留广覆盖OPD、只屏蔽明确损伤而非unknown全部弃用），先写独立对照/协议和可证伪判据。记录为何它比继续稀疏拟合更有辨别力；不改旧合同/门槛，不扫参数/挑checkpoint/更换评测器，不承诺正结果。设计不足就继续CPU理论/代码审计，不能为耗尽预算提交。

本12小时独立GPU请求总上限32400 L40S秒，最多4个新作业，每作业≤10800秒、1张兼容L40S；旧/失败/未用预算不退款或结转。每次必须先独立冻结研究问题、原始起点、匹配对照、唯一变量、终点及五质量、预算/停止规则，必要CPU检查通过，然后用本目录runner.py reserve --name local_expert_vNN_arm --seconds N原子预留，再记录submission与watch.json。reserve拒绝即停止GPU推进。每次提交前probe minilab-gpu/sablab-gpu/scu-gpu/preempt_gpu；sablab跨项目allocated+pending承诺≤6，不动其他项目，无requeue/自动重试。不能合理排到或不能保证完整主比较时先CPU准备。

以plan.json的expires_utc为12小时截止：到期不再reserve/提交GPU或安排新唤醒，最后一次只收尾；已提交作业按自身硬时限结束。确认512/final768持续封存，同128dev重复使用的区间只是探索性，不把训练池当新增独立prompt。更新本目录checkins/编号.md、HANDOFF.md和CURRENT；有新证据时同步README/项目概览/论文计划。简洁记录新证据、反证、实现状态、实际账务/请求预算、下一可证伪问题及论文缺口。少读长日志；先完成接收与检查，再推进有价值的独立工作。
