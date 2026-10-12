# 实际接受状态：目标、即时学习、后续训练与采样后缀

本轮只做前向机制诊断，不训练、不改变Phase68配方或门槛。要求Phase68 verified/random
两作业均COMPLETED 0:0且完整128开发/五质量评测完成，才能冻结来源并提交。
主研究问题：真实被训练的修正在哪一环未兑现？Phase67只覆盖共同初始状态；
本轮使用Phase68训练时保存的实际state/context/base_action/target/behavior。

每臂纳入全部实际接受查询；若超过64个，则按预定hash截取64个，不看终点结果挑选。
两臂使用各自历史状态，不是跨臂配对实验。重复训练prompt按原ID聚类，不能当独立新题。
最多128状态，五模式640终点，384新终点、256旧终点复用、256次重放原图哈希。
每次GPU重放必须严格匹配原noop与teacher PNG；不匹配就停止，不放宽容差。

模式依次为：behavior原动作+原behavior后缀；teacher原目标+原behavior后缀；
immediate_local刚完成该次真实更新的学生动作+原behavior后缀；final_local最终16轮
RAW学生动作+原behavior后缀；final_suffix最终学生动作+最终学生后缀。
所有模式共用保存的原前缀state；final_suffix不是全程最终学生。完整部署收益仍由
Phase68的128开发集比较负责，本轮不得替代。固定Flow10/512/text128/fp16/学生CFG4.5，
原目标来自GenEval教师CFG1；同一后端、同一任务评测器、全部五质量。

即时权重不通过重训练重建：限定每epoch一个更新机会、一个fit、无跳过的非空更新；
下一非空epoch保存的behavior就是上次更新后策略，最后一次取固定最终RAW。
每次前后权重的真实L2位移必须与原更新日志一致，验证所有输入hash、实际选择名单、
冻结behavior和最终学生权重。先核对整个链，再作GPU推理。空批无优化器更新。

逐段主诊断B joint：teacher-behavior、immediate_local-teacher、
final_local-immediate_local、final_suffix-final_local，另完整报告对behavior/teacher差。
报告OCR exact/CER、结构、unknown、PickScore/26、CLIP、Aesthetic、HPSv2.1、ImageReward
全部均值和按prompt/族聚类的配对描述区间；不挑分数。原teacher的B直接修复/损伤子集、
时间索引0–2/3–5/6–8、target MSE和漂移也报告。B子集是机制解释，不反过来训练选择器。

teacher没有收益：先查候选能力与A/B不同，不增加拟合；即时拟合未兑现：考虑目标几何、
步幅或单次拟合；即时有效而最终丢失：调查跨批干扰；局部有效而后缀变坏：调查多步策略
改变。效果可交互，可多环同时失败。该诊断不证明因果唯一，也不是新方法成功门槛。
通过任何诊断不自动启动下一训练。跨臂筛选效果仅看Phase68同实际LR日程的开发比较。

独立上限1作业×1 L40S×4500秒，含加载/重放/推理/任务和全部五质量，最多3200质量值。
不回收旧或失败预算，无训练、无自动重试/requeue/自动后续作业。技术失败停止，CPU报告
错误另立修复目录；原完成档案不变。提交前probe四分区、sablab跨项目分配+pending≤6。
允许CANCEL抢占，失败不重排。confirmation/final封存，人工辅助，agent读图0。
夜间提交另须先按当夜runner reserve预留，此脚本不提交作业或建立定时器。
