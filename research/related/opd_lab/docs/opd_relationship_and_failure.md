# 当前实现与DiffusionOPD：2026-09-30核对

当前方法属于外部专家蒸馏，复用官方确定性OPD的局部transition-mean损失。
它目前是单查询＋严格正例筛选的改造，不等同于官方完整训练协议，也不是OPSD
reward-gradient self-distillation。behavior副本参与采样/验证，目标动作来自外部教师。

Phase66各臂共同从纯SD3.5-M＋fresh LoRA开始，最终均为物理单adapter完整生成。
早期发布OPD Student续训、固定j=1和双adapter时段部署属于历史实验，不能用于
描述当前Phase66。原生对照使用官方SOPD-GenEval trainer，但只是64题/16轮适配短跑，
不是充分训练的三教师MOPD或论文指标复现。

核心发现：每轨迹9个监督点的原生适配得到9216个目标；本地all只用1024个；
验证和同量随机各只剩22个，二者均发生严重退化。837/1024个验证查询含unknown，
815涉及文字；156个有已知修复且无确认损伤，却仍因unknown拒绝。拒绝loss为0
不等于未选状态的动作不变。Adam实际步幅也不会因loss除64而必然同比降低。
这些证据支持优先检查稀疏学习覆盖/更新稳定性，尚未证明某个单一优化器原因。

五质量没有纳入当前接受门槛；单GenEval教师的结构迁移可能损失文字能力。
局部动作更像教师也不保证完整轨迹有效，Phase67对此提供了顺序相关的分段证据。
现有代码检查未发现能独立解释退化的bug，不能将“通过测试”称作算法有效。

下一轮候选步幅开关已实现并通过CPU测试，尚未提交。缩小步幅后仅恢复起点不算
论文贡献；仍必须证明verified对匹配random及强OPD基线有完整任务增量且质量可接受。
保留广覆盖OPD训练、让终端证据校准教师监督是后续待独立设计的主线，不能与
当前诊断同时混改。所有旧负结果保留，独立confirmation/final仍封存。

[本次完整审计、数值和边界](../experiments/opd_relationship_audit_20260930_v2/SUMMARY.md) ·
[上一轮完整结果](../experiments/phase66_67_closeout_20260930/report.html)。
