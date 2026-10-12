# 每批重复拟合的前置接口核查：06:00 EDT CPU记录

目的：检查04:00草案的固定实际LR、每批1→4次拟合，是否能直接使用当前框架及旧诊断。
这是实施准备，不是新的SD3训练、实验收益或正式冻结协议。GPU提交0，agent读图0。

## 已实际检查

通过真实train_loop配合ToyBackend运行小规模接口测试；不加载SD3、视觉评分器或图片。
3项测试全部通过（tests.txt）：

|检查|CPU观测|
|---|---|
|批内重复拟合|2外轮×4查询×4拟合，8次optimizer更新、32条backward query|
|候选/选择频率|仅8条训练query、2次behavior刷新；每外轮只选择一次|
|state/target|每批4次拟合复用完全相同state、时间步和detached target|
|实际LR匹配|每次更新的epoch/group/fitting/接受量/总量/LR/did_update在V/R逐项相同|
|真实状态记录|仅2份批前behavior snapshot，没有单独的每次拟合后完整权重|
|旧接口保护|Phase68审计与Phase69权重链明确拒绝fitting_epochs=4|

上述结果只验接口；toy模拟的修复标签不作为方法效果，未计算/汇报图像收益。
本次没有修改活跃训练代码，也没有放宽旧诊断的保护条件。

## 一个必须补的接口，而非仅改配置

当前sparse_update_study.validate_observations限定fitting_epochs==1，且要求每条更新的
fitting==0。accepted_state_mechanism.qualify_update_chain也只允许每外轮单group单次fit。
因此Phase68/69原配置有明确支持范围，已完成结果不受影响；新四次拟合不能原样套旧报告。

engine只在fitting==0保存该批state/target/behavior。下一轮批前behavior能对应整个拟合
块结束后的权重，不能当成“第1次optimizer更新后的权重”。各次parameter_step_l2之和
也不等于整个拟合块的参数位移，不能用范数相加伪造严格权重链校验。

未来最小接口方案：保留单group/外轮，明确命名post_fit_block（全部K次更新之后）。
保存块前/块后的digest及实际整块参数位移，以后继behavior或最终RAW逐一核对。
如还要测第一步后的终点，必须额外保存对应真实权重；缺少该快照时不要宣称已有此证据。
仍记录每次optimizer step/LR/梯度缩放是否成功，random逐条与verified匹配。

## 进入真实训练前的条件

1. 先补齐Phase69同图五质量所需独立执行协议与授权；当前仍0/2550，技术失败不重试。
2. 实现独立多次拟合审计入口与前述权重记录，并测试缺失快照、错配步数、空批和实际LR。
   已有Phase68/69专用审计保留原限制，避免改变历史定义。
3. 明确只最终16外轮RAW；4次fit意味着每个非空外轮4次更新，不再把16轮称为16次更新。
4. 冻结同起点/同实际LR/同量随机的比较、全部任务与五质量、计算上限和停止规则。
   跨1/4次拟合的在线接受集合会改变，不能冒称相同目标集合的纯重复拟合效应。
5. 仍需独立开发joint增益；训练修复恢复、MSE下降、几何投影增加都不足以进入确认集。

Phase68每臂实际约5600 L40S秒；本夜只余单作业4500秒，不能假定完整两臂训练评测可容纳。
没有新增GPU、reserve、agent或timer，原已请求18900/23400及失败额度不变。

## 今后的续看

本次已经完成必要接口检查。若后续没有外部状态变化，核对账务/预算/监控并写接收记录
即可；不重复跑这些测试或为每小时产生新结果而反复分解同一数据。
原Phase68完整对照、Phase69任务部分/质量缺失和独立CPU修复继续作为当前完整状态。

[原草案](../local_expert_v69_cpu_failure_audit/NEXT_PROTOCOL_DRAFT.md) ·
[动作几何CPU证据](../phase69_action_geometry_20260930T0900Z/SUMMARY.md)。
