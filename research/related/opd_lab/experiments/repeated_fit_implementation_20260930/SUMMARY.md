# 多次拟合记录与审计：实现完成，CPU验收通过

更新：2026-09-30 22:45 EDT。本轮是实现与验收，没有新SD3训练、GPU作业或图像收益结果。
agent读图0、截图0、远程视觉0、agent0、timer0。已完成实验的原档案保持不变。

## 实现了什么

- 新增record_fit_blocks，默认关闭；仅支持已有semantic_batch_v1、单进程、每外轮一个
  累积组，并要求保存选中状态。记录功能不修改loss、学习率、目标、优化器或刷新频率。
- 每次fit前核对完整state/target/context/record/seed/time和选择mask，最后再核一次。
  相同批内不重采样、不重选目标。发生意外变化即报错。
- 保存整块更新前后真实权重，以及每个optimizer step的前后权重digest；整块位移直接
  从前后权重计算，不用各步范数相加。保存空批的恒等变化，没有伪造零学习率更新。
- 独立validate_blocks核对查询与epoch、逐fit实际LR/成功更新/计数、选中状态与目标、
  候选文件hash、相邻块权重链及最终RAW。verify/random还须同时匹配配置/来源/数据
  合同、任务分层数量与每个fit的实际更新日程。
- 新增scripts/audit_repeated_fitting.py，接受完成的run和可选verified-reference，检查
  最终checkpoint元数据及SD3 RAW导出，输出独立summary/sources/manifest，不修改原run。
- 4fit之后的权重命名为post_fit_block。它不代表第1次更新后；K>1没有第一步的单独
  权重快照。Phase68/69旧审计继续拒绝4fit，不改变旧科学定义或完成档案。

## 已实际验收

118项CPU测试全部通过，详见[cpu_tests.txt](cpu_tests.txt)。其中新多次拟合测试27项：
真实engine＋ToyBackend跑1/4fit、开启/关闭记录的参数/Adam/EMA/查询一致性、全空批、
中间空批、断点恢复、逐fit随机计数/LR匹配、完整合同不匹配拒绝，以及被修改的目标、
缺失更新/快照、错误权重digest/净位移/阶段名称/最终RAW/checkpoint元数据的拒绝。
旧稀疏更新、在线选择、查询日程、真实状态机制相关回归91项同时通过。

另外只用CPU读取Phase68保存的张量数据：两臂各51真实选中状态、16份behavior快照，
合计102状态/32快照，来源SHA核验及完整条件/目标hash通过。候选新配置相对旧配置
仅有fitting_epochs和观测开关record_fit_blocks两处变化。没有解码或读取图像。
[真实张量检查](real_tensor_check.json)记录原始来源、计数与状态，不是一次新训练。

实现来源副本见source/，相对Phase70源码快照的差异见implementation.diff。新flag默认
关闭；旧run仍应使用自己冻结的framework，不能把当前源码强行用于旧签名的恢复。

## 还没有完成的工作

完整SD3的4fit执行尚未GPU验收。上述结果不证明更多fit能提高修复或开发joint，也不
保证3小时运行上限足够。原始零LoRA初始化的GPU guard仍须由下一执行入口保留。
需要将整块之后的权重接入机制前向、规范命名post_fit_block_local，并接入全部任务/
五质量、来源绑定、真实账务及完整报告。运行脚本/输入指纹/预算reservation尚未冻结，
本轮没有sbatch。不能把旧immediate_local报告直接改名就声称已测试新执行链。

下一轮研究合同已经在新结果出现前固定：[LEARNING_PROTOCOL.md](LEARNING_PROTOCOL.md)。
它保持实际LR、严格筛选和同量随机，比较4fit与只读Phase68的1fit；以共同128旧dev的
完整部署joint及全部质量判断，不以训练修复率单独推进。研究合同冻结与执行输入冻结
是不同阶段；当前只完成前者，未宣称实验已经开始。

## 最新科学结果仍为Phase70/68

Phase70在验证组自身51训练状态：behavior/teacher/即时局部/最终局部/最终后缀的joint
为1/33/1/1/2。教师相对behavior的五质量描述区间全正，32个直接修复最终局部保留0。
支持学习兑现瓶颈，不证明选择优于随机。Phase68完整dev仍验证8/128、随机9/128，
原门槛未通过。Phase54训练修复恢复但dev落后随机的反例继续约束下一实验解释。
