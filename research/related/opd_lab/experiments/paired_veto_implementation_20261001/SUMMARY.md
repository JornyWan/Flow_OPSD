# 成对时间日程与明确损伤veto：CPU实现验收

完成时间：2026-10-01T05:41:23.244965+00:00。没有新SD3训练/推理/任务评测、GPU提交或agent读图。

## 实现

- 新日程balanced_family_template_paired_time_v3：保留原prompt顺序、逐查询noise与64训练
  池；只改变时间日程。同族/模板的一对独立prompt共享时间索引，每9轮覆盖0..8。
- 新semantic_veto_v1：只屏蔽valid A的true→false，收益不抵消损伤，unknown让验证器
  弃权并保留原教师监督；从不标certified_safe。技术error使整个批次停止。
- random按批次×任务×模板×时间匹配veto保留数；仅使用来源计数，自己的valid标签
  不影响名单。不强制挑反面目标。共同目标仍使用原transition loss、/64归一化。
- 接入shared engine/runtime、manifest算法标签、独立版本的count reference和日志。
  支持从完成veto的冻结原始输入准备fresh random，未复制已训checkpoint；记录真实
  常量LR、更新数和梯度norm。旧semantic_batch_v1的科学规则和已存4fit合同不变。
- 当前veto版本刻意限制constant LR、1fit、paired manifest、单进程/单累计批；不支持
  沿用旧postfit或真实选中状态诊断，避免模式和权重链被误标。新GPU报告仍待接入。

## 验证

最终125项CPU测试通过（43.12s），包括81种两子句三值组合、unknown/error、时间/模板
计数匹配、随机不看自身标签、无损伤时与all的参数/Adam/EMA完全相同、初始输入克隆、
断点恢复和旧选择/稀疏/多fit审计回归。前次3个失败分别是新测试日志键写错以及旧
脚本未加入PYTHONPATH，已独立记录并纠正；没有GPU失败或算法结果被掩盖。

真实64训练prompt日程检查：1024查询、16轮、32分层、512个大小恰好为2的cell；每对
不同prompt和noise、相同时间。每个分层16轮内覆盖全部9时间，每索引2或4条查询。
旧v2日程完全重算一致；新旧除query_index外所有查询字段一致。未把旧A标签搬到新状态。
全局时间查询数为0:114,1:112,2:112,3:116,4:120,5:120,6:114,7:110,8:106。

这是提高对照可交换性的实现，不证明新mask一定不同或学生一定获益。两项都损伤的
pair仍没有替代空间；实际mask差需新运行报告。平衡时间的相关性不同于独立均匀抽样，
不宣称新轨迹目标严格无偏。不同运行的在线状态也不当作同状态配对。

## 下一步与边界

[研究合同](RESEARCH_PROTOCOL.md)已前瞻固定all/veto/random三臂问题和停止判据；
[NEXT](NEXT.md)列出独立控制器、RAW/LR/A重放、完整任务和五质量报告的执行待办。
最多三作业各10800 L40S秒，来自既有32400上限，当前预留0/提交0；不是新增预算。
初始仍纯SD3.5-M/fresh LoRA，64训练/128已暴露开发仅有限机制试验。
原生强基线、足量训练、多seed、标准基准/采样鲁棒性和封存集仍未完成。

主动源码被本次前瞻修改后，之前foundations审计的两个源码SHA可在原冻结快照找到，
映射见foundations_source_reconciliation.json；完成档案没改。新源码9文件完整复制并
记录SHA，供后续准备执行时核对。
