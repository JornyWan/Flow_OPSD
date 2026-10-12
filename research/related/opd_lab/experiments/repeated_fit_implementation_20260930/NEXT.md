# 下一项实际工作

1. 将保存的fit_blocks/*.pt中的after权重接入post_fit_block_local端点，保留同一历史
   prefix及冻结behavior后缀；最终局部/最终后缀和全dev仍分清，不用第1次更新标签。
2. 为模式/权重/评分元数据完整性与CPU报告增加集成测试；保留原评分器、cohort检查、
   原候选PNG重放、128dev/全部五质量和更新匹配审计。
3. 依据LEARNING_PROTOCOL.md准备新verified4/random4执行输入，核对首轮候选一致性，
   固定预算reservation，再在每次sbatch前probe四分区并记录决定。
4. 两臂均成功才生成完整比较；无新timer，技术失败停止，不重试、不回收失败预算。

当前训练提交0，研究合同已固定；独立预算预留与执行输入尚未冻结。
