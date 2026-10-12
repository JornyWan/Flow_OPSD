# Phase67执行与后续判断

已提交1064833，preempt_gpu/minilab/low，1张L40S（运行时须确认）、75分钟，自动Requeue=0，单次尝试。冻结协议和输入保存在本目录。最初普通分区探测未提交任何作业；随后在首次GPU提交前明确允许CANCEL抢占，预算不变。38项CPU检查通过。

本轮只解释共同初始64训练状态：base/teacher + all/verified各三种轨迹，512终点/2560质量值。重建先核对64个dt/距离及128原始PNG；若失败，不降低标准或重试。全生成仅default adapter，j=0的后缀/完整路径应16/16图哈希一致。A旧子句重评分必须128/128一致。

检查真实sacct、submission/intent、progress、qualification、execution及失败记录，先确认原作业而非重复提交。运行中只监控和独立CPU工作。任何GPU技术失败/抢占停止，部分记录不能冒充完整报告。完成档案不可改，CPU问题另建修复目录。

正常完成且Slurm COMPLETED 0:0后，在opd_lab下运行独立CPU报告：

```bash
PYTHONPATH=experiments/local_expert_v67/framework/python:experiments/local_expert_v67 ../../conda_envs/envs/diffusionopd/bin/python experiments/local_expert_v67/report_initial_state_mechanism.py --scope experiments/local_expert_v67 --output experiments/local_expert_v67_cpu_report
```

同时完成Phase66 random1064749；只有四臂齐且账务成功才能运行原report_clean_base_study.py，输出local_expert_v66_cpu_report。不能用本诊断替代verified−random主比较或改冻结门槛。

解释顺序：先检查直接教师是否在这批状态确有收益，再看局部动作是否保留；随后看后缀替换的损伤；最后比较完整生成的前缀/状态分布作用。检查全部质量和正负转移，允许各段交互，不强制单因归因。局部目标误差是共同初始状态的误差，非历史全部接受目标的训练loss。

第一批A base/teacher joint8/7，仅1目标被接受；不能预设有平均教师增益。后来13个A联合修复因缺中间checkpoint未被复现。历史默认反向不保证逐位重放，因此未执行完整重训原型。

下一训练协议必须依据这两轮结果单独写出唯一变量和匹配对照；不自动调LR/扩大gate/增加保持项或规模。确认512/最终768仍封存，人工标签辅助，零agent视觉读取，无新定时器。

## 19:24 EDT启动验收

实际NVIDIA L40S已通过64/64状态和128/128原PNG一致；输入/原冻结后端源码核对通过。目前正在生成新分段对照，无任务收益结论。随机主对照已完成12/16轮、9次更新，逐批分层计数仍匹配。最新状态见latest_status.json。
