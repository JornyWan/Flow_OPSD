# 下一项已实现诊断：真正接受的修正能否兑现

实现与18项独立CPU测试已完成；连同旧分段和Phase68报告回归共40项通过。
Phase68验证组16次真实更新的前后权重链已通过数值核对，覆盖51状态/35独立prompt。
这是CPU来源验收，不是GPU重放或学习效果结论；随机仍在训练，来源尚未完整冻结。
初次CPU查看遇到checkpoint目录尚不可见，后续目录和文件出现后成功核对；无GPU重试。

五个阶段：原动作 → 教师动作 → 刚完成真实更新后的局部动作 → 最终局部动作 → 最终后缀。
前三个对照均沿原冻结behavior续完，区分目标有效性、即时学习、后续干扰与采样扩展。
即时权重来自下一次非空更新前保存的真实behavior，最后一次用最终RAW；无重训练重建。
原PNG与A语义必须重放一致。两臂历史state不同，分别诊断，不伪装成配对选择器增益。

两臂完成后运行（从opd_lab，设置PYTHONPATH=python:scripts）：

    ../../conda_envs/envs/diffusionopd/bin/python scripts/prepare_accepted_state_mechanism.py --scope experiments/local_expert_v69

只能在核对无同名目录/作业、冻结检查与测试通过后准备提交。若夜间提交，先通过
`experiments/overnight_20260930/runner.py reserve --name local_expert_v69 --seconds 4500`
预留，再执行该新scope中的submit_accepted_state_mechanism.py。提交器每次probe四分区，
幂等intent、一次作业、无重试，保留独立4500秒上限。当前没有提交或预留此诊断。

完成后用冻结框架中的opdlab.accepted_state_mechanism report输出到新的CPU报告目录。
该前向诊断不包含训练、全程最终学生或独立测试；完整部署以Phase68开发比较为准。
