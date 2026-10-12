# 提交前选择GPU分区

2026-09-28用户要求：每次提交查看minilab-gpu、sablab-gpu、scu-gpu和抢占队列，
优先使用合适且最快可获得的GPU。当前抢占分区实际名为`preempt_gpu`。
此偏好已写入工作区与OPD Lab的AGENTS.md，对后续提交持续有效。

`sablab-gpu`同时使用GPU上限为6，统计用户全部项目的作业，不是本实验作业数量。
提交前同时检查pending GPU预留，以免它们一起启动超限；保守纳入sablab-gpu-low。
本地检查不能约束其他项目绕过此流程的同时提交，因此实际提交前必须刷新全用户计数。

新launcher应在每次实际sbatch前调用`opdlab.gpu_scheduling.probe`，保存原始调度
快照与选择；检查`selected`、剩余授权预算及一次性提交意图，再执行实际提交。
探测本身不会提交、取消或搬迁任何GPU作业。估计不是保证开机时间。

```bash
PYTHONPATH=python ../../conda_envs/envs/diffusionopd/bin/python scripts/probe_gpu_partitions.py \
  --job-script runs/EXPERIMENT/job.sh \
  --out experiments/EXPERIMENT_execution/scheduling_before_submit_001.json
```

默认保持脚本里固定的GPU型号及GPU数/节点数、CPU、内存和时限，用`sbatch --test-only`
比较各分区真实可调度性。脚本必须声明明确的`--gres=gpu:TYPE:COUNT`。
测试输出虽可能显示一个Job编号，但没有真实GPU任务提交。

若协议已验证多个型号兼容，可重复传`--compatible-gpu l40 --compatible-gpu a100`。
不能为了空卡擅自换掉需要精确动作回放的硬件。探测选择最早可信启动估计；一分钟内
视为近似同时，优先普通队列。不同型号的预计完成时间还须结合已有实测速度判断，
此脚本不虚构GPU速度换算。memory/CPU/QoS/账户/预留都可能使“空卡”仍需排队。

默认对抢占队列也做只读测试，但不选它执行。只有协议允许中断及其恢复/失败处理时
传`--allow-preempt`；不会因此开启自动重试或扩大预算。适合独立且可恢复的工作可
并行推进，已经冻结为串行的实验保持其原失败停止与比较协议。

Phase57的科学协议固定L40、三池串行、技术失败不重试。第一池已完成，第二池已
运行；不改已冻结源码、任务脚本或完成档案。剩余池的调度检查与任何pending状态下
的同型号分区调整在独立执行记录中保存，不重提、不取消运行作业。
