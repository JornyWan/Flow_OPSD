# Phase57调度补充：科学协议不变

2026-09-28用户要求每次GPU提交比较四个分区，并限制sablab并发GPU≤6。
原Phase57源码、脚本、科学协议、已完成pool1归档均保持原样。pool2正在执行，不搬迁。
本记录只补充尚未提交pool3的资源观察及必要的pending状态同型号调度调整。

已做实际只读探测：minilab的L40可立即开始；scu虽有空闲L40，但当前预计约6小时后；
preempt也可立即开始，但本实验技术失败不重试，不为同等开机时间承担抢占。
sablab无L40，不能把本轮擅自切到其他型号；用户全项目sablab当前2 GPU。
原始输出见`phase57_pool3_initial_probe.json`；这是时间点观察，不是未来保证。

独立观察器会在pool2写出GPU完成标志后、原管理器对pool2做CPU审计期间，刷新pool3
的只读探测。原管理器仍是唯一的GPU提交者，观察器绝不sbatch真实作业、不启动新训练。
当pool3提交可见时再核实：若已RUNNING则记录并结束；若仍PENDING，重新探测，
只有同L40、同minilab账户、普通minilab/scu分区中更早的选择才调整同一job的Partition/QoS。
紧贴更新前再次查询PENDING；若Slurm拒绝或已运行则不重试。保存更新命令、返回值及
实际状态；不取消重提、不修改GPU数、CPU、内存、时限、预算或训练内容。

观察最长24小时；不唤醒agent，不发送新对话消息，不新增GPU预算，不改现有完成提醒。
今后的新管理器应在每次提交前直接使用`opdlab.gpu_scheduling.probe`，避免这种对既有
冻结管理器的外置观察。GPU异构速度收益须另按已验证兼容性和实测耗时判断。
