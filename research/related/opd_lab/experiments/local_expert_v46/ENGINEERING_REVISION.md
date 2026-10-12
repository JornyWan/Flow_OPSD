# Phase46启动修订（训练/生成/评测前）

首轮1062322耗时123秒，因CPU缓存参考与CUDA张量直接torch.equal比较而失败；0更新、0图、0新评分。
原PROTOCOL.md“无自动重试”保持原文。本文件明确另立一次有限的工程修复提交，非静默重跑：
两处replay_without_decode前将保存的trace通过已有to_device_trace转到backend.device。
没有改损失、优化器、训练样本、目标mask、候选/阈值、更新次数、评测/判定或随机seed。
运行目录换为le46_learning_002；原目录、日志、代码快照、attestation与旧analysis_frozen保留。
分析程序仅改运行目录；此处新analysis_frozen在重启之前再次记录。
第二次最多5277秒，加失败123秒仍≤原Phase46的5400秒；共享剩余时间与失败均记账。
该修订并非由任何效果分数触发。若再次失败，不隐式重试，先保留失败证据并分析。
