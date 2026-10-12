# 本轮完成后的固定处理

1. 首先查1065020和1065024真实sacct、failure/execution、manifest；失败停止，不自动重试。
2. 两臂完整通过后，在新的CPU报告目录运行冻结report_sparse_update_study.py，原完成档案不写入。
3. 主看新verified−新random joint；同时看每个新臂−各自旧臂及新两臂−base/native/all。
   全部OCR、结构、五质量、任务族、接受覆盖、时间分布、实际参数位移和GPU秒一并报告。
4. 减轻旧退化但没有verified相对random/native的增量：只记稳定性改善，停止扩大稀疏配方。
   未减轻退化：反驳此步幅规则足以解决问题，不追加lr/epoch扫描。
   joint与质量必要门槛均通过：也只支持另立真实接受目标兑现和独立复现协议。
5. 当前保存真实选择状态和当时behavior，随后可直接比较教师一步与拟合动作的终点；
   未运行之前不能宣称已测兑现。多专家/广覆盖OPD改造、标准benchmark和多seed仍是独立缺口。

按需状态命令（LAB目录）：
`PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/status_sparse_update_study.py --scope experiments/local_expert_v68 --output experiments/local_expert_v68/watch.json`

完成后的CPU报告命令（仅完整且Slurm成功后）：
`PYTHONPATH=python:scripts OMP_NUM_THREADS=1 ../../conda_envs/envs/diffusionopd/bin/python experiments/local_expert_v68/report_sparse_update_study.py --scope experiments/local_expert_v68 --output experiments/local_expert_v68_cpu_report`

单seed与反复使用的128开发题只提供探索性证据；confirmation512/final768未启封。
