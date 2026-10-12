# Phase76：官方完整训练循环通过，下一步准备足量对照

2026-10-09晚（UTC2026-10-10）。Slurm1077242 COMPLETED0:0，minilab-gpu/ai-gpu12，NVIDIA L40S，实际250/1800请求GPU秒，无重试/新timer。完整运行和审计通过；166冻结输入及保存的审计证据哈希已核对。

本轮从**纯SD3.5M＋fresh rank32/alpha64 LoRA**开始，不是续训发布的OPD Student。初始LoRA B全零，功能上对应底座。使用官方固定commit的SOPD-GenEval完整主循环，保留loss、FP16/CFG、Adam、学习率、每卡batch和EMA调用。只加入可剥离且AST核对通过的观测调用、已有生命周期/本地资产绑定；单卡与联合任务数据属于任务适配，不能称原八卡论文的完整复现。

|检查|实测|
|---|---:|
|真实Adam更新|8/8，无AMP跳步|
|采样/训练batch|3/3|
|每更新轨迹/监督状态|9/81|
|累计轨迹/监督状态/反向次数|72/648/216|
|实际见过的训练prompt|59/512|
|每批实际LR|3e-4|
|GPU显存峰值 allocated/reserved|36.833/37.787 GiB|
|每轮平均时间（含观测I/O，末轮含checkpoint）|23.648秒|
|RAW/EMA改变的张量数|381/381|
|EMA检查点与原生调用CPU重放最大差|0|

已实际复核原生EMA在global_step7连续调用三次：（epoch6,batch2）、（epoch7,batch0）、（epoch7,batch1），不是自行改成每optimizer step一次。RAW、EMA、Adam第8步、scaler及RNG保存完整。保留实际prompt ID、采样前CPU/CUDA RNG和初始latent噪声；官方same_latent=false使用共享随机流，没有编造每图整数seed。首次原生batch3在L40S上可运行，不需因显存问题缩成batch1或新增梯度checkpointing。

## 它回答了什么

完整官方循环能在现有设备上按这份配方运行；先前“正式batch/EMA/真实更新尚未验收”的工程缺口缩小了。8更新、59个唯一prompt远不够判断任务收敛、泛化或超越OPD。原生采样有72次训练图解码，agent读图0；本轮joint/OCR/结构/CER/unknown和五质量均未评，不把权重变化当质量提升。确认512/final768封存。

Phase75的CFG精度消融结果不改：修正只部分缩小梯度/更新偏差，未满足原等价门槛，也不是方法效果判负。本轮直接用官方入口，因此无须先宣称本地重写与官方等价，便可继续建立有明确来源的baseline。

## 下一轮具体配置

已经写出[256更新配置草案](native256_config_draft.json)和[完整学习/评测草案](LEARNING_DRAFT.md)：相同512 TRAIN、纯底座fresh LoRA、固定最终256更新，主报EMA，RAW作诊断；预期2304轨迹、20736监督状态。按本轮均值外推训练约6054秒，即101分钟，**不包括启动及完整评测，不是已分配预算，也不是收敛保证**。

正式评测需要按同一Flow40/CFG4.5协议重新生成S0、最终EMA、最终RAW，各128探索dev，共384图，全部任务与五质量；不能复用旧Flow10或发布Student当S0。下一提交前仍需接好该评分管线和权重来源验收，固定共同推理精度与包含生成/评分的总时限。当前256训练尚未提交，没有开放确认/最终集。

随后才测试两步闭环目标与匹配的OPD/计算对照。闭环损失已有9项CPU检查，但真实SD3集成尚未完成。优先完成这组能区分方法贡献的学习比较，不因工程关卡通过预设正向结果，也不因单个理论/数值反例否定全部方向。

20项原生/观测CPU检查与6项独立报告检查通过。完成报告首次读取遇到NFS结果文件尚未对登录节点可见，随后待文件可见只重新执行CPU报告；未重启训练、未改完成档案。

[Slurm与数值审计](audit_1077242/summary.json) · [原始验收副本](audit_1077242/acceptance_copy.json) · [冻结协议](../local_expert_v76/PROTOCOL.md)
