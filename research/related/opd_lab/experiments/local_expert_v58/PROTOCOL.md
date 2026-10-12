# Phase58：GenEval教师CFG1.0 vs4.5，固定局部候选诊断

用户2026-09-29要求执行下一步；本轮落实刚讨论的教师工作点核查。独立执行上限
1张L40、1800秒，不使用旧预算余额，不自动重试或进入下一轮。无训练、无optimizer。

问题C1/C2：当前局部教师CFG4.5与官方GenEval工作点1.0的差异，能否解释候选端的
任务/质量取舍？唯一改变为query1处教师CFG4.5→1.0，动作幅度仍1。原学生前缀和
冻结学生后缀CFG均4.5，Flow10 shift3、512、fp16 autocast/FP32权重latent/VAE不变。
旧CFG4.5与noop完整复用Phase55三池全部192个不同train状态，不只取34个接受目标。
不生成开发128题，不打开confirmation512或final768，不做第二噪声或CFG网格。
这不等于完整教师采样、原生OPD复现、学习效果或独立确认。

每题核验同噪声/context/prefix和两个原anchor动作，并实际查询教师CFG4.5，与旧缓存
query1逐值相等。另实际查询CFG1.0，经后端单条件分支计算；CFG4.5分支两条件。
这与官方teacher CFG实现一致，记录实际forward examples，不把CFG1与半幅混淆。
每池第一题用旧noop/CFG4.5动作重算完整后缀，核验6个旧PNG哈希。所有新后缀动作
必须在新状态重算。保存192个新教师动作、192个终点/轨迹；teacher与anchor均不更新。

新终点192，task与全部5质量值960；另固定8旧终点40质量值parity，绝对误差<1e-4；
固定8旧task parity。覆盖不全、评分错误、轨迹/参考动作不一致均停止有效性判断，
不丢样本、不改评分器。原评分缓存只读，图像不传入agent或远程视觉。

主指标：192状态B joint的cfg1−cfg45配对均差；其它固定对照cfg1−anchor、cfg45−anchor。
全部15指标逐一给出均值、family分层paired bootstrap20000次、seed58058描述性95%区间。
任务OCR/结构分项、unknown、保护损伤、全部5质量（PickScore/26、CLIP、Aesthetic、HPS、
ImageReward）、pool/family/domain/template分组全部报告。原接受34/拒绝158是固定子集；
cfg1下的接受数仅事后机制诊断，不用于重新选样或调门槛。

仅当主joint点估计>=0，五项质量差点估计均>=0，且至少一项质量区间下界>0，才值得
另立学习比较；否则若joint区间上界<0，记为cfg1任务下降；其余记取舍/不确定，均不
自动切换默认CFG或开展训练。即使通过也非正式非劣/学习保证；区间未调多重比较，
不反映训练seed方差。预期可证伪：质量恢复但joint下降仍只支持工作点影响取舍；
五质量无改善则不能把旧退化主要归咎于这个CFG差异。

每次提交前检查minilab-gpu、sablab-gpu、scu-gpu、preempt_gpu，遵守跨项目sablab<=6卡，
保持L40精确回放兼容性。普通分区优先，本轮不选择抢占分区，无中断重试。
预计约20分钟，Slurm30分钟/内部timeout1740秒。失败保留完整traceback和路径。
GPU成功结束后，CPU最多120秒有界刷新/等待完成标记和引用文件；hash不符立即失败。
全量哈希、统计、报告、归档在CPU完成。CPU分析故障只能另立修复记录，不能重跑GPU。
本轮没有新增周期agent定时器或下一轮GPU预算。
