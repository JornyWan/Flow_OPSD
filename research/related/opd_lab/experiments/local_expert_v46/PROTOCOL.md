# Phase46：验证监督能否学进一个正常部署的学生

本协议在任何Phase46训练与新图生成之前冻结。依据Phase45通过的C1复测继续检验C2；
Phase44失败门槛、Phase45通过结果、原始归档均不改写。人工标签只作既有诊断参考，未进入本轮。

## 主张与唯一变化

主张C2：同一外部GenEval教师的局部修正，经终点验证筛选后，能否产生优于等量随机监督的完整学生。
三臂fixed_all、random_matched、verified；仅目标接受mask不同。
复用Phase44全部96个index1真实学生状态及detached教师动作，verified固定接受21项，
random按四个任务族匹配21项（hash phase46-random-v1），fixed_all用96项。
同一发布Student起点、各臂空Adam状态、相同查询顺序，8次全池更新，每次分母96；拒绝项不产生梯度。
原transition-mean loss = .5 mean((dt*(student-target))²)，不加入保持项、reward权重或EMA。
AdamW lr=3e-4, betas=(.9,.999), eps=1e-8, weight_decay=1e-4, clip=1；fp16 GradScaler。
LoRA rank32/alpha64；单训练seed46046。每臂仅第8次更新的checkpoint用于评测，无挑选checkpoint。
先从相同模型/Adam/scaler/RNG重做两次verified首步，审计一致性后重置，审计不计入24次正式更新。
同8次更新不代表同计算量：fixed反向768，random/verified各168，另审计42；计入全部耗时。
这是固定池短程机制实验，不是原生在线OPD、完整OPD/OPSD复现或效率对比。

## 分层评测

机制层：从已暴露的Phase45按family×template各hash取1题，32题×2seed=64例。
复用64个noop和64个原专家终点。三个学生只在原始index1介入，冻结初始学生重新计算后8步，共192张新图。
该层不称独立确认；动作MSE只辅助定位，终点成功才代表局部效用。
部署层：从余下未生成且不在24个预留盲评提示词中的数据，各family×template选1题，32题×2seed=64例。
初始学生与三个训练臂各完整Flow10生成，共256新图；三个臂全程只用自己的student，不加载教师、不分支、不验证。
同一噪声/提示词/CFG4.5/shift3/512；Flow/Euler10、FP32权重和latent/VAE、fp16 autocast与Phase44相同。
训练、机制、部署提示词互斥。原开发缓存的validation字段保留来源含义，另建split=train的训练清单。
全新64个部署案例的唯一主比较：verified−random_matched，主指标B整图OCR exact AND全部GenEval strict。
prompt整簇、family分层bootstrap20000，seed46046，报告配对胜负与95%区间；不是训练seed不确定度。
同时完整报告verified−fixed_all、各臂−noop、OCR/GenEval分项、逐族/模板、B已真约束损伤、A未知、运行错误。
新448图均用冻结A/B和五项质量评测：PickScore、CLIPScore、Aesthetic、HPSv2.1、ImageReward。
128参考终点仅复用带原fingerprint的精确缓存，不能冒充新评分。A/B共享模型；人工非主指标或执行前提。
不以质量分替代联合任务主结果；不预设质量非劣或全部质量改善。

## 成本、核验与停止

新的学习图片预算448，与Phase42–45已耗尽的1256张原额度单独记账；不改写原上限。
最多单L40 5400秒（90分钟），含训练、审计、生成、评分和失败；占共享10800秒预算剩余5425秒。
不自动重试、不追加步数/样本/窗口、不依据评测挑mask/超参数。不直接由助手读任何图，不转远程视觉API。
绑定代码/数据/模型/奖励资产；精确重查96个原学生动作和教师目标，首例完整路线重放，首步重复性检查。
在新的无专家backend加载导出LoRA，校验权重；初始student/behavior路线重放一致。
全量图像/分数哈希绑定、保存轨迹Euler重放、A/B证据CPU重放和8图历史smoke评分一致性。
任何运行错误、丢失案例、绑定失败均阻止解释，不删除问题案例。

## 事前决策

仅当完整生成verified−random点差≥5pp且95%簇区间下界>0，verified−noop点差>0，
且verified−fixed_all点差≥0，才设计有限在线刷新学习。质量全部单独披露，该门槛不授予质量非劣结论。
否则不扩展当前拟合方案：结合训练MSE、局部终点和完整生成区分拟合失败、跨状态泛化失败及整轨迹漂移。
即使局部正面，也不能宣布部署有效；即使完整生成改善，若随机/全部同样好，不能归因于验证选择。
64例是pilot，区间可能宽；失败不证明所有验证蒸馏都不可行，也不通过调主比较来挽救。
多seed、原生baseline、GPU-hours效率、跨sampler/步数稳健性留待后续独立协议。
