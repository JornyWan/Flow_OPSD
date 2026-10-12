# Phase66：同一纯SD3.5-M起点的四臂开发比较

本轮回答：不继承已训练OPD Student时，局部验证训练相对原生OPD适配和
同量随机是否仍有联合任务收益？不是新机制、不是充分训练或论文表格复现。

## 起点与数据
仅读取SD3.5-M transformer架构，CPU生成rank32/alpha64的新LoRA：
A为Gaussian std1/r，B全零；四臂读取同一份字节。不得读取发布Student权重作为起点。
训练前每臂实际GPU检查新LoRA与禁用adapter的SD3预测逐元素相等。
初始LoRA随机数流按排序键生成，分布匹配PEFT，非声称复刻原生RNG字节。
训练64为既有512中按结局无关哈希预选的32层×2条子集；保留原选择而不挑例。
开发为既有全部128个prompt，每条固定新seed；不是新独立prompt或confirmation。
confirmation512/final768封存。仅自动评测，agent/远程/子agent读图0，人工标签不参与。

## 四个训练臂
native：固定官方OPD commit的SOPD-GenEval原生trainer；同64条文本训练池，
每epoch64条轨迹、每条9个训练时间点、batch1、accum64×9，16epochs/16updates。
上游自己的shuffle/sampler/每轮数据抽样保留；与局部臂不称逐轨迹配对。
初始原生内部eval保留，仅对第一条TRAIN prompt做一次无奖励输出，不用作结果。
all：本地单查询，每个教师修正都拟合；是机制对照，不叫原版OPD或DanceOPD复现。
verified：固定semantic_batch_v1规则；同状态分叉noop/教师一步，冻结behavior续跑，
仅明确修复至少一项且不损伤已满足约束、没有unknown的候选被接受；目标detach。
random：来自已完成verified的每batch/族/模板接受数量，按hash随机选等量教师目标；
不看自己的接受标签；从同一新LoRA重新训练，不加载verified训练后的权重。
三局部臂均16epochs，每epoch64条查询，时间index哈希均匀0..8；训练前刷新behavior。
同prompt/noise/time日程；第一轮状态相同，更新后各自状态不同。拒绝loss=0，
固定/64归一化；空接受batch跳过更新。verified/random有效更新数必须相等。
all也计算候选验证供机制核对，其成本必须报告，不能冒称最优单查询效率。

共同SD3.5-M/GenEvalTeacher，教师CFG1、学生CFG4.5，Flow10、512px、text128、
fp16、AdamW lr3e-4/betas(.9,.999)/eps1e-8/decay1e-4，clip1，fresh optimizer。
同训练prompt池和1024轨迹机会，不同监督时间点/在线采样实现/累计反向数量。
原生和局部不是同算力比较；分别报告全流程GPU秒、前向/反向/验证成本。
原生配方只有单GenEval教师，数据/硬件/batch/原始起点显式适配，不是MOPD三教师
论文的严格复现。原生TF32显式关闭，EMA保留但不用于挑选结果。

## 固定评价与可证伪决定
仅最终epoch16 RAW；不得挑中间checkpoint、EMA或seed。单学生完整10步、
物理上仅一个default adapter，保存路由/状态hash/前向数/生成检查后再评分。
原始base+4个训练臂×128=640张开发输出；全部任务和五质量，3200质量值。
主描述比较为verified−native的B joint；机制比较verified−random及verified−all，
同样报告全体对base，OCR exact/CER、结构全部正确、A unknown、四任务族。
五质量为PickScore/26、CLIP、Aesthetic、HPSv2.1、ImageReward；记录token截断。
按prompt在族内配对bootstrap20000次/seed66066给描述性95%CI；不当成训练seed
不确定性，不作多重比较或开发集的确认性显著声明。训练修复兑现本轮未测，
保存候选与训练A证据，但不得把候选修复或MSE下降说成最终学习成功。

技术通过不等于科学通过。只有verified对native和random的joint均正、
两个CI下界均>0、且五质量对base及native没有明确负向CI时，才值得下一轮
扩大512训练和独立训练seed；这仍不证明等效质量。否则不继续叠加机制：
先检查训练覆盖、有效接受数、loss/漂移与原生学习曲线，决定增大数据/训练
预算还是放弃筛选；未知、负值和失败完整报告，不重定义门槛。
若原生或all未能改善base，先标明训练充分性不足，不能据此宣布OPD无效。

## 硬预算与提交
独立新上限4个作业，每个1×L40、5400秒（90分钟），总请求21600 GPU秒=6小时。
包含模型加载、训练、生成、所有评分。自动重试0、新定时器0，不回收失败预算。
native/all/verified可并行；random使用Slurm afterok依赖verified，失败则取消依赖。
每次真实提交前检查minilab-gpu/sablab-gpu/scu-gpu/preempt_gpu，sablab跨项目
running+pending计入最多6GPU；仅L40，本轮不接受preempt和自动requeue。
提交intent用独占文件，重复调用不会重复投递；回调/重试不自动获得新预算。
源码/数据/配置/适配器与全部大模型权重内容SHA固定，scorer另固定资源内容；
已知HF下载锁/metadata不属于模型。技术失败不重试，CPU报告修复独立记录。
