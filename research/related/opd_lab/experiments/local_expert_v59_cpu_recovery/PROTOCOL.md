# Phase59 独立CPU完整性恢复

Slurm1064458保持FAILED（2595/4500 L40秒）。训练、384生成、任务及全部五质量已完成；收尾函数误用960条常量，实际应为1920。独立CPU只校验并分离40条参考评分，未重跑GPU、生成、评分或训练，未改变数值、门槛或checkpoint。

分析视图result.status=complete仅指证据完整，不指Slurm成功；gpu_job_state=FAILED与completion_mode必须保留。原运行、原quality.jsonl的1960行以及失败日志不改。

# Phase59：CFG1下的验证与同量随机局部学习

2026-09-29夜间新授权下的独立协议；Phase58候选门槛已通过，但它不证明训练泛化。
固定从原编号pool1开始，不按三池效果挑最好池。一次L40作业、4500秒上限，最多
384新终点、1920质量值；另固定8旧终点40质量值评分parity，不新增候选图。
使用同一64训练题、同噪声、发布Student起点、相同GenEval权重、固定Flow10/shift3/
512、学生CFG4.5、query1 learner/其余anchor。唯一上游实验变量是教师CFG4.5→1。
真实CFG1候选来自Phase58已审计缓存，每题再次在GPU实际查询并与动作逐值相等。
不是原生OPD，不是单adapter部署，不是独立确认，不是新的在线状态分布。

使用原A ordered_decision阈值对CFG1候选重新判定；不使用B/五质量/人工标签选择目标。
得到verified13个；random_matched使用原mask_key与原family×domain分层，取同量13个。
不是直接沿用CFG4.5旧verified12名单。原接受数量变化是同选择规则作用于新候选的结果；
跨CFG比较同时包含目标与mask变化，不能写成纯动作幅度效应或严格同计算量。
原CFG4.5两学习臂、anchor和候选均只读复用，不改缓存key/指纹或旧归档。

复用Phase55同一fit_arms/fit_step与部署generate，loss分母仍64，无保持正则或新loss。
学习率1e-4、相同AdamW/scaler，恢复原Phase55 fresh_initial_training的参数、空Adam、
scaler和CPU/CUDA/Python/NumPy RNG后再开始；两个臂独立从这个共同状态拟合。
固定16次更新，仅使用最终16步，不选中间checkpoint。各臂208反向query（13×16），
旧各臂192；重复首步两次审计26额外反向计入成本。教师/anchor/behavior不更新。
每个臂完整64训练+128既有dev，复用发布anchor前缀，后缀必须在实际新状态重新采样。
独立导出重载、128训练动作指标回放、384prefix、每臂每队列首条完整轨迹回放均检查。

主指标：128dev的新verified−新random_matched的B joint配对差。
关键工作点对照：新verified−旧verified，另新random−旧random；两个新臂与anchor。
旧两臂彼此及与anchor完整报告。全15项指标（含5质量和PickScore raw/26两个单位）
逐项均值和family分层paired bootstrap20000次，训练seed59059、dev统计seed59060。
训练8对/开发8对，共240区间；family/domain/template、unknown、保护损伤、修复/损失
和条件分母全报。64训练的接受修复兑现/拒绝漂移是解释，不代替dev主指标。
全部CI描述性、未校正多重比较，只涉及128独立dev prompt，不反映多训练run方差。

下一池推进门槛预声明：主verified−random点估计>0、新verified相对旧verified的joint
点估计>=0，且全部5质量差点估计>=0。全部满足仅记“值得另立独立训练池复核”，不是
选择贡献成立、质量非劣或确认性结果。否则停止扩大本CFG学习配置，不能通过改阈值、
步数或质量子集使它通过。无论门槛如何均报告相对原学生的五质量，不能忽略剩余下降。
不打开confirmation512/final768；不新增人类/agent视觉评判；不换verifier。

每次提交前检查minilab-gpu/sablab-gpu/scu-gpu/preempt_gpu，sablab全项目<=6卡。
维持L40精确回放兼容，普通队列，--no-requeue，无GPU失败重试。Slurm75分钟，内部
timeout4440秒。GPU只落完成标记；CPU最多120秒等待文件可见、完整hash/分析/报告归档。
CPU分析错误在独立修复目录处理；原失败记录保留。夜间总预算按请求时限计，不回收
Phase58剩余预算；本轮提交前通过夜间runner reserve独立预留4500秒。管理器不自动
提交下一轮或创建定时器，已有每小时同对话续看依独立夜间计划执行。
