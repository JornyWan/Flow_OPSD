# Phase60：CFG1学习的第2个独立训练池复核

2026-09-29 05:00续看后前瞻冻结。Phase59数值在独立CPU修复后完整：同128dev的
CFG1 verified57、random53；旧CFG4.5 verified57、random63。主差+3.125pp、区间跨0；
新验证相对旧验证五质量点估计均改善，达到父轮“值得独立池复核”门槛，但不证明
稳定选择优势或质量非劣。原Slurm FAILED事实保留，CPU修复无新增预测或GPU。

本轮按既有编号选择pool2，不按数据表现选池。唯一预定变化为训练池：Phase55原
pool2的64题和训练种子55056，和pool1的64训练题不重叠。相同发布Student起点、
GenEval教师权重、CFG1、原A ordered_decision、family×domain匹配和该池原mask_key，
原规则接受10个，随机也10个；旧CFG4.5该池接受11个。候选来自Phase58固定缓存，
不新增候选生成或重新评分选择，不用B、五质量或人工标签调筛选。

保留Flow10/shift3/512/fp16 autocast/FP32权重latent，学生CFG4.5；query1 learner，
其它步anchor。固定16次更新、学习率1e-4、原transition mean loss和分母64。
恢复原pool2 fresh_initial_training的参数、空Adam、scaler和全部RNG；两个臂共同起点。
每臂160反向query，旧每臂176；另重复首步两次审计20个反向query计入成本。
不是跨CFG严格同算力，不是单adapter完整部署，不是原生OPD或在线刷新训练分布。

每臂完整64训练+128原dev，共384新终点、384 task记录、1920新质量值；另固定8旧
终点40参考质量值parity。教师动作64个GPU实际查询逐值核验，128训练动作指标、
384prefix、全路由重放、导出重载、教师/anchor/behavior不变与Adam/RNG恢复检查沿用。
只使用最终16步，不能选中间checkpoint；不扩大确认或最终集，不换verifier。

主指标：本pool2开发verified−random_matched的B joint。新verified−旧verified以及
新random−旧random、各臂−发布anchor都报告。全部15指标均值、训练8/开发8对的
240个描述性区间，family分层paired bootstrap20000，统计seed60060/60061。
训练seed是55056，与统计seed明确区分。全部五质量（PickScore/26、CLIP、Aesthetic、
HPS、ImageReward）、OCR CER/exact、结构、Aunknown、保护损伤、修复/损失、
family/domain/template分组及接受/拒绝训练修复与动作漂移全部报告。

复核规则与Phase59相同：本池verified−random点估计>0，verified−旧verified的joint
点估计>=0，且五质量差点估计均>=0，才记本池复核信号。失败即不扩大此配置；不得
删掉未改善的指标、改步数/学习率/选择阈值。即使通过也只有两个训练池和重复使用的
128dev，不算256独立prompt，不给可靠的跨训练seed置信区间，不宣称三池复制或确认。
后续CPU汇报须并列pool1/pool2，允许给等权点均值，但不能用其掩盖任何一池的负结果。

明确资源：本夜最后一个新GPU作业，独立4500 L40秒（75min），内部timeout4440；
总预留1800+4500+4500=10800，不复用旧未耗余额。提交前检查四分区和跨项目sablab<=6。
普通L40分区、不选择抢占、--no-requeue，无GPU失败重试。Slurm --deadline=当地09:00，
最多等待队列600秒；到时只取消本轮仍PENDING的作业，scancel状态过滤防止取消已开始
运行的任务，不重提。两次既往test-only均预估4小时后、实际立即开跑；若本次预估迟于
截止，但同配置测试仍eligible且存在足够空闲L40/CPU/RAM，允许一次有上述硬截止和短
等待上限的提交，并记录偏差；无适配空闲或测试不eligible则不提交，转CPU工作。

Phase59暴露的工程问题已在共享代码修复：参考评分覆盖按实际endpoint×5核验，包含
重复/缺失/未知key/非有限值检查；调度脚本解析支持已知no-requeue/requeue布尔标志。
只改变完整性检查，不改变生成、拟合、评分、指标或门槛。相关CPU回归必须通过，
旧归档/运行快照不改。新GPU完成后在CPU做哈希/分析/报告归档；CPU错误独立修复。
本轮完成后不再提交任何GPU，即使得到正向结果；剩余夜间时间做结果分析和论文整理。
