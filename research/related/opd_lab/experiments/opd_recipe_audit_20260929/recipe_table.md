# 固定上游与Phase60实际配方对照

只读源码/完成记录审计；非新增模型结果。来源文件快照及SHA256见evidence.json。

| 项目 | 官方MOPD | 官方SOPD-GenEval | Phase60实际 | 含义 |
|---|---|---|---|---|
| 学生起点 | SD3.5-M + 新建LoRA；默认train.lora_path=None | 同MOPD，可显式覆盖LoRA路径 | 发布OPD Student + 原池空Adam/scaler/RNG | 从强学生继续训练；不能与官方从头设置直接归因。适配基线须显式绑定同发布权重及空优化器。 |
| 教师与CFG | PickScore/Aes4.5、OCR4.5、GenEval1.0，三个slot轮换 | 仅GenEval1.0 | 仅GenEval1.0；旧Phase55为4.5 | 单教师应先与SOPD-GenEval适配比较，不能把缺少多教师都归为偏离所有OPD。 |
| prompt与训练状态 | 各slot官方数据；每轮当前学生新轨迹 | 官方GenEval数据；每轮当前学生新轨迹 | 自定义联合约束：64训练prompt固定噪声/状态，重用16次；128dev | 当前为固定池机制诊断。在线行为刷新是尚未完成的强对照差异，不能把它的效果预先归因。 |
| 教师查询/监督位置 | 每轨迹查询全部10步；监督前9步j=0..8 | 同MOPD | 固定j=1教师目标；学习和部署均只在j=1用learner | 不是原生全轨迹OPD；时间索引是零起始，j=1即第2次查询。 |
| 基础损失 | noise_level=0时，0.5 mean((学生下一状态均值−教师均值)^2) | 同MOPD | 0.5 mean((delta_sigma*(v学生−stopgrad(v教师)))^2) | 确定性Euler下代数等价；单项loss相同不代表状态分布、归一化或训练算法相同。FP32运算次序不保证逐比特等价。 |
| 监督目标选择 | 不按终点收益筛选teacher动作；proxy用于监控 | 同MOPD | 规则A接受10个，或同family×domain随机10个；拒绝54个跳过 | 同池验证/随机是匹配的选择机制对照；没有学到可靠的拒绝保持策略，不能称完整选择目标。 |
| 梯度归一化与有效batch | 每微批3图均值；27次累积=3批×9监督步；每卡81状态/更新 | 同MOPD | 10个监督状态分别除以全部64槽位，累积一次更新；固定16更新/臂 | 10/64不是10个状态均值；cfg.accumulation=12不控制这个专用路径。不能从字段表推断有效batch。 |
| 优化器 | AdamW lr3e-4，betas0.9/0.999，eps1e-8，wd1e-4，clip1 | 同MOPD | AdamW lr1e-4，其他列出超参相同；两臂相同原空状态 | LR相差3倍；不能仅比较相同optimizer steps称公平算力或同配方。 |
| EMA与行为模型 | EMA decay上限0.9/间隔8；用于eval/save，采集恢复raw当前学生 | 同MOPD | 专用fit_arms路径无EMA更新或EMA部署；anchor/behavior固定 | config里的ema_decay/ema_interval不代表此路径用了EMA，不能归为EMA学生比较。 |
| 文本编码长度 | max_sequence_length=128 | 同MOPD | max_sequence_length=256 | 即使文本未截断，也不是完全相同的编码输入设置；作用大小尚未测量。 |
| 训练采样 | Flow，noise_level0，10步，512，学生CFG4.5；fresh RNG | 同MOPD | 确定性Euler/Flow10，512，学生CFG4.5，模型scheduler shift3；每图CPU FP32固定seed | 基本solver族/分辨率一致；不同RNG来源和状态历史不保证相同轨迹。 |
| 部署与默认评测 | 一个student adapter全轨迹，默认EMA评测40步 | 同MOPD | raw learner在j=1，其余9步发布anchor；评测10步，无部署teacher | 两份adapter时间路由，尚未达到单adapter正常生成；公平评测要固定共同采样协议并披露各自部署开销。 |
| LoRA结构 | rank32/alpha64，8个attention投影模块 | 同MOPD | 同发布Student rank32/alpha64，dropout0 | 此项已匹配；不能把效果差异笼统归为LoRA秩。 |
| 质量保障 | MOPD含质量专家，但loss不保证任意五项终点质量单调提升 | GenEval单教师同样无五质量保证 | 任务规则A筛选，五质量完整报告，但不作为当前选择损失/门槛 | 不能从“OPD”名称推出所有分数单调改善，也不能把质量下降已唯一归因于缺少Aes教师。 |

## 源码位置

- 学生起点：[native_base](sources/native_base.py) 第123行；[native_train](sources/native_train.py) 第675行；[local_guidance](sources/local_guidance.py) 第61行
- 教师与CFG：[native_config](sources/native_config.py) 第80,217行；[native_train](sources/native_train.py) 第875行；[local_guidance](sources/local_guidance.py) 第75行
- prompt与训练状态：[native_train](sources/native_train.py) 第881行；[native_train](sources/native_train.py) 第914行；[local_fit](sources/local_fit.py) 第91行
- 教师查询/监督位置：[native_train](sources/native_train.py) 第513行；[native_train](sources/native_train.py) 第942行；[native_train](sources/native_train.py) 第1153行；[local_deploy](sources/local_deploy.py) 第122行
- 基础损失：[native_train](sources/native_train.py) 第1177行；[native_train](sources/native_train.py) 第1169行；[local_loss](sources/local_loss.py) 第107行
- 监督目标选择：[native_train](sources/native_train.py) 第1191行；[local_mask](sources/local_mask.py) 第81行；[local_mask](sources/local_mask.py) 第82行
- 梯度归一化与有效batch：[native_train](sources/native_train.py) 第526行；[native_train](sources/native_train.py) 第1190行；[local_mask](sources/local_mask.py) 第84行
- 优化器：[native_base](sources/native_base.py) 第90行；[native_train](sources/native_train.py) 第711行；[local_guidance](sources/local_guidance.py) 第58行
- EMA与行为模型：[native_train](sources/native_train.py) 第454,498行；[native_train](sources/native_train.py) 第483,501行；[native_train](sources/native_train.py) 第1216行；[local_fit](sources/local_fit.py) 第69行
- 文本编码长度：[native_train](sources/native_train.py) 第317,336,782,887行；[local_backend](sources/local_backend.py) 第165行
- 训练采样：[native_config](sources/native_config.py) 第66,149行；[native_train](sources/native_train.py) 第894行；[local_backend](sources/local_backend.py) 第40,185行
- 部署与默认评测：[native_config](sources/native_config.py) 第40,127行；[native_train](sources/native_train.py) 第454,498行；[local_deploy](sources/local_deploy.py) 第122行
- LoRA结构：[native_train](sources/native_train.py) 第670行；[native_train](sources/native_train.py) 第671行；[local_backend](sources/local_backend.py) 第121行
- 质量保障：[native_config](sources/native_config.py) 第83,171行；[native_train](sources/native_train.py) 第1191行；[local_mask](sources/local_mask.py) 第77行
