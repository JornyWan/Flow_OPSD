# 从开发诊断到可审查的论文实验

更新依据：用户要求完成低学习率计划后继续下一步，并同时形成更完整、适合论文的实验。
本轮执行上限为新建的10800单GPU秒，旧预算与归档不改写。不是承诺在三小时内完成整篇论文。

## 主张、方法与当前证据

核心方法仍是**经过终端结果验证的外部专家局部修正蒸馏**。外部教师产生动作，冻结学生
提供相同后缀；以声明语义约束的修复和损伤选择监督，训练后的单学生独立生成。
冻结学生参与补全不把该方法变成OPSD式reward-gradient self-distillation。

| 主张 | 应证明什么 | 已有证据 | 本次补充 | 仍缺什么 |
|---|---|---|---|---|
| C1 局部机会 | 相同学生状态上，动态选择优于最强固定介入，保护原有语义 | Phase45冻结确认中有正信号；A/B共享评测模型 | 不重复调选择器 | 更广任务分布、独立评价器复核 |
| C2 部署兑现 | 有价值的目标能被学会，并转移到单学生完整生成 | Phase46单轮有限池有正信号；Phase47在线增量未通过 | Phase48单因素lr；Phase49教师/学到的单步/完整路线三层对照 | 共同起点原生继续训练、独立训练seed、连续在线轮次 |
| Q 能力保持 | 改善联合任务同时保留文字、结构和一般质量 | Phase47结构改善伴随OCR退化，开发集质量下降 | Phase50全部200条DrawBench、2噪声、5检查点、5指标及两噪声特征距离 | 全量标准GenEval/OCR、更多重复和保留边界 |
| C3 效率 | 同GPU-hours优于合理基线，部署无额外教师/搜索 | 有逐作业成本、NFE、训练和验证计数 | 所有新作业/失败/加载/评分单独记账 | 真正的计算匹配训练曲线；此前专家预训练和搜索成本完整清单 |
| R 稳定性 | 独立训练与更细步数/其他solver仍有增益 | 尚未建立 | 不从Flow10结果外推 | 至少3独立训练seed；Flow10/20/40、DPM2 10/20/40 |

Phase48–49均使用已暴露的32题，只能定位问题。Phase50覆盖公开标准测试分布，但不宣称
这些公开prompt从未出现在任何上游训练/调参中；也不是联合能力的确认集。

## 当前实际执行顺序

1. Phase48：只改变lr 3e-4→1e-4，三臂恢复各自相同step8、Adam、scaler、RNG；8更新、
   分母48、原mask/目标/采样保持，96新图。1200秒上限。
2. Phase49：无论48的结果如何，都测teacher/high-lr-student/low-lr-student各介入index1
   一步，冻结同一个旧学生的前缀与后缀；全32题共96图，比较三层收益。1200秒上限。
3. Phase50：预定五个检查点（裸backbone、发布OPD Student、Phase46/47/48 verified），
   200 DrawBench提示词×2噪声×5模型=2000图；全部五项质量和CLIP特征距离。7200秒上限。
   49/50可用两块GPU并行；两者总成本仍相加。模型名单在48效果分析之前冻结，不选赢家。
4. 将结果、未解决问题和下一项检验汇总为同一个论文证据页，并输出可导出的数值表/图。

标准数据审计必须保留：GenEval2212行=553prompt×4；OCR1018行=1018不同prompt；
DrawBench1000行=200prompt×5。不能按原始行数假装独立样本。Phase50有意用每题2噪声、
Flow10；这不同于原生Flow40和5重复，不能混报作者配方结果。

## 基线注册：名字与实现相符

| 对照 | 在主表中的作用 | 状态/实施边界 |
|---|---|---|
| 裸SD3.5-M、发布OPD Student | 参考能力起点 | Phase50预定评测；发布学生的预训练成本未知，不能当成本相同的自训基线 |
| 原作者DiffusionOPD继续训练 | 排除额外训练、专家信息本身的收益 | 本地固定作者commit、native入口与短smoke已可运行；正式共同任务继续训练尚未完成 |
| 当前fixed/random/verified | 选择机制对照 | 共用kernel、固定状态池；不能改名为原版OPD或DanceOPD |
| DanceOPD公开实现 | 单查询、低噪声teacher query的更强简单对照 | 本次重新获取并固定版本，需核对路由、直接velocity-MSE与时间分布；不同于当前index1/dt²机制臂 |
| DiffusionNFT作者实现 | 同终端任务效用下的在线RL对照 | 本次重新获取并固定版本；保留正常在线采样，不能喂verified筛出的候选冒充原NFT |
| 轨迹收益加权蒸馏 | 检验是否只需给好轨迹更大权重 | 待明确同一任务效用与权重规则；称机制移植，不冒称REST复现 |
| 同候选赢家整图SFT | 区分局部动作监督与搜图收益 | 共享搜索费用、按case分组，不能挑赢家后忽略被搜索的失败图 |
| OPSD / ReFL | 可微奖励的辅助比较 | 原版OPSD需reward gradient；OCR exact/离散结构主实验不直接提供该梯度。surrogate另命名并计训练成本 |

正式方法比较应从共同初始化、相同数据拆分与奖励定义出发。当前Phase47/48三臂继承各自
Phase46历史，适合臂内更新对照，不能拿来代替正式统一起点的算法主表。

## 下一阶段正式设计应如何冻结

- 建立新train/dev/confirmation/final-test清单，明确文字词表、对象组合和模板的分离；
  已用32题标记development/exposed，不回收成held-out。公开基准与自定义联合任务分表。
- 主比较预先唯一指定为verified versus计算匹配的原生OPD任务适配版本。random、fixed、
  DanceOPD/NFT是必要对照或次要比较；主指标仍整图联合成功，文字/结构和损伤全部报告。
- 至少3份独立训练随机性：采集噪声、prompt次序和random mask均独立；每份各自在线刷新。
  仅改变测试噪声不是3个训练seed。按训练run分别报告，然后对prompt和run变异分别建模。
- 用历史配对差的方差估算样本量并做保守敏感性；保存到precision_plan.json。
  几十题不足以可靠检验5pp。正式确认不能因负结果不断加样本；在看确认分数前固定上限。
- 训练GPU-hours与评测GPU-hours分开；总研究成本另列全部失败/开发搜索。相同optimizer
  steps不是同算力；baseline可以在节省的分支/验证计算中多训练，不能故意烧空计算配平。
- Flow10先成立后，再固定模型测Flow20/40与DPM2 10/20/40，报告实际NFE与CFG。DPM
  分支训练需要完整solver history；仅有部署DPM接口不证明分叉目标实现正确。
- 如Phase49显示局部动作也无收益，先处理目标可学性/训练覆盖；如果局部有效而整图退化，
  再单独测一种轨迹保持或更短更新。旧retention无稳定收益，不能默认作为已证明修复方案。

## 来源与创新边界

2026-09-24核对作者[DiffusionOPD](https://github.com/ali-vilab/DiffusionOPD)、
[DiffusionNFT](https://github.com/NVlabs/DiffusionNFT)及公开[DanceOPD](https://github.com/worldbench/DanceOPD)代码。
[DiffusionOPSD](https://github.com/worldbench/DiffusionOPSD)明确自称external implementation，保留此来源标注。
最接近的同期工作还包括[REST](https://arxiv.org/abs/2608.09226)与
[ReNFT](https://arxiv.org/abs/2609.00061)；本次仅重新确认论文/代码来源，不称完成了全文级新颖性审计。
“奖励筛选”“反事实分支”“局部credit”这些词本身不构成创新。方法论文需要实证证明
外部专家的具体修正价值、学习兑现、共同条件下的基线优势和计算收益。
