# Phase47：各臂独立在线刷新一轮

事前冻结；用户要求继续下一步。Phase46归档、门槛和正结果保持不变。本轮没有使用人工标签。

## 问题与预算决定

检验C2在线继续训练：验证组在自己刷新采集的状态上再拟合后，是否优于它自己的Phase46检查点。
不能把Phase46已存在的优势当成本轮增益。三臂分别延续自己的模型/Adam/scaler历史，而非新建共同初始化。
主要比较为新提示词上after_verified−before_verified；其它臂的前后变化与横向比较完整报告。
没有“旧固定池再训练8次”的对照，因此不能将变化唯一归因于refresh，而不是额外优化。

剩余共享GPU时间2888秒，完整96状态/臂方案及配对前后评测超出稳妥预算。
本轮明确缩小为48训练prompt各1份新噪声；评测32新prompt各1份原冻结噪声，非Phase46的每题2seed。
训练仍覆盖全部48开发prompt，四任务族各12；评测四族各8题，先覆盖本族全部仍可用模板，再按hash补足。
排除预留人工题后只剩40题，color/count各6种、multi7种、relation8种模板仍有未生成题；
不能声称每族都有8个新模板。该情况在CPU准备检查时发现，尚无本轮GPU作业或任何新结果。
选择不依赖结果，排除全部既有评测提示词和24个预留人工题。不是正式足量统计确认。
新图片预算512，与原1256以及Phase46的448分开记账。单L40最多2700秒，含所有加载/审计/失败。
成本估算用Phase46实际2537秒：本轮训练状态减半，增加在线采集/评分及前后检查点，但取消旧局部介入诊断。
失败不删案例、不自动加时、加图或重试；先记录原因与已用资源。原共享10800秒上限不改。

## 固定算法与一次在线轮次

每臂恢复Phase46各自step8的student、非空Adam、scaler、RNG；优化器超参数完全不变。
固定GenEvalTeacher/index1全幅单步、Flow/Euler10 shift3、CFG4.5、512、fp16 autocast/FP32权重latent/VAE。
每臂先以自己的step8学生冻结为behavior，用同一48训练prompt和同一份新采集noise seed生成完整noop，
再从自身index1真实状态执行教师一步并由自身冻结behavior重算后8步；3×48×2=288张采集图。
相同冻结A/B评分路径；训练选择只读取A。B仅诊断，不影响mask、优化器或目标。
fixed用自身全部48目标；verified用自身A接受目标；random从自身48状态按族匹配verified接受数量，
hash phase47-random-mask-v1固定随机mask。它使用verified的数量控制，属于机制对照，不是独立原生算法。
三臂分别在自身状态拟合8次尝试。原transition-mean loss，分母为当前全部48槽位；拒绝项不回归noop。
整批无接受时跳过Adam/scaler且记录skip；实际step可能停在8，否则到16。不换lr、EMA、保持项或reward权重。
behavior在本轮采集起点显式刷新，拟合8次期间保持step8不变；下一轮采集才能再刷新。

## 真正部署的配对评测

7个模型：原发布Student，三臂各自before(step8)与after(本轮结束)，每模型完整生成32图，共224张。
推理仅加载学生，不加载教师，不分支、不选择结果。新无专家backend先精确加载verified导出LoRA。
7策略同prompt同noise，同分辨率/采样器/步数；只用预定最终检查点，不能看分数挑中间checkpoint。
主终点仍为B整图OCR exact AND全部GenEval strict。prompt整簇、family分层bootstrap20000，seed47047。
唯一主比较after_verified−before_verified。报告所有臂相对自身before、noop，以及after_verified对两after基线。
验证增量−随机增量的差分仅次要描述性结果。不同臂继承的历史不同，不称公平从头训练或同GPU总预算。
报告全部OCR/结构分项、A未知、运行错误、逐族/模板、已真约束损伤。
对224个部署终点全部评测PickScore、CLIPScore、Aesthetic、HPSv2.1、ImageReward，质量不替代主终点。
288采集终点只评测任务证据用于训练和诊断，不冒充部署质量样本。A/B共享模型，不是独立视觉真值。
新采集noise和新random mask各一份，不是多训练seed稳定性证明；跨sampler/步数尚未验证。

## 核验与结果决定

模型/数据/代码/奖励资产绑定；每臂首次noop和教师路线完整重放，教师真实状态目标重查。
CPU契约验证非空Adam恢复、各自状态、显式行为刷新、空mask跳过、训练只读A和历史smoke字段兼容。
GPU审计：从旧step8使用一条固定查询做连续step9/10，再从磁盘step9恢复做step10，参数/Adam/scaler精确一致。
审计共3次单查询更新，随后重置为各臂原step8，不泄露进正式训练；全部成本计入。
两次评分阶段各先重放8张原smoke，所有生成图/轨迹/目标/缓存和A/B证据最终全量核验。
任何错误或绑定失败阻止主结果解释，不静默修复标签或删图。
只有主差≥5pp且95%下界>0，同时after_verified点值不低于after_random、after_fixed，才设计更长在线验证。
否则不自动继续当前在线轮次；既有Phase46正结果保持，可根据增量与损伤定位下一项单因素问题。
不按本轮结果回头调整主比较、窗口、筛选阈值或超参数。0助手直接读图，0远程视觉API。
