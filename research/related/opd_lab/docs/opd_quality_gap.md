# 为什么局部续训质量低于初始OPD Student

核对日期：2026-09-29。只读源码和既有数值；无新GPU、生成或图片读取。

## 先明确比较起点

Phase55–57配置中的initial_adapter指向`models/DiffusionOPD/Student/lora`，即发布Student。它不是原始SD3.5。
Phase50用200个DrawBench prompt×2个配对噪声，在统一Flow10/CFG4.5/512下测得：

| 指标 | 原始SD3.5 | 发布OPD Student |
|---|---:|---:|
| PickScore /26 | 0.848978 | 0.918903 |
| CLIP | 0.285254 | 0.299403 |
| Aesthetic | 5.397655 | 6.157061 |
| HPSv2 | 0.263059 | 0.333968 |
| ImageReward | 0.628669 | 1.470071 |

五项配对描述性95%区间均为正；这支持发布OPD Student在该评测协议下改善原始模型，不等于本地已从头完成官方训练复现。
这些DrawBench绝对值不能与Phase55自定义联合开发集绝对值直接拼表。
来源：[Phase50数值](../experiments/local_expert_v50/results.json)。

## 当前局部实验不等于原生MOPD配方

官方MOPD跨PickScore/Aesthetic、OCR、GenEval任务轮换并累积各任务监督。当前局部实验固定单个GenEval教师，使用64个冻结状态、11–12个验证接受目标、16次更新，并只在Flow10索引1部署learner；其余查询用发布anchor。训练目标为transition mean匹配，五项质量未作为选择门槛或损失保证。

因此“匹配教师动作”不能推出任意终点评分单调增加，尤其当前比较对象已是较强的发布Student。教师整体任务能力也不能保证单次混合介入及冻结后缀的每项质量都更好。
官方依据：[方法介绍](https://github.com/ali-vilab/DiffusionOPD)、[固定版本配置](https://raw.githubusercontent.com/ali-vilab/DiffusionOPD/0c85bf0193a6da24bb1f72152b612d353a154999/config/opd.py)。

## 下降已经可分解，但完整因果尚未识别

在同192个训练状态，理想验证策略的Aesthetic相对anchor为−0.077，拟合后为−0.382。34个接受教师目标本身Aesthetic下降，拒绝158状态的共享参数变化约贡献78.8%的训练净下降。这个比例仅是同队列算术分解，不能宣称解释了开发下降的79%。

Phase57完整原动作监督压低拒绝漂移约97%，恢复四项质量，但开发joint也下降。这与专项续训改变原有能力的解释一致，尚不能排除其它协议因素或归因于单一机制。评分器/图像/缓存身份已有核验；自动分数不等于视觉真值，但四项一致下降不能忽略。
来源：[质量分解](../experiments/local_expert_v56/QUALITY_DIAGNOSIS.md)、[Phase57审计](../experiments/local_expert_v57_followup/SUMMARY.md)。

## 明确的配方差异与下一步优先级

本次核对确认：官方固定版本MOPD和SOPD-GenEval均设GenEval教师guidance_scale=1.0；Phase55/57配置和代码断言均使用4.5。后端实际根据teacher_guidance执行对应CFG，因此这是有效行为差异。学生CFG均为4.5并不要求教师也必须4.5。

这项差异在旧局部协议中是显式声明的，不能事后改写旧结果，也不能据此断言质量下降已找到唯一根因。当前缺少在同批状态、同教师权重、同后缀下只改教师CFG的配对证据。

下一步优先核查教师工作点和共同起点OPD对照，再扩大新的保持机制。最先可冻结的GPU诊断应当是无需训练的教师CFG1.0 vs4.5局部候选对照，统一报告任务及全部五质量；采样、数据与起点固定，不扫多个CFG、不用确认/最终集。该方案随后已落实为[Phase58冻结协议](../experiments/local_expert_v58/PROTOCOL.md)，Slurm1064447现已COMPLETED，实际1022/1800 L40秒。joint74→84/192、五质量相对4.5均改善；joint区间跨0且相对anchor仍有质量下降。学习效果由独立Phase59检验。

随后需要清楚区分两个基线：官方原数据/原recipe的原生复现；以及共同发布Student起点/共同任务数据的OPD适配续训。后者必须披露适配，不能称完全原版。现有固定/随机局部监督只能作为机制控制，不能替代这些基线。

此前“三态保持 vs 同量随机保持”草案保留为后续机制提案，优先级置于教师配置核查之后。该提案尚未实施，不增加本次GPU开销。

## Phase58实测补充（04:00续看）

全192状态的CFG1−4.5：PickScore/26 +0.026916、CLIP +0.010106、Aesthetic +0.305105、HPS +0.026201、ImageReward +0.284305，五个描述性区间下界均为正。joint差+5.208pp、区间[−1.563,+11.979]。这支持CFG工作点影响候选质量，但相对anchor前三项Pick/Aesthetic/HPS仍下降，且原接受目标34个的joint30→24；不能断言已解释或解决拟合后退化。[完整审计](../experiments/local_expert_v58_followup/SUMMARY.md)。

## Phase59学习补充（05:00续看）

CFG1验证在128dev保持旧验证的57个joint，相对旧验证五质量点估计均提高；Aesthetic区间跨0，其余四项区间为正。相对初始学生仍有Pick/Aesthetic/HPS/ImageReward下降，说明教师工作点修正只解决一部分问题。新随机53（旧63），因此新验证−随机+4题不代表验证成功数进一步提高。原GPU退出FAILED来自评分后的固定记录数断言，独立CPU已完整恢复原数据，未重跑GPU。[完整结果与修复范围](../experiments/local_expert_v59_followup/SUMMARY.md)。

## Phase60学习复核补充（06:00续看）

第二池新验证52/128，旧验证60，新随机50、旧随机52。新验证相对旧验证五质量点估计均改善，但joint−6.25pp、文字exact−10.9375pp、CER+0.2010；未通过冻结的任务不降门槛。相对初始学生，四项质量仍低于起点，CLIP提高。质量恢复与任务保持尚未同时成立。

训练直接修复只兑现5/10，尽管target MSE1.4128→0.0900。拒绝54状态也向教师方向移动，平均投影系数0.8686；训练Aesthetic净下降−0.2440中，拒绝状态加权变化为−0.2241。这是训练算术分解，不是开发退化的因果比例。两池未兑现修复与全局MSE的排序关系不一致，不支持盲目增加训练步数。详见[独立双池报告](../experiments/local_expert_v60_followup/report.html)及[下一步](../experiments/local_expert_v60_followup/NEXT.md)。

## 07:00配方审计：不能把本地退化直接归因于原版OPD

[源码与有效配置核对](../experiments/opd_recipe_audit_20260929_report_fix/report.html)确认：基础loss核及梯度相符，但状态来源、监督j、归一化、LR、EMA和部署方式不同。单教师本身不是偏离所有OPD，因为官方已有SOPD-GenEval。当前专用fit路径不使用通用config的EMA/accumulation/outer_epochs字段，应以实际执行日志和有效配置比较。

共同发布Student/联合数据的原生适配已做CPU预检查，未运行正式训练。两epoch的EMA不会更新，raw_lora必须单独核验。此审计不证明任何差异就是退化原因，也不改变Phase60停止扩大CFG1学习的判定。
