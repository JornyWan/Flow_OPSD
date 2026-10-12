# OPD Lab

<!-- bounded-expert-preparation:start -->
**2026-10-11 15:11 EDT：Phase81完整诊断流程已准备并冻结，202输入、36项CPU检查通过，尚未提交GPU。** 固定强原生Phase77 EMA、64训练prompt、Flow40低噪声query，比较noop/full/bounded/等幅random；256端点和1280五质量，主比较bounded−random joint。教师临时加载后删除，所有学生后缀需精确noop回放；真实GPU验收仍待执行。不会把目标机会当学习成功。

四分区probe后，最早preempt预计明天02:04EDT，75分钟完整时限无法落在03:11EDT截止前；指定空闲ai-gpu21的test-only也未提前。因此未reserve、未提交，没有GPU失败或重复作业。预算仍20700/48600请求秒，余1作业名额；下一既有回调21:11UTC重新检查排队变化。零读图，确认/最终封存，Phase79/80效果结论未变。[冻结协议](experiments/local_expert_v81/PROTOCOL.md) · [完整状态](experiments/local_expert_v81_followup/SUMMARY.md)。
<!-- bounded-expert-preparation:end -->

<!-- phase80-current:start -->
**2026-10-11 11:11 EDT续看：Phase80完成并独立审计，尚未支持普遍梯度冲突解释。** Slurm1077779 COMPLETED0:0，实际765/请求4500 L40S秒；179冻结输入、三权重/潜变量哈希、192完整记录与汇总核对通过。三权重首状态GPU loss/梯度合同差均0。无训练、新图或新质量评分；确认/最终封存、agent读图0。

16训练prompt的64共同S0状态上，原生/两步EMA的full-vs-local梯度cosine为0.850/0.817，full-vs-no-Jacobian为0.963/0.962，均无反向状态。Jacobian并非零或必然无用；单状态方向也不等于Adam实际更新。两步模型j0首步误差16/16高于原生，j3为14/16，较晚j6/j8各10/16较低；总体loss接近。提示时间步拟合取舍，不能证明它导致dev损失。

下一候选是**保留首步监督的等权首步/两步loss**，检验endpoint-only是否牺牲局部拟合；先完成接口/梯度/预算检查，不盲目detach或加fit。候选仅草案、未提交，终端验证筛选仍未加入。Phase79仍为EMA joint50对原生49，区间跨0；RAW43对45及HPS下降保留，全五质量不删。

请求预算仍20700/48600秒，新增1/2作业，实际两作业合计15117秒，额度不退款；最多还可1项≤16200秒。本次未新增GPU。现有timer存活，下次17:11UTC（13:11EDT）。[独立审计](experiments/local_expert_v80_followup/completion_20261011/SUMMARY.md)；[下一候选草案](experiments/autonomy_20261011_2h/PREFIX_SUPERVISION_DRAFT.md)。
<!-- phase80-current:end -->

<!-- phase79-result:start -->
**2026-10-11：Phase79完整完成并独立审计，尚未证明超过原生OPD。** Slurm1077741 COMPLETED0:0，实际14352/16200 L40S秒；256真实更新、完整权重/EMA/来源/数据/LR/评分一致性检查通过，384新端点与1920主要质量齐全。确认/最终集封存，agent读图0。

主比较最终EMA：两步joint **50/128** 对原生 **49/128**，+0.78125pp，探索性配对95%区间[-7.8125,+8.6133]pp；16新增、15损失。OCR72→75，结构90→85，CER0.197563→0.179771，A_unknown30→24、B_unknown均0。家族joint颜色21→17、计数14→14、多重4→2、关系10→17；不挑关系类当总体成功。RAW43对45，HPS差区间全负，完整保留。

EMA五质量差：PickScore−0.000467、CLIP+0.002432、美学−0.001126、HPS−0.001527、ImageReward+0.006740，区间均跨0。训练epoch耗时1.875倍，额外教师查询/j9覆盖仍在，不能称算力匹配或效率优势。此轮测试两步学习组件，终端验证筛选尚未加入。学习到接近原生的能力，但方法超越仍待证实；也不能将区间跨0当作等效或idea无效。

下一步先诊断终点误差对局部误差的掩盖与梯度传播。独立梯度分解核心9项CPU检查通过，真实SD3接线/GPU验收未完成；拟定4500秒诊断尚未预留或提交。当前不盲目加fit/扫LR/挑seed；本次新增GPU0。全部报告、反证、统计限制与下一诊断见[完成报告](experiments/local_expert_v79_followup/completion_20261011/SUMMARY.md)。
<!-- phase79-result:end -->

<!-- phase79-current:start -->
**2026-10-11 03:05 EDT：Phase78完整通过，Phase79完整学习比较已启动。** Phase78作业1077722 COMPLETED0:0，实际510/1800 L40S秒；8真实更新、EMA/导出差0、峰值37.079GiB、平均44.085秒/轮，首8轮prompt序列与Phase77相同。它是工程成功，未评任务收益。

Phase79作业**1077741**在preempt_gpu/ai-gpu21运行，单L40S协议，16200秒(4.5小时)硬上限，无requeue/自动重试/timer。四分区probe完成，sablab跨项目承诺2/6，本项目未占用其配额。预计训练11286秒＋评测约3194秒，另留启动余量，不保证如期结束。

从与Phase77相同纯底座/fresh LoRA开始，固定256更新、batch3、实际LR3e-4、密集anchor；使用Phase78已验收两步闭环目标。**主比较为新方法最终EMA−原生OPD最终EMA(49/128)，不是只比较初始底座。** 最终RAW诊断；384新图/1920主要五质量及完整joint/OCR/结构/CER/unknown，共同Flow40部署。重生成128 S0要求与Phase77图像/任务精确一致、质量差<=1e-4，全256轮prompt/LR日程核对。16项新增比较/评测检查通过，含真实旧评分与checkpoint。

两步增加教师查询/学生计算和j9覆盖，不能称同信息或同GPU算力对照。终端验证筛选尚未加入；本轮只测试新的学习组件。CLIP单项小降不自动阻止推进，所有质量/损伤仍完整报告，旧Phase77门槛不改。结果仍待完整训练/评测；确认512/final768封存、零agent读图。
<!-- phase79-current:end -->


<!-- phase78-current:start -->
**2026-10-11：Phase78 两步闭环目标已接入原生训练入口，1077722在preempt_gpu/ai-gpu21运行。** 单L40S、8更新、1800请求秒，无requeue/自动重试/timer。25项CPU检查通过；真实GPU microbatch1的一步退化输出/梯度差0、完整两步与重计算输出/梯度差0，前置验收已过；8次完整batch3训练/EMA/耗时验收仍进行中。源码与四分区probe已冻结，完成Phase77档案未改。

这是新方法的短轨迹学习组件：教师和学生从同一behavior状态各走两步，学生梯度穿过自己的中间状态，教师目标detach。按dt1/(dt1+dt2)缩放终点误差，保持常速度情形的原一步时间权重；第一步前向重计算节省显存。保留实际LR3e-4、密集9anchor/81例每更新、原生Adam/EMA。额外教师查询10→19/轨迹、学生计算增加且覆盖j9，不能称同算力/同信息对照。

用户接受CLIP单项小幅下降，不将其作为后续推进的自动否决条件；所有质量和损伤仍完整报告，不改Phase77旧门槛。Phase77 joint17→49是原生对照，**尚非我们的创新收益**。本轮不评dev或五质量，不证明超过OPD；通过后按实测成本冻结同起点256更新完整学习比较。终端验证筛选尚未加入，确认/最终集封存，agent读图0。
<!-- phase78-current:end -->


<!-- phase77-result:start -->
**2026-10-10：Phase77完整完成并审计，1077269 COMPLETED0:0，实际9225/10800 L40S秒。** 原生SOPD-GenEval从纯SD3.5M训练256次，20736状态、覆盖507/512训练prompt；最终EMA/RAW权重链通过，384同条件Flow40开发图和全部1920主要质量值齐全。2旧任务参考完全一致、40旧质量分数差0，零agent读图。

同128探索dev：S0/最终EMA/最终RAW joint **17/49/45**，OCR **73/72/70**，结构 **32/90/88**。预声明EMA−S0 joint **+25pp**，描述性95%区间[+14.84,+35.16]pp；新增44、损失12，其中9例OCR损失、3例结构损失。颜色/计数/关系主要提升，多重约束仅3→4/32。

EMA相对底座五质量：PickScore+0.000925、CLIP−0.005792、美学+0.152862、HPS+0.014832、ImageReward+0.349881；美学/HPS/ImageReward区间全正，PickScore与CLIP区间跨0。冻结“五质量均值都不降”条件未全过，不称无代价提升。RAW也有明显joint收益，不能挑RAW替代预声明EMA。

这是**原生对照真正学到收益**，尚未证明我们的创新方法超过OPD。不能把相对旧实验的改观唯一归因于训练量，原生实现/覆盖/EMA/采样等也不同。下一步同起点、实际LR、数据/更新预算匹配的两步闭环学习，先验收one-step退化一致性，另报额外计算；不同时加入筛选机制。确认/最终集继续封存，同128dev属探索性，无新GPU/timer。
<!-- phase77-result:end -->


<!-- phase77-current:start -->
**2026-10-09 23:05 EDT：Phase77 已提交并启动，作业1077269，minilab-gpu/ai-gpu12。** 单张L40S协议，训练加评测硬上限10800请求GPU秒；无自动重试/requeue/timer。四分区已probe；sablab现有跨项目分配2＋pending12，本项目未向其新增承诺，未改其他作业。

本轮建立原生任务适配SOPD-GenEval强对照：纯SD3.5M/fresh LoRA开始，官方batch3/27反向/81状态每更新，固定LR3e-4，256实际更新，共2304轨迹抽样/20736监督状态。保留官方loss、FP16和重复step EMA；最终256 EMA为主、RAW仅诊断。256是覆盖/预算选择，不是已收敛或论文规模复现。

训练后自动完整评估S0/最终EMA/最终RAW：相同128探索dev、相同噪声，统一Flow40/CFG4.5/text128/512px、FP32 CFG组合和单adapter，384新图；完整joint/OCR/结构/CER/unknown与五质量、配对区间。旧Flow10/published-student结果不替代新S0。2个既存任务参考要求证据完全一致，8个既存质量参考40值容差1e-4。新评测链10项CPU检查通过，含真实Phase76 EMA导出/状态检查差0；当前已核对3/256真实Adam更新、实际LR3e-4、每次81状态、实际L40S及共同初始哈希。256训练与完整评测尚待完成，**尚无本轮效果结果，也未测试创新方法**。

确认512/final768封存、零agent读图。下一项是与该原生对照匹配的固定两步闭环学习；若结构提升但OCR/joint下降，先分析专家任务取舍，不默认更多fit能解决。
<!-- phase77-current:end -->


<!-- phase76-result:start -->
**2026-10-09晚：Phase76官方完整训练循环通过，1077242 COMPLETED0:0，实际250/1800 L40S秒。** 从纯SD3.5M/fresh LoRA开始，官方batch3/每轮9轨迹81状态，完成8真实Adam更新、648监督状态，实际LR3e-4无跳步。见过59/512训练prompt；峰值显存36.83GiB，均值23.65秒/轮。原生step7三次EMA调用及最终checkpoint重放差0，RAW/EMA均已改变。20项CPU＋6项报告检查通过。

这是完整主循环/显存/吞吐验收，未评任务与五质量，不能当作方法提升或充分收敛。256更新配置与384图完整评测草案已写出；训练部分粗估101分钟，评分整合/总预算仍待冻结，256训练尚未提交。下一步进入足量原生任务适配OPD及匹配闭环学习；不再以本地数值必须逐位等价作为无期限前置条件。零agent读图，确认/最终集封存，无重试/timer。

[本轮结果与下一轮配置](experiments/local_expert_v76_followup/SUMMARY.md)。
<!-- phase76-result:end -->


<!-- phase75-result:start -->
**2026-10-09：Phase75三路径消融完整完成并独立审计，1076817 COMPLETED0:0，实际250/1800 L40S秒。** 同27状态/教师目标/初始权重各1更新：仅修CFG为FP32后，相对官方梯度差0.9707%→0.7517%、整体参数更新差2.8247%→2.4732%；原梯度/参数门槛仍未过。目标差0，loss差缩至1.16e-10，RAW单adapter重放差0。共同部署单prompt/new-seed终点latent差3.43%→3.10%，不能据此推断图像质量或任务成败。171输入/5张量档案核对通过；无新图/任务/五质量结果。

结论是CFG修正部分缩小实现偏差，不能直接把数值门槛未过判成idea无效。下一步优先用官方入口建立足量任务适配OPD，再做匹配的固定两步闭环学习；不无限追求逐位一致。闭环数学核心9项CPU检查通过，尚未接入真实训练/GPU。确认/最终集封存，零读图，无新GPU/timer。

[结果、解释与后续](experiments/local_expert_v75_execution_20261009/SUMMARY.md) · [下一学习比较](experiments/local_expert_v75_execution_20261009/NEXT.md)。
<!-- phase75-result:end -->


<!-- phase74-result:start -->
**2026-10-09：Phase74真实SD3数值验收完成，梯度/Adam参数一致性门槛未通过。** 作业1076775 FAILED1:0，minilab-gpu/ai-gpu13实际L40S，308/1800秒。教师目标差0、loss最大差4.49e-8，但梯度全局相对差0.9707%（阈值0.1%）、Adam后参数最大差5.19e-4（阈值3e-6）；27状态/同起点各1更新及RAW单adapter导出重放完成，重放差0。由阈值主动停止，未自动重试。

已独立加入显式CFG组合精度选项，默认保留旧行为；31项相关CPU检查通过，尚未GPU验证修复是否足够。官方FP32输出后CFG与本地先CFG后FP32有已证实差异，但本轮不足以唯一归因全部更新偏差，也不证明此前质量退化根因。无新图或任务/五质量结果，确认/最终集封存。下一步同状态三臂精度消融，再做充分训练的任务适配OPD；短轨迹创新实验继续等待学习链验收。

[完整结果、账务、候选修复和下一步](experiments/local_expert_v74_execution_20261009/SUMMARY.md)。以下为历史记录。
<!-- phase74-result:end -->

<!-- phase74-current:start -->
**2026-10-08：Phase74原生/本地学习路径验收已实现，25项CPU检查通过，真实GPU验收待资源。** 官方81状态/27次反向累积的CPU梯度最大差1.75e-10、Adam后参数差0。发现原生EMA在同global_step重复调用与本地每更新调用不同；另确认官方Accelerate先转FP32再组合CFG，而本地先组合再转FP32。二者都不能直接解释此前全部RAW退化。

原准备包遗漏原生AMP输出转换，已在提交前发现并独立准备修订包；本地算法和完成档案未改。四分区probe未找到协议规定一小时内可启动的L40S，GPU提交0；单次修订验收上限1800秒，无重试/timer。下一步先测真实SD3目标/梯度/Adam/导出重放，之后才冻结充分训练的任务适配OPD对照。短轨迹方法尚未实现，尚无超越OPD的新证据；零读图，confirmation/final封存。

[本轮发现、修订与资源状态](experiments/local_expert_v74_followup/SUMMARY.md)。以下保留历史记录。
<!-- phase74-current:end -->

<!-- phase73-current:start -->
**2026-10-08：Phase73补评分完整结束，OCR局部机会门槛未通过，停止扩大当前路由。** 作业1074344 COMPLETED0:0，实际148/900 L40S秒；320主端点/1600五质量齐全，8旧参考40值最大差0，15项冻结CPU检查通过。原Phase72 FAILED档案保留；本次无生成、训练或任务重评，零读图。

固定64训练prompt，S0/GenEval局部/GenEval完整/OCR局部/OCR完整：joint **10/11/22/7/7**，OCR **27/28/32/35/53**，结构 **16/24/46/12/9**。OCR局部新增8例文字正确但只有1例同时joint成功；4例joint损失均为结构损失。独有修复1、净增−3，未达到原≥4/≥2门槛。

GenEval完整相对S0五质量及其描述性区间均为正；OCR局部四均值下降、CLIP升但区间均跨0；OCR完整五均值均降，其中美学/HPS/ImageReward区间全负。全部指标与区间已封存。这是训练cohort专家前向诊断，不是学生增益；GenEval完整CFG1与S0CFG4.5，不能将差异仅归因于权重。

下一优先项是验收完整OPD学习路径并建立足量、同任务/教师信息的强对照，查清教师能力为何未转为学生收益，再检验验证目标选择；不盲目加OCR或fit。尚未提交下一训练。确认512/final768封存，无新timer。**当前仍未证实我们的方法超越匹配OPD。**

[完整HTML](experiments/local_expert_v73_cpu_report/report.html) · [解释与五质量](experiments/local_expert_v73_followup/SUMMARY.md) · [下一步实施准备](experiments/local_expert_v73_followup/NEXT.md)。以下保留前序带日期记录。
<!-- phase73-current:end -->

<!-- phase72-current:start -->
**2026-10-05：Phase72 作业1067893 FAILED 1:0，已完成独立 CPU 接口修复，五质量仍未完成。** 实际1084 L40S秒，原4500请求秒不退。本次未新增GPU、重试或timer，零读图，原失败目录/运行快照不改。

故障是新增endpoint漏写质量token audit需要的 `word`，不是模型训练失败。320端点和320任务评分齐全；128旧图摘要记录及完整A证据匹配，四adapter不变、0更新、预测次数符合计划。缺全部1600质量值，因此暂不判资格门槛或方法效果。此前21项合成测试未覆盖真实评分接口；现33项CPU测试通过，包含出错接口回归测试。已独立准备只添加冻结OCR目标字段的完整输入副本；GPU补评分尚未执行。

下一步只需复用现存生成结果补五质量，并按原门槛解释专家互补性，再决定学习协议。Phase71完整反例仍为base/all/veto/random joint9/5/3/4，尚未证实超越baseline。保留任务平衡OPD监督、用终端证据替换基准专家目标仍是假说；必须有相同专家信息的强匹配控制、充分训练与完整学生评估。确认512/final768封存。

[失败与修复核对](experiments/local_expert_v72_cpu_repair/SUMMARY.md) · [独立补评草案，未提交](experiments/local_expert_v72_cpu_repair/RECOVERY_PROTOCOL_DRAFT.md) · [研究反思](experiments/research_reset_20261003/REFLECTION.md)。旧Phase72 status.json仅为当时启动记录，最新真实状态以此独立审计和sacct为准。
<!-- phase72-current:end -->

<!-- phase71-live-monitor:start -->
**2026-10-01 06:16 EDT：Phase71三臂完整结束，冻结门槛未通过，停止扩大当前veto配方。** all1066045/veto1066041/random1066093均COMPLETED0:0；最终16RAW、384新开发终点和1920五质量全部通过来源/权重/评分核对。相同128题，base/all/veto/random的joint为9/5/3/4，OCR54/11/8/5，结构28/65/68/69。veto−random joint −0.78125pp，描述性95%区间[-3.125,+1.5625]，未证实筛选优势。

veto相对random五质量为PickScore微降、CLIP/美学微升、HPS/ImageReward下降，HPS描述性区间全负；相对base/all四质量下降、仅美学提高。实际每臂16次同LR更新，veto/random各987目标，同任务/模板/时间数量严格匹配；1024名单决定仅30条不同，后续在线状态不配对。训练修复兑现本轮未测。旧native12/old_all13仅历史背景，不是本轮同日程或完整MOPD证明。

实际GPU17478 L40S秒（4.855小时）；请求预算32400/32400已满，未用时限不退，不再提交GPU。12小时窗口已于17:05:15 UTC（13:05:15 EDT）到期，原timer已退出，36次回调均有接收记录；没有后续唤醒或预算结转，不重复完成分析。完整报告与解释已封存；确认512/final768未用、agent读图0。下一问题是同当前起点的专家联合任务能力与局部互补性，现有草案未冻结/未提交，不能预设加专家或更多fit有效。

此前已完成[CPU来源与路径准备](experiments/teacher_qualification_preparation_20261001/SUMMARY.md)：固定64训练病例/五路由，19来源SHA和7项新检查通过；识别旧诊断起点/队列不适配、未保存latent前缀及alpha=1计算路径差异。未进行新生成、训练或GPU验收，正式协议/权重链/评分集成仍待完成；不是新的正向结果。

[完整三臂HTML](experiments/local_expert_v71_cpu_report/report.html) · [解释与全部CSV](experiments/local_expert_v71_followup/SUMMARY.md) · [本次检查](experiments/autonomy_20261001_12h/checkins/20261001T170515Z.md)。

**窗口已收尾，研究目标尚未实现。** [12小时总结与下一步](experiments/autonomy_20261001_12h/CLOSEOUT.md)。本次仅核对账务/退出状态并整理交接，无新实验结果；零读图，完成档案未改。
<!-- phase71-live-monitor:end -->

**2026-10-01 02:32 EDT首批验收：** Phase71两组完成1/16轮。64初始状态、记录的128候选哈希及A证据跨臂完全一致；all学习64/64，veto只屏蔽5条已知损伤、学习59/64，含37条unknown弃权保留。实际LR均3e-4，各完成一次更新，评分技术错误0。新规则已在真实SD3首批按设计执行，完整训练/随机对照/开发任务与五质量仍待完成，不能当作学习增益。预算21600/32400未变，无新提交、零读图。 [首批数值核对](experiments/local_expert_v71/runtime_audits/first_epoch.json)。

**2026-10-01 02:17 EDT：Phase71 已启动两组，尚无新效果结果。** 全部监督 all `1066045`、明确损伤屏蔽 veto `1066041` 均在 preempt_gpu/ai-gpu21 运行；各1×实际L40S、3小时硬上限，Requeue=0。287项输入指纹/共同配置/源码核对通过，两组fresh LoRA与禁用LoRA的初始输出逐位相同。每次提交前四分区probe完成：普通兼容分区预计等待超出窗口，sablab无兼容L40且跨项目承诺5/6；按已冻结中断即停止规则使用preempt_gpu。

本轮比较保留广覆盖OPD、只屏蔽已知损伤与全部监督/同量同时间随机屏蔽；unknown是验证器弃权，保留基础监督，不称安全。三臂同实际LR3e-4、1fit、16轮/64训练、最终RAW/128旧dev及全部任务与五质量。随机从共同纯底座开始，须待veto完整COMPLETED 0:0再提交。请求预算已预留21600/32400秒，余10800留给随机，失败不返还、不重试。旧Phase68 dev验证8/128、随机9/128的结论未改变；确认/最终集仍封存，agent读图0。

[当前作业与下一步](experiments/local_expert_v71/STATUS.md) · [CPU前置核对](experiments/local_expert_v71/cpu_preflight.json) · [活跃交接](experiments/autonomy_20261001_12h/HANDOFF.md)。以下为带时间的历史记录。

**2026-10-01 02:04 EDT：三臂执行、审计和完整报告已实现，69项CPU检查通过。** 包含A证据/实际mask/LR重放、最终RAW核对、fresh随机、完整768终点/3840质量报告和一次性提交保护。旧base复用的6个生成/评分模块与Phase66 native/all快照完全一致。Phase71正在单一CPU进程214501中做完整资源指纹和输入冻结，尚未完成准备、未预留或提交GPU；不可重复启动。预算0/32400，零agent读图。

[执行实现与准备状态](experiments/veto_execution_implementation_20261001/SUMMARY.md) · [冻结研究合同](experiments/paired_veto_implementation_20261001/RESEARCH_PROTOCOL.md)。以下为前序记录。

**2026-10-01 01:41 EDT：成对时间日程＋明确损伤veto已接入共享训练引擎，125项CPU检查通过。** 新规则保留unknown并标记验证器弃权，只屏蔽确认true→false；随机匹配任务/模板/时间/数量并从共同初始输入重新训练。无损伤时与all的参数、Adam和EMA完全相同。真实1024查询形成512对，旧prompt顺序/noise不变；旧v2日程重算完全一致。三臂研究合同已独立固定，下一项补控制器与完整RAW/LR/A证据/任务/五质量报告，尚未GPU验收或提交。预算0/32400 L40S秒，零agent读图。

[实现与验收](experiments/paired_veto_implementation_20261001/SUMMARY.md) · [三臂研究合同](experiments/paired_veto_implementation_20261001/RESEARCH_PROTOCOL.md) · [执行待办](experiments/paired_veto_implementation_20261001/NEXT.md)。以下为前序记录。

**01:22 EDT CPU续查：** 5120条训练A记录重放一致。Phase68验证状态中明确损伤56/1024；若仅屏蔽这些状态将保留968项，但大量仍unknown，不能声称安全。原日程按批次/任务/模板/时间匹配后，56个veto中51个没有替代选择，随机对照几乎相同；先设计有可交换候选的时间平衡日程，再决定独立协议。6项检查通过，无新GPU。 [分解与对照可行性](experiments/selection_feasibility_20261001/SUMMARY.md)。

**2026-10-01 01:16 EDT：OPD原理审计发现监督覆盖与时间权重的额外差异，先研究再决定新训练。** Phase68两组均51/1024目标、35训练prompt且实际LR匹配，但验证44/51集中在索引0–3、索引7–8为0；随机对应35/51和6/51。原损失带Δt²，随机保留的系数总量是验证的2.2565倍。这不是梯度/Adam位移比例，也不是已证明根因。均匀单查询可无偏估计全时间平均目标；严格正例mask改变了监督支持和权重。更多fit只能检验兑现，不能补回缺失覆盖。5项数学边界CPU测试、14来源SHA及两臂统计重算通过，无新模型结果。

下一项先用冻结训练A记录分解明确损伤/收益/unknown及同任务同时间随机对照的可行性，再决定原4fit诊断或另立“保留OPD监督、只屏蔽已知损伤”的协议；不改旧门槛。Phase68完整dev仍验证8/128、随机9/128；Phase70直接教师修复明显但最终局部未兑现。论文仍缺强匹配OPD基线和稳定完整部署增益。

用户新授权的12小时窗口为**10月1日01:05–13:05 EDT，每20分钟续看**。本地同线程queue进程4026239运行，同线程接收测试已于05:18:28 UTC确认；非原生云定时。独立上限32400 L40S请求秒/4作业、每作业≤10800秒，目前预留0/提交0。本线无运行GPU；确认/最终集封存，agent读图0。

[原理、证据与新发现](experiments/opd_foundations_20261001/SUMMARY.md) · [下一步可证伪问题](experiments/opd_foundations_20261001/NEXT.md) · [12小时活跃交接](experiments/autonomy_20261001_12h/HANDOFF.md)。以下带日期内容为历史记录。

**2026-09-30 22:45 EDT：多次拟合的记录与独立审计已实现，118项CPU测试通过。** 每个fit检查状态/目标固定，记录每步实际LR及整块前后真实权重，空批不更新；记录开关在Toy训练中不改变参数/Adam/EMA。Phase68的102个真实SD3保存状态、32个行为快照也通过CPU来源及张量接口检查。4fit块后模型明确命名post_fit_block，旧单fit审计保持原限制。下一轮研究合同已固定，尚需接入机制前向/完整报告并冻结执行输入，GPU尚未提交。本轮没有新的模型收益结果，agent读图0。

[实现与验收](experiments/repeated_fit_implementation_20260930/SUMMARY.md) · [下一轮研究合同](experiments/repeated_fit_implementation_20260930/LEARNING_PROTOCOL.md)。以下带日期段落为历史记录。

**2026-09-30 15:20 EDT：Phase70质量补评完整完成，瓶颈定位更明确。** Slurm1065386 COMPLETED 0:0，实际99/900 L40S秒；510主终点2550质量值补齐，8参考图40值与原记录完全一致。验证组自身51训练状态的joint为behavior1、teacher33、即时局部1、最终局部1、最终后缀2；直接教师的五质量改善区间均为正，但32个直接修复最终局部保留0。随机自身状态也有教师收益，因此这支持学习兑现不足，不证明筛选优势。Phase68完整开发仍验证8/128、随机9/128，门槛未通过。原Phase69 FAILED与夜间档案不改，零agent读图；当前无运行GPU或新timer。下一步先完成多次拟合独立权重审计，再冻结同实际LR、同量随机的1/4次fit比较，未提交新训练。

[完整机制HTML](experiments/local_expert_v70_cpu_report/report.html) · [解释与全部差值](experiments/local_expert_v70_followup/SUMMARY.md) · [下一比较草案](experiments/local_expert_v70_followup/NEXT_PROTOCOL_DRAFT.md)。以下带日期段落为历史状态，质量缺失/未验收的旧描述已由本次独立补评更新。

**2026-09-30 09:00 EDT夜间收尾：8次续看全部接收，原timer已退出。** Phase68完整对照未建立筛选优势：同128dev验证/随机joint8/9，五质量均值相对纯底座提高，但冻结门槛未通过。Phase69在质量入口因cohort元数据遗漏技术失败，已完成任务部分显示验证组32个教师直接修复的即时/最终局部保留0，五质量仍0/2550。接口已CPU修复，几何与多次拟合准备已独立归档，尚无修复后GPU验收或新训练。本夜请求18900/23400、实际12547 L40S秒（含Phase68原作业全时长）；未使用额度不结转，失败未重试。目前无本线运行GPU/后续唤醒，零agent读图。研究目标尚未实现，下一步需补质量证据与新配置审计，再决定匹配学习比较。

[完整早晨交接（任务、五质量、预算与下一步）](experiments/overnight_20260930/MORNING.md) · [已完成Phase68报告](experiments/local_expert_v68_cpu_report/report.html) · [Phase69任务部分及失败边界](experiments/local_expert_v69_cpu_failure_audit/SUMMARY.md)。以下为历史记录。

**2026-09-30 04:00 EDT续看：Phase68完整对照未建立筛选优势；Phase69技术失败，任务部分已独立审计。** Phase68验证/随机joint为8/9（同128dev），五质量均值相对初始提高，但冻结门槛未通过。Phase69 Slurm1065085因新增诊断遗漏cohort元数据，在质量评分入口FAILED 1:0，实际1296/4500 L40S秒；这是实现错误，不是算法失败。102状态、204原PNG精确重放、510任务结果齐全，五质量0/2550。验证组自身51训练状态的behavior/teacher/即时局部/最终局部/最终后缀joint为1/33/1/1/2；32个直接修复即时与最终局部均保留0，后缀保留1。目标loss下降但动作修正未充分兑现，仍非开发效果或质量保持证据。独立CPU修复与52项测试通过，原档案不改、GPU不重试。本夜请求18900/23400不变，剩余额度不自动使用；已有timer下一次05:00 EDT。

[Phase68完整结果](experiments/local_expert_v68_cpu_report/report.html) · [Phase69任务部分/质量缺失](experiments/local_expert_v69_cpu_failure_audit/SUMMARY.md) · [实现修复](experiments/local_expert_v69_cpu_failure_audit/FIX.md) · [下一步未执行草案](experiments/local_expert_v69_cpu_failure_audit/NEXT_PROTOCOL_DRAFT.md)。

下一步先补齐质量证据合同；未来有限拟合比较必须保留同实际LR和同量随机，并以完整部署joint与五质量判定。Phase54已证明局部修复恢复不保证胜过随机，不能只增加步数或追逐训练修复率。以下带时间记录为历史状态。

**2026-09-30 06:00 UTC续看：** random正常推进至8/16轮、30目标；前8轮两组监督分层数量、实际LR和更新序列完全匹配。验证51接受中，训练A标记33个joint修复（24prompt）、18个部分修复，44个集中在索引0–3；这是训练标签统计，尚非B评测或学习兑现。未产生新开发结果，未新增GPU，保留完整随机比较后再作诊断的顺序。[记录](experiments/overnight_20260930/checkins/20260930T060000Z.md)。

**2026-09-30 05:19 UTC：Phase68验证组完成、随机运行。** 新验证joint8/128、OCR44/128，初始9/54；五质量均值与描述性区间相对初始均正，表现为质量恢复，尚未证明任务或筛选增益。旧验证joint1/OCR5；51接受/35训练prompt/16更新，真实权重链通过。新增接受状态分段诊断18项测试、合计40回归通过，未提交。用户新睡前授权：美东02–09点每小时续看；原GPU预算不变、额外最多2×4500秒，先冻结及预留，不自动重试，不读图。

[单臂完整证据](experiments/local_expert_v68_verified_interim_20260930/SUMMARY.md) · [下一诊断](experiments/accepted_state_diagnostic_design_20260930/README.md) · [夜间交接](experiments/overnight_20260930/MORNING.md)。以下带日期段落为历史进展，当前状态以上述链接为准。

**2026-09-30 04:05 UTC：运行中完成CPU问题定位。** Phase68前三轮192查询/10个真实接受目标已核验，首轮128候选hash及A证据一致，loss复算与前两次真实参数位移吻合。另发现严格gate会拒绝“原项unknown、教师项已true且另有确认修复”的候选；离线区间规则使旧验证缓存接受22→55/1024，初始64候选的回放joint10→11，但这是受已暴露B启发的事后分析、不是学生学习结果，且仍仅5.4%监督。PickScore/Aesthetic相对严格回放略降也完整报告。4项新CPU测试含729组合穷举通过。当前规则/学习率/预算不改，没有新增GPU；下一项机制诊断应检查本轮实际接受状态上的教师→拟合动作→后缀收益。 [证据、反证与诊断路径](docs/phase68_failure_diagnosis.md)。

**2026-09-29 23:45 EDT：Phase68匹配步幅诊断已提交。** verified1065020在scu-gpu运行，random1065024在preempt_gpu等afterok。两臂实际lr=3e-4×接受数/64，唯一算法变量；同纯底座、64训练、16轮、128旧dev和全部五质量，保存真实选中状态及参数步幅。121项CPU检查与155冻结输入核验通过。两个1×L40S作业各2小时上限，总请求14400 GPU秒，无自动重试/新定时器/agent读图。尚无新效果；稳定性恢复不等于筛选有效，confirmation/final封存。 [协议](experiments/local_expert_v68/PROTOCOL.md) · [状态](experiments/local_expert_v68/STATUS.md)。

**2026-09-30 UTC：完成OPD关系与失败设计审计。** 当前保留官方确定性局部loss，但已改成单查询＋严格正例mask，不能称完整原生OPD。验证仅22/1024目标；837含unknown（815涉及文字），156有确认修复且无已知损伤仍因unknown拒绝，原因可重叠。3072条三臂选择决策精确重放，99项CPU检查通过；未找到能独立解释退化的实现错误。稀疏步幅候选代码已实现测试，尚未冻结Phase68运行协议或提交GPU。旧续训/双adapter与当前纯底座/单adapter在论文计划中已分清。 [完整解释](docs/opd_relationship_and_failure.md)。

**2026-09-29 23:00 EDT：Phase66/67全部完成并归档。** 纯底座/原生SOPD-GenEval/all/verified/random的128开发题joint为9/12/13/1/1，OCR为54/22/28/5/3。verified−random为0pp、描述性CI[-2.344,+2.344]pp；两稀疏臂均22目标/11更新且逐批计数一致，不能只归因于验证器。验证五质量相对初始CI全负；相对随机五项均值低、其中四项CI全负。当前严格筛选配置停止扩大。

Phase67同64初始训练状态：验证学生一步/接管后缀/全程joint为10→7→0、OCR28→18→4；全程相对后缀joint区间全负，后缀及前缀扩展各有四质量明确下降。目标MSE降低未转成完整生成收益。直接教师joint9对初始10，本批平均任务增益也未建立；2个直接B修复在两种学生局部动作中均未保留。训练状态诊断不能替代开发结果或全22历史目标兑现率。

两个作业均COMPLETED 0:0。640开发＋512机制终点、5760质量值齐全；128原PNG、128 A子句、16路由恒等检查通过，随机RAW/优化器/loss审计也通过。下一步优先隔离共享稀疏更新的稳定性，新的训练协议/预算尚未冻结或提交。原完成档案保持不变，确认/最终集封存，零agent读图。

[完整汇总HTML](experiments/phase66_67_closeout_20260930/report.html) · [当前状态](experiments/CURRENT.md) · [下一步边界](experiments/phase66_67_closeout_20260930/NEXT.md)。

**19:24 EDT实测核对：** Phase67已在实际L40S通过64/64初始状态及128/128原PNG逐位重现，正在生成分段对照，尚无收益结果。Phase66随机完成12/16轮、9次更新，计数匹配继续通过。[为何这样诊断与解释边界](docs/learning_gap_diagnostic.md)。

**2026-09-29 19:20 EDT：Phase67机制诊断已启动，Phase66随机对照继续。** 新作业1064833在preempt_gpu，独立1×L40S×75分钟上限，Requeue=0；被抢占或技术失败停止，不重试。普通分区预计次日，首次真实提交前单独记录调度调整；sablab实际占用5张，本轮未增加。38项CPU检查通过，零agent读图。Phase66 random1064749已完成11/16轮、8次更新，已完成轮次的数量/族/模板与验证参考逐项一致，最终结果尚未齐。

已知纯底座/原生SOPD-GenEval/全目标/验证的128开发题joint仍为9/12/13/1；严格筛选组五质量均下降，不能扩展这套配置。新诊断在共同初始64状态比较最终学生的“一步动作＋初始后缀”“一步动作＋最终后缀”“完整最终生成”，以区分动作迁移、后缀与前缀影响；不是新训练或有效性证明。仅初始状态能严格重建，不能冒称复测后来22个接受目标；第一批原A仅接受1个，base/teacher A joint为8/7。要求原128张PNG哈希一致及16个零前缀路由恒等检查。全512终点/2560五质量；确认/最终集封存。

[当前状态](experiments/CURRENT.md) · [Phase67协议](experiments/local_expert_v67/PROTOCOL.md)。

**2026-09-29 18:21 EDT：Phase66三臂完成，严格正例组明显退化，停止扩大。** 纯底座/原生SOPD-GenEval适配/全部局部目标/验证的开发joint为9/12/13/1（各128题），OCR为54/22/28/5、结构28/66/63/24。验证−原生joint为−8.594pp，95%区间[-14.063,-3.906]；验证五质量相对base/native/all区间全部为负，已违反冻结推进必要条件。random1064749已按依赖启动，仍用于区分筛选偏差与稀疏学习，尚无最终结果。验证仅22/1024目标、11更新；2048条训练A重放、RAW/训练权重一致性和loss归一化核查通过，21项CPU测试通过，暂未发现这些实现错误。下一步完成随机和四臂归因，不自动放宽gate或扩大训练；图像读取0、新GPU0。 [最新完整分项](experiments/local_expert_v66_interim_20260929T2221Z/SUMMARY.md) · [检查与下一步](experiments/phase66_followup_20260929/NEXT.md) · [CURRENT](experiments/CURRENT.md)。

**历史回顾已集中管理：** [65轮HTML](presentations/2026-09-29_research_history_v1/index.html) / [讲解摘要](presentations/2026-09-29_research_history_v1/notes.md)，8个研究阶段，逐轮起点、结果、边界、原记录快照。阶段号不等于独立算法数；早期已有纯底座小实验，近期才是官方Student续训。今后按共同实验矩阵推进，把工程修复独立归档。

**2026-09-29最新：Phase65已补齐24图任务和全部五质量，独立CPU审计完成。** 评分差异已在原两候选上复现：训练默认/仅工作区模式完整证据8/8一致，确定性模式4/8一致但语义8/8一致，差异仅教师检测置信度。新评分合同通过；8dev初始/验证/随机joint为2/2/3，OCR为5/3/5，验证五质量相对初始均升、相对随机2升3降，仅作两次更新的工程小测。Slurm1064725仍FAILED：全部评分完成后的缓存锁mtime检查报错；52资源SHA和243冻结输入复核不变，CPU恢复报告独立保存，无GPU重试。实际247/900 L40秒。已实现并测试64/512任务×模板均衡日程，规模训练未提交；39项CPU检查通过，确认/最终集封存。 [报告](experiments/local_expert_v65_cpu_audit/report.html) · [审计](experiments/local_expert_v65_cpu_audit/SUMMARY.md) · [规模设计](experiments/scale_comparison_design_20260929/README.md)。

**2026-09-29 最新：Phase64两组训练已齐，评分一致性检查技术停止。** Slurm1064714 FAILED，实际290/1800 L40秒；原Phase63的185秒另计。随机臂完成2次更新，与验证臂逐批监督4/1一致、mask有6/32不同；首轮32候选图和A证据完全重放。24个共同开发终点已生成，但第一张训练候选A完整证据重评分不等，任务和五质量尚未计算，不能报告新效果。独立CPU补齐失败前取证和生成记录，19项检查通过，严格门槛未放宽。下一步仅诊断/评分旧图，不重训练或生成；当前无本线GPU或定时器，确认/最终集封存。[原记录](experiments/local_expert_v64/SUMMARY.md) · [CPU审计与下一步](experiments/local_expert_v64_cpu_parity_audit/SUMMARY.md)。

**2026-09-29下午更新：Phase62在线接口已验收，Phase63部分完成后技术停止。** `semantic_batch_v1`接入原engine，保留unknown/error并使用verified冻结的逐批分层数量匹配随机；105项CPU测试和历史192条决策重放通过。Phase63验证组两轮两次更新完成（接受4/16、1/16）；准备随机组时计算节点找不到git，Slurm1064571 FAILED，实际185/1800 L40秒，random和开发/五质量尚未运行。独立CPU修复改为克隆冻结输入，52项回归通过，真实随机目录已准备但未提交；没有重训verified或自动重提GPU。后续需补未执行的随机组与共同评测，不能报告新的效果增益。读图0，确认/最终集封存。

[在线接口](docs/semantic_online.md) · [Phase62](experiments/local_expert_v62/SUMMARY.md) · [Phase63原失败](experiments/local_expert_v63/SUMMARY.md) · [执行修复与下一步](experiments/local_expert_v63_cpu_execution_fix/SUMMARY.md)

**09:00夜间最终状态：已推进到Phase60，六次整点续看全部收到，定时器已自然退出。** Phase58三池候选joint74→84/192、五质量相对旧教师改善；Phase59/60两池学习的新验证/随机为57/53、52/50（共用128dev），选择增量区间均跨0。池2旧验证60→52，门槛失败，停止扩大CFG1；四项质量仍低于初始学生。真正专项收益存在，稳定筛选贡献/质量保持/强OPD对照/单adapter部署仍缺。全夜960新终点、4800对应质量，实际6228 L40秒；10800请求秒全部预留、余额不回收，未新增GPU。已封存夜间汇报，白天优先共同起点SOPD-GenEval适配和部署验收。[最终HTML](experiments/overnight_20260929_closeout/report.html) · [交接](experiments/overnight_20260929_closeout/SUMMARY.md) · [下一步](experiments/overnight_20260929_closeout/NEXT.md)。

**08:00 CPU比较检查已完成：** Phase59/60池内验证−随机的共同起点、状态、实际训练量、评测和部署匹配；只差名单及选择方式，不能把当前不稳定收益归于这些比较条件不一致。跨CFG同时改了目标数/名单/分层和实际训练量，保留整套配置比较、拒绝“纯幅度/同算力”标签。6臂1152既有终点、5760质量绑定及22项CPU检查通过；没有新预测。部分任务大权重旧记录仅size/mtime，正式协议须补完整指纹；两adapter、推理成本未知仍明列。正式共同起点OPD对照缺预算/成本/数据选择/checkpoint等8项，保持draft_not_submittable。本夜不再新增GPU，Phase60停止结论不变。[简洁HTML](experiments/comparison_contracts_20260929/report.html) · [检查范围](experiments/comparison_contracts_20260929/SUMMARY.md) · [下一步](experiments/comparison_contracts_20260929/NEXT.md)。

**07:00 CPU配方审计：** 官方MOPD和SOPD-GenEval默认每轮采新学生轨迹、监督前9步、lr3e-4并使用EMA；当前固定64状态、只训j=1、lr1e-4且专用路径不更新EMA。单项transition-mean loss及梯度一致，整体训练/部署协议不同。已有共同发布Student起点的512train/128dev原生适配预检查，但仅两epoch、未提交；短跑EMA不会更新，须单独核验raw权重。14项配方、35份源码/记录快照、17项CPU检查完成；没有新GPU结果或因果归因。[配方HTML](experiments/opd_recipe_audit_20260929_report_fix/report.html) · [解释](experiments/opd_recipe_audit_20260929_report_fix/SUMMARY.md) · [下一步比较合同](experiments/opd_recipe_audit_20260929_report_fix/NEXT.md)。

**当前（2026-09-29 06:00续看）：Phase60已正常完成并审计，第二池未通过冻结推进门槛。** 同128dev，池1旧验证/随机57/63→CFG1的57/53；池2旧60/52→CFG1的52/50。新验证−随机分别+3.125/+1.5625pp，区间均跨0；池2验证相对旧配置−6.25pp，不能用相对随机的小正差掩盖。两池五质量相对旧验证点估计均改善，但相对初始学生仍有四项下降。池2接受修复仅兑现5/10；CPU动作诊断显示拒绝状态也明显向教师移动，MSE不足以解释全部终点成败。本夜停止扩大CFG1学习、不追加GPU；实际6228 L40秒，请求10800秒已全部预留。 [双池简洁HTML](experiments/local_expert_v60_followup/report.html) · [完整审计与解释](experiments/local_expert_v60_followup/SUMMARY.md) · [下一步](experiments/local_expert_v60_followup/NEXT.md) · [夜间汇总](experiments/overnight_20260929/MORNING.md)。

**2026-09-29教师配方核对后的优先级更新：** 发布OPD Student在既有DrawBench评测上五项质量均高于原始SD3.5；当前下降发生在其后的单GenEval局部续训。另确认官方GenEval教师CFG=1.0、当前局部协议=4.5，候选影响现由Phase58配对诊断识别，Phase59/60两池学习均恢复部分质量，但第二池未保住joint，停止扩大CFG1学习配置。先核查教师工作点与共同起点OPD对照，再扩大三态保持机制；旧草案保留、未提交GPU。[完整说明](docs/opd_quality_gap.md)。

**2026-09-29研究复盘：** 新增零GPU诊断发现原A规则在已有128开发状态直接选择仅44个joint（初始38、固定教师50）；24个B教师修复只接受6个，漏掉18个均含unknown。三池158个拒绝状态中，A明确报告约束损伤30个，其余128个不能等同于已证实应保持。只对有损伤证据的状态保持并与同数量随机保持比较，保留为未提交机制草案；当前先补共同起点OPD适配配方核对。[问题与下一步](experiments/research_reset_20260929/SUMMARY.md)。

[项目概览：主线、实现与证据](docs/project_overview.md) · [当前实验状态](experiments/CURRENT.md) · [论文主计划](docs/paper_plan.md) · [汇报材料库](presentations/index.html) · [GPU调度规则](docs/gpu_scheduling.md)

本次mentor材料已封存：[离线HTML](presentations/2026-09-28_mentor_progress_v1/index.html) · [10页PDF](presentations/2026-09-28_mentor_progress_v1/mentor_progress.pdf) · [全部材料ZIP](presentations/2026-09-28_mentor_progress_v1/offline.zip)。10页主线、7页补充分开呈现历史正信号、对照与失败；图表、讲稿、来源、复现脚本统一管理，后续另建版本。

当前主线：**把经过终点验证的外部专家局部修正蒸馏给学生，研究同图文字、数量、颜色和关系的联合成功率。** 工程已经具备采集、冻结后缀验证、目标筛选、局部拟合、续训、配对自动评测与不可变归档。稳定的筛选增益、质量保持和正式论文主表尚未建立。

**Phase57已停止并完成两池审计：原动作监督恢复部分质量，但任务收益下降。** 三池中前两池GPU正常完成，管理器CPU读文件失败后停止，第三池未提交；第二池在独立CPU记录恢复分析，原失败与完成档案不改。开发验证masked→full分别57→39、60→43（各/128），主差−14.0625/−13.2813pp，逐池区间均为负。验证full−随机full为+1.5625/+3.1250pp，区间均跨0。PickScore、Aesthetic、HPS、ImageReward四项回升，CLIP略降且区间跨0；相对初始学生仍不能称质量保持。拒绝动作漂移约降97%，但接受目标MSE升约2.2倍，原直接修复兑现15/21→12/21。冻结的“每池主差不降”必要条件已失败；这是2/3池部分结果，不生成三池区间、不自动进入下一轮。实际6802/16200 L40秒，768新图/3840质量值；每次反向64相对原12/11/11，不是同算力比较。确认/最终集封存，本回调GPU0、定时器0、读图0。

[Phase57两池数值报告](experiments/local_expert_v57_cpu_recovery/report.html) · [审计与解释](experiments/local_expert_v57_followup/SUMMARY.md) · [下一步边界](experiments/local_expert_v57_followup/NEXT.md) · [冻结协议](experiments/local_expert_v57/PROTOCOL.md) · [原失败记录](experiments/local_expert_v57/failure.json)。

**Phase56已完成并审计：半幅改善质量，但未通过任务门槛，停止这次幅度搜索。** 全192个已有训练状态的联合成功为原学生43、单查询全幅教师修正74、半幅69；half−full为 **−2.604pp，描述性95%区间[−9.896,+4.688]**。五项质量相对全幅均提高且区间下界均>0，但joint点估计下降，不满足冻结的同时改善规则；不能据此进入半幅学习，也不能解释成已证明任务变差。半幅相对原学生仍有PickScore、Aesthetic、HPS下降。无训练、无新开发/确认/最终集；Slurm1064082真实COMPLETED，992/1800 L40秒，192新图/960质量值，旧40个质量值重评分最大差0。

[Phase56完整数值补充报告](experiments/local_expert_v56_cpu_report_completion/report.html) · [完成审计与解释](experiments/local_expert_v56_followup/SUMMARY.md) · [下一步边界与可证伪问题](experiments/local_expert_v56_followup/NEXT.md) · [冻结协议](experiments/local_expert_v56/PROTOCOL.md)。原归档508个文件与ZIP核验通过；缺失的domain分组和27个子指标区间在独立CPU记录补齐，原18个区间复算一致，原档案与判定不改。本回调新增GPU0、定时器0、agent读图0。

**最近一次完整三池学习实验是Phase55：三个独立训练池均已完成并归档。** 同128开发题，发布学生38/128；三池随机63、52、56，验证57、60、59。主差验证−随机分别−4.6875、+6.2500、+2.3438pp，等权平均 **+1.3021pp，探索性95%区间[−7.5521,+9.6354]**。按冻结规则为“不稳定/不确定”，没有达到确认推进门槛，不能宣称筛选稳定优于随机。

验证相对发布学生平均+16.1458pp，随机也提升+14.8438pp。训练原直接修复保留24/30，说明目标大部分可以学到；但双方训练joint均81/192，开发筛选贡献仍不确定。验证相对发布学生的PickScore、Aesthetic、HPSv2、ImageReward下降且区间完全小于0，CLIP区间跨0；质量保持未成立。独立CPU诊断另发现34个被接受教师目标仅5个在五项质量分数上均不下降。进一步在同192训练状态分解：理想验证策略（拒绝保持原动作）的Aesthetic差−0.077，真实拟合模型−0.382；约79%的训练Aesthetic下降来自拒绝状态。任务选择不保证一般质量，而masked训练跳过拒绝状态也不保证保留其原动作；这两层问题需要分别检验。

[Phase55完整数值报告](experiments/local_expert_v55/report.html) · [完成审计与解释](experiments/local_expert_v55_followup/SUMMARY.md) · [历史半幅草案（已落实为Phase56）](experiments/local_expert_v55_followup/NEXT.md) · [真实账务](experiments/local_expert_v55/accounting.json)。三个作业均COMPLETED，共7867/16200 L40秒，1536新图、7680质量值；confirmation/final封存。只有128个独立dev题，不能把三池当成384题；三次run的区间仍探索性。Phase55完成回调已接收；当轮每小时定时器在首次计划提醒前自动停止，周期唤醒0次。Phase56独立预算已结算，不复用旧余额或周期定时器。旧mentor展示是第一池完成时的封存快照，完整Phase55以本报告为准。

[Phase54完整报告](experiments/local_expert_v54_cpu_recovery/report.html) · [解释与局限](experiments/local_expert_v54_followup/SUMMARY.md) · [下一轮前瞻计划](experiments/local_expert_v54_followup/NEXT.md) · [Phase53报告](experiments/local_expert_v53_resume/report.html) · [Phase52报告](experiments/local_expert_v52/report.html) · [早期历史](docs/early_progress_v1_v20.md)

下文保留源码来源、原版入口及早期工程验收说明。小规模运行验证不等于论文数值复现；历史各轮的小样本结果不自动累积成稳定方法证据。

独立于旧实验代码的新框架：**固定上游源码复现 OPD/OPSD；在独立、可审计的局部介入引擎中实现“经终点验证的专家修正”。** 旧 CA-OPD、Q-Core/Q-OPSD 训练代码没有复制进来。已有本地模型、奖励权重和经 hash 验证的数据作为只读资产复用。

第一阶段实验已完成：64-prompt/1920 图冻结诊断、原版及本地训练/续训/评测 smoke、生成图复核、三档学习率的一次更新检查。见 [local_expert_v1 状态与证据](experiments/local_expert_v1/STATUS.md)、[机器可读结果与哈希](experiments/local_expert_v1/results.json) 和 [完整事件记录](experiments/local_expert_v1/ledger.jsonl)。这些结果尚不构成论文数值复现或新方法有效性证明。

## 源码与复现边界

| 源码 | 固定 commit | 来源 |
|---|---|---|
| DiffusionOPD | `0c85bf0193a6da24bb1f72152b612d353a154999` | [ali-vilab 作者仓库](https://github.com/ali-vilab/DiffusionOPD) |
| DiffusionOPSD | `9c4c48cdfe5c3fa712d32957345d75fae9295696` | [项目页](https://diffusionopsd.github.io/)链接的 [worldbench 仓库](https://github.com/worldbench/DiffusionOPSD)，README 自称 external implementation |

两者都是重新联网 clone 的仓库。`upstream/` 保持干净；每个 run 从固定 commit 做独立源码快照，隔离 `PYTHONPATH`，不受旧 fork 或后续编辑影响。`upstream.lock.json` 可用于重新获取同一版本。

**已完成源码/config 一致性检查、45 项测试、真实 SD3.5 冻结干预 smoke；原版 MOPD/OPSD 均完成两轮 GPU 训练、保存和重载评测。新方法及教师轮换对照完成两轮训练、真实续训、raw/EMA 的 flow/DPM2 10 步统一评测。第二阶段另完成四臂各 12 次更新，效果与限制见阶段记录。完整论文数值复现尚未完成。** `native` 入口直接调用上游程序，不重新写一份“类似 baseline”。OPSD 公开仓库本身的来源性质及可复现范围如上，不将它称为未经确认的作者原始训练代码。

原版 OPD 存在两个运行问题：`while True` 不读取 `num_epochs`，且保存函数没有调用。默认 managed 模式只对 run 快照添加 epoch 上限、checkpoint hook 和必需的 OCR 权重路径绑定；完整 diff 写入 `runtime_patches.json`。损失、采样、路由、optimizer 和 EMA 原代码保持不变。`--literal` 保留原始无限循环，**不会定期保存**。OPD 原入口没有实现完整 resume，本框架不会把 `resume_from` 当成成功续训；新方法引擎支持完整续训。

## 文件导航

| 文件 | 作用 |
|---|---|
| `python/opdlab/native.py` | 固定上游、配置差异审计、原版训练/评测适配 |
| `python/opdlab/method.py` | 单步候选、逐约束保护、终端选择、OPD loss |
| `python/opdlab/engine.py` | 冻结 behavior、采集/fitting、Adam/EMA/RNG、分布式训练 |
| `python/opdlab/backends.py` | 真实 SD3.5/PEFT 后端及 CPU toy |
| `python/opdlab/verifiers.py` | OCR exact match、GenEval 子句、可选 aesthetic guard |
| `python/opdlab/evaluate.py` | 每图固定噪声、统一协议、配对比较、机制诊断 |
| `python/opdlab/diagnostic.py` / `report.py` | 固定查询网格、逐候选图片证据、断点检查、prompt 聚类统计与复核页 |
| `python/opdlab/training_queries.py` | 按任务均衡且逐 pass 覆盖全部 prompt 的共同训练清单 |
| `python/opdlab/audit_ocr.py` | 不参与训练的第二 OCR，记录 executable/model/image 哈希 |
| `python/opdlab/vlm_audit.py` / `relation_audit.py` | 独立 Qwen 审查与明确关系接口对照，保留原始回答和 unknown，不作为训练奖励 |
| `python/opdlab/audit_metrics.py` | 区分未知标签、拒答与误判，报告覆盖率和含拒答的召回率 |
| `configs/` | 可读配置；运行时补齐默认值并保存完整 resolved config |
| `docs/method.md` | 方法定义、控制臂、已实现和未实现边界 |
| `tests/` | loss/选择器/真实小型 SD3+PEFT/续训/评测协议测试 |

## 当前机器

在本目录执行以下命令。`site.local.json` 已绑定本机权重和原 OPD Python；`site.example.json` 是可迁移模板。配置里路径相对**执行命令时的工作目录**解析，resolved config 改为绝对路径。

```bash
cd /midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/opd_lab
export OPDLAB_PY=../../conda_envs/envs/diffusionopd/bin/python
export PYTHONPATH="$PWD/python"
"$OPDLAB_PY" -m opdlab doctor --site site.local.json --out docs/doctor.local.json
"$OPDLAB_PY" -m pytest -q
```

不要 source 旧实验的 `env.sh`。launcher 会过滤旧算法环境变量，并优先加载 run 内上游源码。资产默认对小文件做 SHA256，对大权重记录文件大小/mtime，**这不是完整大权重内容 hash**。正式冻结实验可用 `prepare --hash-weights` / `doctor --hash-weights`。

OPD 上游固定 diffusers 0.33.1、transformers 4.40.0、PEFT 0.10.0；OPSD 上游要求较新版本。两套环境不能混为一套。当前机器的 OPSD 使用本目录独立 `.venv-opsd` 覆盖环境，复用原环境的 torch/CUDA 和奖励依赖，覆盖的包只写入新目录；详见 [环境说明](docs/environments.md)。这不是完整重装的隔离环境，实际 package versions 每次记录。

## 原版 OPD / OPSD

先准备，再执行。prepare 不分配 GPU、不开始训练；run 目录不能覆盖。

下面的 8-GPU 命令适用于 8 卡单机。当前集群每节点最多 4 卡：prepare 增加 `--nodes 2`，使用 `scripts/submit_slurm.py --nodes 2 --gres gpu:4` 启动每节点一个 launcher。总 world size 仍为 8。两节点 rank/资源匹配测试和 Slurm test-only 已通过；实际跨节点 GPU 训练尚未验证，已准备的完整 recipe 计划见 campaign。

```bash
# MOPD；也支持上游 sopd_pickscore / sopd_ocr / sopd_geneval preset
"$OPDLAB_PY" -m opdlab native-prepare \
  --upstream diffusionopd --preset mopd --site site.local.json \
  --world-size 8 --run runs/native-opd-main --set num_epochs=300

# 在已分配的 8-GPU 单机作业中执行
"$OPDLAB_PY" -m opdlab native-run --run runs/native-opd-main

# OPSD public open3，保留上游默认 300 updates
.venv-opsd/bin/python -m opdlab native-prepare \
  --upstream diffusionopsd --preset sd35_open3 --site site.opsd.local.json \
  --world-size 8 --run runs/native-opsd-main
.venv-opsd/bin/python -m opdlab native-run --run runs/native-opsd-main
```

检查 `config.audit.json`：包含上游 defaults、resolved 配置、每项变化及路径绑定/显式 override 分类。除权重、输出路径和明确 `--set` 外不改原 recipe。OPSD Pick-a-Pic train 必须匹配新上游 recipe 的 SHA256/25415 行，test 从新上游 DrawBench 复制；不信任旧 test.txt。

`--set KEY=JSON` 使用 JSON 值，例如 `--set train.learning_rate=0.0003`。每一项非路径修改都属于新的实验变体，不能仍称为无改动的论文配置。改变 world size 也会改变有效 batch/更新条件。

原版 native 评测也保留：

```bash
# OPD 的 --checkpoint 是包含 lora/ 的 checkpoint 根目录
"$OPDLAB_PY" -m opdlab native-eval --run runs/native-opd-main \
  --checkpoint /absolute/path/to/checkpoint-300 --dataset ocr \
  --out runs/native-opd-ocr-eval

# OPSD 的 --checkpoint 直接是 LoRA 目录
.venv-opsd/bin/python -m opdlab native-eval --run runs/native-opsd-main \
  --checkpoint /absolute/path/to/lora --dataset drawbench \
  --out runs/native-opsd-drawbench-eval
```

默认只生成精确的 `plan.json`，加 `--execute` 在 GPU 上执行。native 评测保留上游批处理/随机数习惯，**不是**下述统一配对协议。不要把其奖励数值与新 OCR exact match 混用。

## 新方法与控制臂

`data/demo/joint.jsonl` 仅有三个用于展示 schema 的样例，**不是训练集或 benchmark**。正式实验先替换成足够多的 train/validation/test JSONL；每行包含 id、prompt、split、group、constraints。loader 拒绝重复 id、同 prompt 或 group 跨 split。`group` 应由数据制作方编码模板族/文字词表/物体组合隔离，代码无法自动证明语义没有泄漏。

```bash
"$OPDLAB_PY" -m opdlab prepare --config configs/verified_sd35.json \
  --site site.local.json --run runs/verified-pilot --world-size 1 \
  --set 'dataset="/absolute/path/to/joint_constraints.jsonl"'

# 建议先用独立 run 做不更新模型的干预诊断
"$OPDLAB_PY" -m opdlab run --run runs/verified-pilot --diagnostic
"$OPDLAB_PY" -m opdlab analyze --run runs/verified-pilot --out runs/verified-pilot/analysis.json

# 训练使用另一个新目录，从相同初始配置准备
"$OPDLAB_PY" -m opdlab prepare --config configs/verified_sd35.json \
  --site site.local.json --run runs/verified-train --world-size 8 \
  --set 'dataset="/absolute/path/to/joint_constraints.jsonl"'
"$OPDLAB_PY" -m opdlab run --run runs/verified-train
```

控制臂使用同一配置另建目录，只改 `--set 'method="fixed"'`、`random`、`disagreement` 或 `single_query`；按需要改 `fixed_teacher`、`fixed_alpha`、`accept_probability`、`disagreement_min`。single_query 是固定教师单查询对照的名称，未冒充 DanceOPD 全实现。所有控制臂共用同一组候选查询，终端诊断成本也计入。

另有 `round_robin`：按全局查询序号轮换教师。冻结干预诊断使用 `make-queries` 显式生成 prompt × seed × time 网格，并在 config 中指定 `diagnostic_manifest`。同一个 prompt/seed 的 noop 终点必须跨介入位置逐像素一致；每个候选图片和评分单独保存，只有完整且有效的诊断可以运行 `report-diagnostic --run ... --out ...`。详见 campaign 的冻结配置和查询清单。

`verified_fixed` 只对指定的固定教师/alpha 作验证筛选；`verified` 扩展到全部候选；`protect_satisfied=False` 单独关闭已满足约束保护。第二阶段四臂共用 `training_manifest`，固定每个查询的 prompt、初始噪声与时间位置，按任务均衡采样并在每个完整 pass 覆盖数据。验证该清单时不允许修改 data/seed/world size/budget 后继续使用旧清单。

原版入口可用 `native-prepare --smoke-examples 4` 明确缩小每个 train/test split；只改 run 内数据快照并保存变换前后 hash。OPSD 单卡 smoke 还需要 `--smoke-recipe-world-size 8` 以及全部 batch/group 的显式覆盖，因为上游默认 48×24 分组配置无法在单卡构造。此选项不能用于未标记的正式复现。

默认从原 backbone + 新 LoRA 开始，**没有**从发布 Student 或旧 Q checkpoint 接着训练。若显式设置 `initial_adapter`，对照必须从同一个 checkpoint 开始。

## 保存和续训

```bash
"$OPDLAB_PY" -m opdlab run --run runs/verified-train --stop-after 10
"$OPDLAB_PY" -m opdlab run --run runs/verified-train \
  --resume runs/verified-train/checkpoints/epoch-000010
```

checkpoint 有 `training.pt`（学生/Adam/GradScaler/EMA/outer epoch/optimizer step）、每个 rank 的 RNG，以及分别标记的 `raw_lora/`、`ema_lora/`。续训要求配置、数据、源码、资产和 world size 一致；不接受只加载 LoRA 冒充完整 resume。恢复边界是完整 outer epoch。崩溃后超出 checkpoint 的部分日志需先作为失败尝试归档，不能混入下一次运行。

## 统一评测

在包含所有方法共用测试数据的 prepared local-engine run 上评价任意 base/LoRA。

```bash
"$OPDLAB_PY" -m opdlab eval --run runs/verified-train --out runs/eval-base-flow40 \
  --adapter-kind base --sampler flow --steps 40 --seeds 42 43
"$OPDLAB_PY" -m opdlab eval --run runs/verified-train --out runs/eval-verified-flow40 \
  --adapter runs/verified-train/checkpoints/epoch-000100/raw_lora \
  --adapter-kind raw --sampler flow --steps 40 --seeds 42 43
"$OPDLAB_PY" -m opdlab compare runs/eval-base-flow40 runs/eval-verified-flow40 \
  --out runs/comparison-flow40.json
```

同一 sample id/replicate 的初始 FP32 噪声在 CPU 上独立生成，不受 batch/rank 划分影响。协议记录 sampler、steps、CFG、分辨率、prompt 长度、精度、reward 定义、权重身份、环境和 adapter 类型。比较器拒绝不匹配协议、不完整结果和 raw/EMA 混比；base 可与 raw 或 EMA 各自比较。外部未标记 LoRA 必须写 `--adapter-kind upstream_unknown`，不能猜测其平均方式。

跨 OPD/OPSD adapter 统一评测时，为每个实验臂都加 `--python .venv-opsd/bin/python`，使用同一套较新 PEFT 环境兼容新旧导出格式；不要通过删除新 LoRA 配置字段使旧库勉强加载。实际解释器和版本保存于 `environment.json`，包版本差异会使比较被拒绝。

置信区间按 prompt 聚类：同 prompt 的多个 seed 先平均，再做 paired bootstrap。两三个 demo prompts 不能支持统计结论。跨 sampler/steps 的实验应在**每一格协议内**计算“方法 − 相同协议 base”，再横向看增益，不能直接比较 flow40 的方法与 dpm2-10 的 base。

## CPU 全链路 smoke

```bash
"$OPDLAB_PY" -m opdlab prepare --config configs/toy.json --run runs/toy-demo
"$OPDLAB_PY" -m opdlab run --run runs/toy-demo --stop-after 1
"$OPDLAB_PY" -m opdlab run --run runs/toy-demo --resume runs/toy-demo/checkpoints/epoch-000001
"$OPDLAB_PY" -m opdlab eval --run runs/toy-demo --out runs/toy-base --steps 4
"$OPDLAB_PY" -m opdlab eval --run runs/toy-demo --out runs/toy-trained --steps 4 \
  --adapter runs/toy-demo/checkpoints/epoch-000003/raw_lora --adapter-kind raw
"$OPDLAB_PY" -m opdlab compare runs/toy-base runs/toy-trained
```

toy 验证工程契约，不预测真实图像方法效果。真实 GPU 小规模链路已验证；正式多约束 benchmark、预算匹配训练和完整泛化矩阵仍待完成。
