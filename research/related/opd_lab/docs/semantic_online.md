# 在线终点验证接口：semantic_batch_v1

这个版本把近期A选择语义接到已有`engine.train_loop`，复用采样、loss、Adam/scaler/EMA、
checkpoint及完整student轨迹的训练后检查。它是本地方法实现，不是原生OPD训练入口。
旧`selection_protocol=legacy`行为保留，旧run与完成档案不修改。

## 配置与运行顺序

新配置显式设置`selection_protocol="semantic_batch_v1"`，仅支持单进程、一个固定教师和
`alphas=[1.0]`。可选method为`verified_fixed`、`random`、`single_query`。
概率、utility等旧选择器旋钮不参与新规则；非默认值会报错，质量阈值亦不在本版本实现中。

1. 用原`runtime.prepare`及`runtime.launch`运行verified_fixed。共享循环每轮从当前student
   刷新behavior，按冻结日程查询不同时间步，候选只复用prefix，改变动作后重新执行冻结后缀。
2. 选择在一个完整采集批次之后执行，按每个optimizer accumulation group记录分层接受数。
   完整运行结束才发布`selection_reference.json`，未完成或中断的run没有可消费的完整参考。
3. random配置指向这个参考。prepare会复制并哈希参考文件，检查原verified完成状态、来源日志、
   全部配置/模型/源码/训练数据及每个查询id、noise seed、时间步、分层和数量。
   计算节点使用`runtime.prepare_matched_random(verified_run, destination)`直接克隆已冻结的
   输入，不调用git、不复制训练后的checkpoint；原始published Student和fresh Adam不变。
   Phase63曾因计算节点没有git而在普通prepare阶段失败；该修复独立记录、未自动重提GPU。
4. random从同一个初始Student及fresh optimizer/scaler/RNG开始，但之后使用自己的在线状态。
   它从各层全部槽位按固定hash抽取相同数量教师动作，不根据自己A接受数决定监督量。
   允许随机与验证自然重叠，不强行排除验证选中的样本。

family层用声明的非OCR约束id组合；domain以record.group中的模板分层，通常比仅photo/
illustration更细。因此需要每层有足够候选，避免每层仅一个槽位导致随机和验证必然相同。
Phase63使用每层两个train样本，并报告实际mask差异；没有差异就不能检验选择效果。

## 语义与训练合同

- `ClauseEvidence`只接受bool/None。None是有效弃权，错误另存error并停止该批训练。
- `SemanticVerifier`包装原A证据实现；B、人工标签和五质量不作为选择输入。
- `decision`按record声明顺序调用原`semantic_accept`，完整修复且无损伤/unknown才能接受。
  JSON键排序不能改变reason/unknown列表的记录顺序，缺失或额外约束立即拒绝。
- 目标detach、原transition loss、所有槽位分母不变。当前shared engine仍计算零权重拒绝槽位；
  不把它称为节省所有拒绝反向计算。整个accumulation group接受数为0时不更新Adam或EMA。
- 两组每个group的接受数和是否更新匹配。随机组自己的状态/教师目标可不同；不能称后续
  所有训练状态完全一致。FP16 scaler跳过更新会停止实验，避免隐藏更新次数差异。
- checkpoint导出不得刷新behavior，真实PEFT回归已验证；每轮刷新发生在下一轮采集前。
- 新接口继承当前local engine的EMA调度，不声称与官方OPD的全部训练配方等价。

## 评测与部署边界

通用engine的postfit给出A局部动作和完整student轨迹的训练诊断；它不是开发/独立评测结果。
旧`evaluate`的JointVerifier标签语义不同，新协议显式阻止误用该入口。
真实模型小验收使用`semantic_acceptance`：删除behavior adapter后只留default生成全轨迹，
再复用当前TaskScorers的A/B和五质量接口。Phase63的小样本汇总不是正式论文评测器或主表。

正式比较还需锁定完整128dev、正常OPD强对照、总成本、最终raw/EMA规则、权重大文件全哈希。
confirmation/final封存；不根据小样本开发成绩选择checkpoint、修改验证器或扩大预算。

## 已完成验证

Phase62：105项CPU测试通过；历史3池192条A决策逐字段重放一致，接受数12/11/11。
覆盖label语义、外部count reference、来源/配置/数量篡改、empty Adam/EMA、冻结边界、
full-student路由、原流程回归、完整resume和checkpoint。
GPU端真实模型/A评分共存、运行成本、部署及任务/质量验收另见Phase63记录。
Phase63目前仅verified两轮训练完成；random准备失败，完整部署和任务/五质量尚未运行。
独立CPU执行修复52项检查通过，真实已训练源码的随机目录亦在禁用PATH后准备成功，未提交。
