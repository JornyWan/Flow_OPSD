# 08:00：池内比较匹配，科学收益仍未建立

没有新增GPU或预测。原Phase60停止规则不变。独立CPU记录核对Phase59/60和旧Phase55
pool2共6个臂、1152个既有终点、5760项质量绑定；22项负例/完整性测试通过。
[简洁HTML](report.html) · [机器结果](comparison_results.json) · [下一步](NEXT.md)。

| 比较 | 检查结论 | 能解释什么 |
|---|---|---|
| Phase59验证 vs 随机 | 同起点/状态/训练量/评测/部署，仅名单及选择方式不同 | 可评估这套固定局部流程的选择贡献 |
| Phase60验证 vs 随机 | 同上 | 第二训练池的机制重复，仍只有128独立dev |
| 池2 CFG1 vs 旧CFG4.5 | 不能标为纯幅度、严格同训练量 | 教师CFG、目标、名单/分层和数量11→10一起变；有效的整套配置比较 |
| 池2 vs 池1 | 不能标为同训练条件 | 合理的独立训练池重复；数据/种子/初始RNG本来不同，非实验错误 |
| 原生默认fresh LoRA vs 发布Student续训 | 不同初始化 | 默认配置不是共同起点强对照，也不是已跑出的结果 |

“匹配通过”不是显著性或方法成功。实际同128dev的新验证/随机为57/53、52/50，
区间均跨0。池2验证相对旧60降为52，门槛失败；五质量相对旧验证均值改善，
相对初始学生四项仍下降。完整均值包含OCR/CER/unknown、训练和开发五质量，
见[aggregates.json](aggregates.json)，全部区间仍用原报告，不另算新的挑选统计。

## 检查依据与边界

- 不只比较路径或配置：读取student/teacher tensor digest、冻结初始历史哈希和实际恢复验收，
  actual updates/active queries/denominator/forward/backward日志；原始训练状态来自归档blob哈希。
- 检查同prompt、约束、noise/replicate、完整train64+dev128、重复/缺失/非有限值、图片内容哈希绑定。
  图片哈希从已审计JSON读取，没有打开图片。A unknown保留，错误不能作为成功值通过。
- 重新构造原family×domain哈希排序的随机名单。Phase59参考用40质量值未混入1920 endpoint指标。
- 当前部署实际两adapter、query1 learner/其余anchor，专家和verifier不在推理路径。
  参数总量和推理延迟未实测；同拟合查询量不能自动升级为同总GPU预算。
- scorer源码/环境/已有权重清单相符。**旧任务权重中部分大文件只有size/mtime，而非SHA**；
  不把一致的metadata写成完整内容证明。共享池内评分流程及parity支持当前范围；正式重跑须补全指纹。
- 本检查依赖既有tensor/GPU审计，不重新加载全部checkpoint或回放模型；输入读取68份JSON/源码，
  全部重新哈希未变。旧完成档案、Phase59 GPU失败与独立CPU恢复的事实都保留。

初版CPU提取器遇到旧result不列初始优化器快照的格式差异，未产生统计；在
[独立修复记录](../comparison_contracts_20260929_cpu_fix/REPAIR.md)中改为引用后续CFG实验冻结的
原快照SHA与实际恢复证明。没有修改旧档案，没有用路径相同推断状态相同。

## 对论文的实际帮助

现在更清楚：目前选择差不稳定，不能通过“其实对照不匹配”轻易解释掉；相同loss也不能
让这套局部流程等同于官方OPD。正式问题应是：在共同强学生、数据和总成本下，终点验证
相对认真训练的同专家OPD，是否提供额外任务收益且保住一般质量？

已有512训练视图与native两epoch CPU预检查，但没有正式训练、预算和checkpoint规则。
[正式对照草案](formal_comparison_draft.json)自动标为draft_not_submittable，8个缺口未齐，
没有生成sbatch或申请资源。这个CPU准备不是新的方法贡献或正式基线结果。
