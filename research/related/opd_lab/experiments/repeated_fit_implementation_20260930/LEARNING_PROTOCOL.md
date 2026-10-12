# 下一轮研究合同：固定实际LR，每批4次拟合

合同版本：repeated-fit-v1，2026-09-30。
研究问题、终点与下面的判定规则在新GPU结果出现前固定。执行入口、数据/代码指纹及
GPU预留尚未冻结，当前未提交。这份合同不改变Phase68/69/70的结果或门槛。

## 假设和比较

问题：更充分拟合已经验证有收益的局部动作，能否同时提高训练修复兑现和单个学生
完整生成的联合成功？验证选择是否仍提供超过同量随机的增益？

新verified4与random4使用同一纯SD3.5-M＋fresh LoRA、空AdamW、原64训练prompt、原
1024条prompt/noise/time日程，16外轮。每非空轮4次fit，最多64次成功optimizer更新。
random逐批匹配新verified的任务族/模板/接受数量；每个fit实际LR一致，保持
3e-4×接受数/64。分母仍为全部64查询。空批跳过optimizer/EMA，不伪造零LR更新。

仅改变fitting_epochs=1→4；新增record_fit_blocks是观测。loss、教师、CFG、时间分布、
优化器、EMA、rank32/alpha64、512px、text128、Flow10及严格unknown处理保持原协议。
每批只采集/选择一次，4次fit共享原state/target/context；拟合块之间正常刷新behavior。

只读参照Phase68 verified1/random1、纯底座，以及Phase66 native/all。1/4fit在线轨迹与
后续目标集合可能不同，不能称同一固定目标的纯回归消融。旧native/all实际LR不同，
不能作为筛选因果对照；新两臂训练更新量匹配但选择/评分成本不等，不称总算力相同。

## 固定终点

主指标：全部128已暴露dev、共同prompt/noise、最终epoch16 RAW、单adapter完整Flow10的
verified4−random4 joint。每臂128个终点。同步报告OCR exact、结构、CER、unknown和全部
PickScore/26、CLIP、Aesthetic、HPSv2.1、ImageReward；不挑checkpoint、seed或最好子集。

机制：每臂最多64个真实选中训练状态，使用原choose_queries的固定哈希规则，与结果
无关。端点依次为behavior、teacher、post_fit_block_local、final_local、final_suffix。
前两者必须重放原PNG hash；post_fit_block_local使用该批4次更新全部完成后的真实权重
和该批冻结behavior后缀，不叫immediate/first_step。最终后缀仍有历史前缀，不是部署。
报告直接修复保留、全部任务/五质量、MSE和教师方向投影；按原prompt聚类，不跨臂
当成同状态匹配。少于64则全取，没有选中状态则明确无可测机制集合。

区间使用固定seed66066、20000次prompt聚类bootstrap，均为描述性，未多重比较校正。
单训练seed和反复使用dev仍不足以作论文确认；confirmation512/final768继续封存。

## 固定判定：这是筛选值得复验的门槛，不是论文有效性确认

只有同时满足以下三点，才支持另立独立重复实验：

1. verified4−random4与verified4−纯底座的dev joint描述区间下界都严格大于0。
2. verified4的dev joint点估计高于已列出的native和all参照；同时保留它们实际LR不同
   的限制，不把此项改称同条件原生算法胜出。
3. verified4−random4及verified4−纯底座的五质量均值差全部≥0。该零容忍均值筛选
   不等于证明质量非劣；所有质量区间、绝对值和相对native/all差异仍必须报告。

未达到则不自动扩大4fit、打开封存集或改门槛。
训练修复/MSE改善而dev失败：停止仅以更多fit解释失败，转查泛化/共享参数影响。
训练修复仍未兑现：区分目标终点敏感性与学生拟合能力，不自动升LR或再加fit。
两臂都提高但verified不胜random：说明更多拟合有用，不支持验证选择的额外贡献。
Phase54训练修复恢复而dev落后随机的历史反例始终纳入解释。

## 执行与资源边界

两个顺序依赖作业，各1×L40S、10800秒硬上限，总请求上限21600 L40S秒。新实验独立
记账，不结转Phase70或旧夜间余额。先verified，只有成功后random才能训练；每次实际
sbatch之前probe四分区，sablab跨项目allocated+pending承诺≤6，Requeue0，技术失败不重试。
若不能合理排到兼容GPU则暂缓提交；不取消其他项目作业，无新timer/agent，无agent读图。

最大新开发图256、机制新图384（原行为/教师候选复用），最大主终点评价896张、五质量
4480值，另有训练候选最多4096张。原Phase68训练含加载/核验约4953/5057秒，完整作业
5575/5676秒；缺乏拆分后的实际fit耗时，不能保证4fit在3小时内完成。硬时限不延长，
技术未完成就停止，不将不完整主比较当结果。

提交前还有独立执行工作：把新的块权重接入上述机制端点及完整报告，测试模式名称、
权重来源、评分cohort、残缺输出拒绝与判定规则；冻结执行输入和预算reservation。
CPU118项与102个真实保存状态接口检查不替代这套SD3完整执行链的验收。
