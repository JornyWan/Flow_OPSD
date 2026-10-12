# Phase64：两臂训练完成，24图已生成；评分一致性门槛失败

Slurm1064714，scu-gpu/L40，FAILED 1:0；实际290/1800 GPU秒，Requeue=0。
没有重复训练verified，没有重排或新定时器。旧Phase63实际185秒单列，两次实际合计475秒，
请求上限合计3600秒，未用预算不回收。

## 已完成

random从原发布Student/fresh Adam完成两轮32查询、64候选、2次更新；逐批选择4和1，
与verified完全匹配，两批分层参考日志逐字段一致。两臂mask有6/32槽位不同，并非同一组。
首轮初始策略下32个配对候选图SHA、A标签、完整A证据全部一致，支持采样和训练路径重放。
random两次checkpoint均381/382张量改变，有限，raw导出等于training.pt，Adam step1/2；
EMA未到间隔仍为初始，符合预期。旧verified的19项封存产物hash保持不变。

同8dev、初始/verified/random的24张完整生成图全部落盘且配对seed/prompt/hash核对通过。
源代码执行已越过单default adapter、逐图10次student调用的断言，然后才进入评分；
但旧程序未提前写generation_checks，不能称全部GPU验收已经通过。

## 停止原因及未知

冻结evaluate在第一张旧训练noop候选重评分时，`actual['A']==candidate['details']`失败。
该断言包含检测框、OCR置信度等完整浮点证据；尚未运行下一条clauses equality检查。
原程序没有在失败前保存actual，所以当前不知道是数值微差、结构差异还是语义判定变化。
不能把它当作方法失效，也不能擅自当作无害浮点问题放行。

开发B任务评分0/24，五质量0/120：这些是尚未计算，不是成功率或质量为0。
本轮没有新的joint、OCR/CER、训练修复兑现或质量结论。确认/最终集未使用。

下一步已在[独立CPU审计](../local_expert_v64_cpu_parity_audit/SUMMARY.md)补失败前取证，
保留严格门槛，并设计只对旧图做评分诊断的后续。原失败和本目录现有生成结果保留。
