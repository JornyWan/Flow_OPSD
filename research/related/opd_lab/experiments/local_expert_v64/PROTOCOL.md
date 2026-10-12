# Phase64：补齐在线选择的小规模执行闭环

用户2026-09-29要求继续方案并检查运行结果。本轮另立一次1×L40、最多1800 GPU秒，
不是Phase63失败后的自动重试，也不回收旧预算。Phase63旧请求1800/实际185秒单列；
两次请求合计3600秒，若本轮用满，则两次实际合计最多1985秒。不设定时器，不自动重排。

## 要回答的问题与停止规则

只验收：当前在线semantic_batch_v1能否在真实SD3上完成分层同量随机对照，并将一份
student adapter用于完整10步生成，且取得A接口parity、B任务与全部五质量。
不以8dev结果的正负决定成功，不计算显著性，不把它当论文主表或旧CFG1学习的复验。
技术失败即停止并保留部分结果，不自动扩大预算、换数据或重训。

## 复用与新增

原Phase63 verified_run只读复用：已完成32查询/64候选、2次更新、接受4和1；原FAILED
档案不改。调用已CPU准备的random目录，仍从同一原始发布Student与fresh Adam开始。
严禁对verified再次launch，也拒绝运行已有任何训练尝试标记的random。
随机只新增32查询/64候选；公共评测8个已选dev × 初始/verified/random，共24终点，
120个五质量值。跨两次作业总候选128，不能说本次又生成128。旧失败的partial random不动。

训练模型/配置/源码和开发案例全部沿用Phase63冻结输入：16train，4任务族×2模板，
每格2题；2轮，每轮16查询与1次更新；Flow10、student CFG4.5、GenEval CFG1、512px、
text128、lr3e-4。查询覆盖固定哈希的0..8步，两臂各自刷新自己的在线behavior。
随机消费verified的逐批family/template数量参考；身份/noise/time匹配，不宣称更新后状态相同。
最终只评epoch2 raw；EMA尚未更新，不选择最佳checkpoint。没有改算法或教师候选。

## 执行检查

控制器只launch(random)，随后调用原verified冻结源码中的semantic_acceptance evaluate。
不在计算节点重新prepare或调用git。用于核对源码的live_framework字段指向已冻结框架，
而非之后可变化的工作区。新控制器/所有输入/旧verified关键产物的hash在提交前冻结。
只在独立评测子进程显式传递CUBLAS_WORKSPACE_CONFIG=:4096:8，避免clean_env过滤
该变量后与评测的torch.use_deterministic_algorithms(True)冲突；不改两臂训练环境。

评测删除behavior adapter，只剩default，每图10次student调用。核对两臂监督数、
optimizer更新、mask差异、导出与training.pt相等、第一对训练候选A证据完全重放。
B主指标joint，并全报OCR exact/CER、结构strict、A未知及五项质量。冻结评测器原始
result字段保留；CPU报告可从原24条评分补结构和配对赢输，不新增GPU评分、不选子集。
训练修复兑现未单独生成，本次只能报告训练选择/更新事实与8dev效果，不能假称兑现率。

## 调度与后续

每次唯一提交前探查minilab-gpu/sablab-gpu/scu-gpu/preempt_gpu；L40兼容性不改，
sablab跨项目运行+pending承诺GPU≤6。禁requeue；一小时内无可信开始估计则留待prepared。
原始失败和完成档案不写入。confirmation/final仍封存，人工标签只辅助，全程agent读图0。
通过之后先汇总实际时间/算量/接受率和完整结果，再冻结正式共同起点原生OPD、单查询、
同量随机、验证比较；不自动提交更大实验或按这8题调参。
