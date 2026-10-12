# Phase62完成：在线选择接口已接入，CPU验收通过

本轮实际实现`semantic_batch_v1`，没有复制新的训练器。共105项CPU检查通过：
102项综合回归加3项GPU验收准备检查。历史Phase55三池192条数值证据与旧决策逐字段一致，
接受数12/11/11；没有读图、模型调用或新增任务/质量预测。

## 变化

- 原engine可使用保留bool/None/error的A证据；unknown弃权、技术错误停止训练，B不参与选择。
- 随机对照读取验证组完整运行后冻结的逐optimizer批次、分层接受数。两组在线状态可以不同，
  监督数量不会因各自验证接受数不同而悄悄变化。
- 固定query身份/日程/配置/源码/资产，并核对参考与verified原始完成记录；prepare即阻止
  换LR、缺少批次、改数量、改prompt/seed/time或用未完成参考。参考复制并纳入manifest。
- 沿用原采样/拟合/保存/续训：detach目标，重算后缀；空批次不动已有Adam动量/EMA；
  完整resume逐位一致。训练后诊断执行整条student轨迹，旧legacy默认保留。
- 误用旧JointVerifier评测会明确报错，不静默改变标签。真实A/B和五质量验收由Phase63负责。

CPU检查中修正了一个序列化细节：旧A JSON键按字母排序，历史decision按声明顺序排列unknown。
现在显式按声明顺序核对并计算；接受结果没有变化。过程记录见DEVELOPMENT_NOTES.md。

## 边界

新接口目前仅单进程、单固定教师、全幅动作；没有新保持loss、CFG搜索或质量约束。
family/template层比简单photo/illustration域更细，需要足够候选才有非平凡随机对照。
这些是实现检查，不是新方法收益。强原生OPD、完整128dev、总成本匹配仍需独立正式协议。
旧Phase60停止结论不变，confirmation/final未触及。

[接口文档](../../docs/semantic_online.md) · [历史数值重放](historical_parity.json) ·
[下一步真实模型验收](../local_expert_v63/PROTOCOL.md)
