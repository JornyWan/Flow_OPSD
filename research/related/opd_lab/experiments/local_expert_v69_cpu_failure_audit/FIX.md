# Phase69质量评分元数据错误：独立CPU修复记录

Slurm1065085于2026-09-30 03:26:28 EDT FAILED 1:0；实际1296/4500 L40S秒。
不是抢占。102实际选中状态生成、204原PNG重放、510任务评分及102局部误差完成，
进入PickScore质量token审计时因KeyError('cohort')停止，全部五质量均未产生。

## 根因与修复

新诊断的generate函数写出了id/prompt/word，却漏写standard_metrics.token_audit要求的cohort。
这是本轮新增诊断代码的接口错误，由我负责；原18项CPU测试没有覆盖生产者→真实消费接口。
不影响已完成Phase68训练/开发报告，不能据此判定算法有效或无效。

活跃源码加入统一quality_metadata生产函数；prepare在预留GPU前检查全部源记录，
score在加载任何评分模型前检查契约。原评分器、阈值、预测与冻结Phase69均未改。
新增回归调用真实token_audit，覆盖五种metric路径；另验证坏元数据先于模型加载失败。
使用stub tokenizer只检查元数据兼容性，不声称检验了真实分词截断或质量推理。

本目录保存独立端点和评分副本，共1020行，每行只增加cohort，逐行旧/新digest在
metadata_patch.json。任务标签、图片路径与sha未变，未重推理。原缺字段错误已用原
510端点复现；修复副本通过同一消费接口。副本不是补齐的质量结果。

## 已核验与尚未核验

52项CPU测试通过（含10项独立任务审计测试）；完整日志tests.txt。
542冻结输入、204PNG精确重放、510文件hash/任务绑定、102局部误差绑定均通过。
Phase68的报告和收尾manifest共7文件复核不变；本目录sources.json保存失败档案指纹。
所有原始任务结果均报告joint/OCR/structure/CER/unknown；区间按35个原prompt聚类。
跨臂不是同状态对照；final_suffix共用历史前缀，不等于新模型完整部署。

质量依然0/2550：PickScore/26、CLIP、Aesthetic、HPSv2.1、ImageReward全部缺失。
新修复仅获CPU验收，尚无修复后的真实GPU质量验收。不产生完整实验通过或质量保持结论。
不覆盖原失败文件，不自动GPU重试/重训。原请求4500秒不回收，本轮新增GPU为0。
