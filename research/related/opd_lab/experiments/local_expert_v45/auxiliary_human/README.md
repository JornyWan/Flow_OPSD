# Phase44 人工诊断导入

这是已有24组/48图的分析支持工作，不是Phase45，也不是新模型实验。
当前状态以results.json为准。readiness模式没有生成任何人工答案文件。
约束的true/false/null与未回答严格分开；可只导入部分答案。
只导入导出的勾选答案，备注与抄录不自动补标签；full_visibility不属于本轮rubric。
按四个定向抽样层分别报告，不计算总体胜率/显著性，不改变Phase44扩展决定。
每次导入必须使用新目录。先填写盲评，后看解码后的results.json。

从opd_lab目录运行（若输出目录已存在，改用下一个编号）：

```bash
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/analyze_capability_human_audit.py --answers human_rating/phase44-human-diagnostic-v1-answers.json --output experiments/local_expert_v44_human_answers_001
```

报告可离线打开。来源答案逐字节保存；未知、漏答和历史标签不会被改写。
缺少真实人工答案时，只能确认导入准备就绪，不能得出新的视觉效果结论。
