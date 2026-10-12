# 2026-09-18 原始 idea 与 Phase20 实现复审

本目录是审计与事后敏感性分析，不是 Phase21 训练实验。新增 GPU 作业、训练更新、生成图片、人工标签均为 0。原 Phase20 冻结档案未修改。

- [完整判断与下一步](../../docs/research_reassessment_20260918.md)
- [逐例复算、计数与输入 hash](selection_audit.json)
- [审计来源清单](manifest.json)
- [已有定向测试执行记录](tests.txt)
- [复算脚本](../../scripts/audit_phase20_selection.py)

核心新观察：在 48 个冻结采集状态中，noop / 直接教师 / 原筛选的完整场景数为 16 / 25 / 24，已知损伤为 0 / 4 / 0。接受规则未纳入“仅改善完整可见性”；事后纳入 case2、39 后变为 26/48、0 已知损伤。此反事实不代表训练后模型、独立数据或新的人工确认。

复算方式（在 opd_lab 目录，输出必须是未存在的路径）：

```bash
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/audit_phase20_selection.py --output /tmp/phase20-selection-audit-new.json
```

脚本拒绝覆盖目标文件，核对冻结观察和配对 key 的 hash，核对每个现行接受决策与目标，并从保存 schedule 重新计算拟合数量。源码和输入 hash 写入输出。它不改训练代码或旧标签，也不将 unknown 变为 false。

本次已有测试命令：

```bash
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python -m pytest -q tests/test_method.py tests/test_sd3_contract.py tests/test_native.py tests/test_acceptance_learning.py tests/test_acceptance_collection.py tests/test_supervision_study.py tests/test_supervision_review.py tests/test_prospective_review.py
```

结果：41 passed in 18.11s。这是本次定向 CPU 检查，不是声称重新跑过全部历史 GPU 实验或完成论文数值复现。`tests.txt` 是该已完成调用的记录副本，不是第二次测试运行。
