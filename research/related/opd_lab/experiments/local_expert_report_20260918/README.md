# 局部专家蒸馏：方法、实验与证据报告

成品：[report.html](../../runs/local_expert_report_20260918/report.html)。约72 MB，直接下载这一个文件即可离线打开，不需要localhost、图片目录、CDN或外部JavaScript。大文件保存在已有的git忽略目录runs中；生成器、模板、来源索引和验证记录由git管理。

包括：

- 原始研究问题、方法公式、通用引擎与Phase20固定状态消融的区别。
- 阶段1–20的摘要、证据边界、完整STATUS原文，以及25张原有报告图。
- Phase20按单步/完整生成、子集、成功/损伤/未知切换的交互结果图。
- 人工复核覆盖、共同7组结果、11处字段分歧、12组原始整体意见。
- 三套完整案例浏览器：Phase20学生评测256个观察槽、监督采集96个、Phase19教师机会48个。400个观察槽与25张报告图去重后共有393张原始PNG；所有图片保持原始字节。
- 下一步mask消融的计算示意、待检验假设与分支决策，明确标为尚未运行。
- 汇报模式、原图放大、指标CSV与来源JSON导出、打印支持。

历史阶段1–18收录的是完整阶段记录和原有报告图，未把这些阶段生成过的全部原始候选PNG打包。当前三套最新案例集合则没有按结果筛选。助手临时标签、人评、事后oracle与真正学生结果分开表述；未知和缺失不补成失败。

来源：[manifest.json](manifest.json)。构建时重新计算全部最新计数，与冻结结果逐项核对；纳入报告的363项原Phase20归档来源哈希匹配，原始人评分析及图片身份哈希匹配。没有修改原实验数据、标签或旧档案。

浏览器验证：[browser_verification.json](browser_verification.json)。使用Chromium离线打开本地file，检查全部112组/400个观察槽可达、交互指标计数、人工未知/缺失、历史搜索、图像放大、CSV导出、桌面与390px手机布局及汇报模式。截图与打印预览保存在成品同目录。验证工具仅安装于`.tools/report-browser`，没有修改训练环境。

重新生成（从opd_lab目录）：

```bash
PYTHONPATH=python ../../conda_envs/envs/diffusionopd/bin/python scripts/build_project_report.py
.tools/report-browser/node_modules/node/bin/node scripts/check_project_report.cjs
```

生成器：[build_project_report.py](../../scripts/build_project_report.py)；模板：[project_report_template.html](../../docs/project_report_template.html)。source_commit记录构建时已有实验仓库快照，生成器和模板同时以独立文件SHA256记录在manifest中。
