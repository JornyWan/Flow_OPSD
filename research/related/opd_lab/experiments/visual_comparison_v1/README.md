# 方法明示的视觉对比

打开 compare.html。原生512px PNG全部内嵌，可下载后离线使用。默认左右对比，支持四方法同排、切换左侧对照、按子集浏览及点击放大。

Phase20包含全部32组新seed完整生成，baseline为共同joint128起点，后三组为128次更新后的学生。未把单步替换结果当作完整生成；完整生成是原实验的辅助路径，此页不改变失败的单步主检验。Phase21包含全部12组确认结果，展示直接介入后续生成；oracle尚未蒸馏。两个固定配置仅作事后直观对照，不替换原先锁定的noop对照。两阶段起点不同，不能跨标签页作数值比较。

本页方法公开，不承担独立盲审功能；没有改原图、标签或冻结报告，也没有新增生成/训练。输入、配对和PNG校验见manifest.json。

重建命令（使用新的输出目录）：

```bash
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/build_visual_comparison.py --output experiments/visual_comparison_v2
```
