# Phase54数值分析完成（CPU收尾）

selection_advantage_not_established

## 训练内诊断（64）

- noop: joint=14/64
- random_direct: joint=16/64
- random4: joint=16/64
- teacher_local: joint=15/64
- verified4: joint=15/64
- verified_direct: joint=18/64
- random16: joint=21/64
- verified16: joint=22/64

## 开发集（主要比较）（128）

- fixed4: joint=52/128
- published_anchor: joint=38/128
- random4: joint=42/128
- teacher_local: joint=50/128
- verified4: joint=43/128
- random16: joint=55/128
- verified16: joint=44/128

GPU 2419秒；完整表与限制见[report.html](report.html)。直接读图0。

GPU作业超时（TIMEOUT，2419/2400秒）。所有预测和评分已落盘，本报告由独立CPU完整性审计后补齐；新增GPU费用0，没有补生成、补评分或更换checkpoint。
