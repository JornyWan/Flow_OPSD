# Phase66运行交接

本轮从纯SD3.5-M加同一新LoRA开始，不读取官方Student作为训练起点。
首先检查status.json、四臂submission/intent、真实Slurm；勿重复投递。
四臂均已提交：verified1064733/native1064737/all1064741运行，random1064749
等待afterok:1064733。首次排队估计超过1小时曾暂缓第四臂；原冻结提交器复查后
队列变为允许，已只补第四臂。科学协议与21600秒总预算未变，全部预算已登记。
不要再投递GPU；验证失败时随机依赖将取消，失败预算不回收。
每次提交均比较四partition，sablab其他项目合计5GPU，未占用sablab名额。

全部完成后在独立CPU目录运行：
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/report_clean_base_study.py --scope experiments/local_expert_v66 --output experiments/local_expert_v66_cpu_report
报告器强制四个作业COMPLETED、5×128配对完整与3200质量值。技术失败不重试；
CPU报告错误另起修复目录，不能改已完成训练/评测档案。若统计结论不正，不换seed或挑RAW/EMA。

固定看verified−native joint，并结合verified−random/all和五质量、每族任务、
接受率、有效更新次数、完整训练和评价成本。三本地臂是相同查询日程，原生保留自身
sampler，监督时间点更多；不是同算力。仅16次更新机会、64训练、一个训练seed，
如原生本身尚未提升base，先考虑训练不足，不宣判OPD无效。正结果只支持另立512规模
与多seed协议；确认/最终集仍封存。当前不自动新加实验或定时器。
