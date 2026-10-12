这是用户2026-09-28在本对话再次要求夜间每30分钟/1小时/2小时自动查看所授权的每小时提醒。继续同一Phase55三训练池实验；不是新GPU预算，不授权第四池或新实验。

1. 零图片、截图、远程视觉或委派读图。只读精简数值、日志、manifest。人工评价只辅助。
2. 读取opd_related/opd_lab/AGENTS.md、experiments/CURRENT.md、local_expert_v55/progress.json、monitor_latest.json、budget.json和真实Slurm accounting。先核验manager是否活着，不重复启动，不重复提交GPU。现有manager独占三个冻结池的串行提交；总上限16200 L40秒，技术失败不自动重试。
3. 立即在experiments/overnight_20260928_v2/acknowledgements.jsonl追加本次wake_utc、接收时间和真实状态。queue返回0不等于对话完成；本行才是接收确认。
4. 若仍在运行，检查已完整归档的池结果、资源与错误，更新持续维护的CURRENT/README/docs，必要时只在独立CPU分析目录解释现有结果；不改运行中的冻结代码、不按效果提前停止或修改协议、不为了提醒而进行重复分析或额外GPU工作。
5. 全部完成时按冻结协议解释跨三池verified16−random16、训练修复兑现和全部五项质量；不能把共享128开发题当成384独立题。确认/最终集封存。多数池为负或平均≤0时停止扩大当前硬筛选配置，先提出可证伪的简洁后续；不擅自叠加机制或扩大预算。
6. 技术失败保留原档案，不自动重试GPU。CPU分析/报告错误只在新独立修复记录解决。若manager意外死亡，只审计真实作业与文件并报告，不新提交或取消任务。
7. 已完成mentor材料在presentations/2026-09-28_mentor_progress_v1，离线HTML/PDF/ZIP和来源已封存；未来结果用新版本，不能改旧manifest对应材料。材料库入口presentations/index.html。
8. 完成/失败后写本目录STOP。现有local_expert_v55_followup_queue会发送一次完成提醒；如同一结果已审计，后续提醒只记录ack并指向已有结论，不重复作业/全量审计。

此定时器每小时一次最多六次，到期停止，不自动扩展。保留旧overnight_20260928过期且0次周期唤醒的记录。
