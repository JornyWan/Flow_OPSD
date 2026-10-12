# Phase72后续检查

已有唯一作业1067893，preempt_gpu，2026年10月3日12:40:36 EDT开始；75分钟硬上限，Requeue0。先查实际sacct、submission/intent、STARTED、failure/result及evaluation/progress，不重复提交。没有新timer。本轮预算4500秒已预留，失败或未用时间不退。排队等待的独立调整见execution_queue/NOTE.md；没有改变科学协议。

完整结果只认status complete与真实COMPLETED 0:0，且128旧图摘要与A证据重放全部通过、四adapter权重不变、实际预测次数正确、320端点及1600质量齐全。启动中的权重校验不等于这些GPU检查已通过。若失败，记录精简原因，不自动重试，不解释部分病例为科学成功；CPU报告缺陷在独立目录修复，已完成档案不改。

资格门槛是OCR局部独有joint修复至少4例且相对S0净增至少2例，非训练成功或统计显著性。全joint/OCR/结构/CER/unknown和五质量、配对差及区间均报告。独有修复依赖B标签，只是训练cohort上的事后oracle机会；不能称可部署路由器。

无互补机会就不扩大OCR局部路由；有机会也只值得另立同专家信息的强OPD学习比较。后续须固定训练任务分布/实际LR/监督量、足够训练与完整学生部署，并区分官方发布学生能力参照、官方配方复现和任务适配匹配基线。不得把加专家本身当验证贡献。确认512/final768继续封存，agent零读图。
