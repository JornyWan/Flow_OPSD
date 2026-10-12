# Phase61完成：共同起点原生OPD工程验收通过

Slurm1064556，scu-gpu / ai-gpu19 / L40，COMPLETED，退出0:0。
实际151/1800 L40秒（0.04194 GPU小时），单作业，无重试/定时器。
四分区实时探测已保存；scu可立即启动，minilab预计需更久，sablab无兼容L40，未选择抢占。

- 官方固定源码SOPD-GenEval适配，从共同发布Student、fresh Adam开始；2轮、2次更新。
- 原生训练进程97.30秒，部署加载与检查49.11秒。
- raw共382个张量，381个更新；有限值、key/shape一致。
- EMA与初始逐位一致，符合更新间隔8尚未触发的预声明预期。
- optimizer所有状态step2；raw/EMA/scaler/EMA状态/RNG已保存。
- 部署仅一个default adapter，10步全部student；没有teacher/anchor/behavior adapter。
- raw保存重载一致，重复轨迹逐位一致；EMA与初始轨迹一致；raw终点确实改变。

**没有新的joint或五质量成绩。** 这是工程验收，未解码新的评测图；4条训练prompt上的latent
轨迹仅检查执行。原生训练自身采样6条轨迹不算效果评测。未经稳态benchmark的约1秒latent
轨迹计时不能与旧实验总GPU秒直接比较。也不能据此声称本方法已实现单adapter部署：本次验收
的是原生OPD。原生snapshot与初始LoRA已记录，任务大权重完整指纹和正式效果协议仍待补。

文件可见性：作业完成后登录节点一度返回acceptance.json不存在；目录刷新后读取到完整结果。
这是NFS目录可见性延迟，没有GPU重试或修改输出。报告根据实际JSON与Slurm账务编制。

[协议](PROTOCOL.md) · [数值](results.json) · [下一步接口与比较计划](NEXT.md) ·
[方法实现审计](../implementation_review_20260929/SUMMARY.md)
