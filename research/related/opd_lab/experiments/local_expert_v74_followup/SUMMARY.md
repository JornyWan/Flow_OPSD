# 完整 OPD 学习路径验收进展

2026-10-08。第一阶段已落实到可执行数值验收；25项CPU测试通过，真实SD3 GPU验收尚未执行。没有新模型效果结论，未读取图片、未使用confirmation/final、未提交GPU或设置timer。

**已证实的部分。** 在相同冻结合成输入上，直接执行锁定官方源码的loss reduction，用真实Accelerate执行27次梯度累积，覆盖9轨迹×9时间共81状态，恰好一次AdamW更新。本地等价平均loss的梯度最大绝对差1.7462298274040222e-10，更新后参数差0。教师目标detach和CFG1/4.5、全9时间的转移函数也通过CPU检查。这不证明真实SD3混合精度路径已经等价。

**发现一：EMA调用频率不同。** 官方在每个微批次后调用EMA；本地engine在成功optimizer update后调用。按官方3微批/更新，在global_step7时，epoch7末及epoch8前两个微批各执行一次EMA；step15类似。16个合成参数更新的原生EMA为12.71161556，单次调用版为7.77176476。该数字只用于证明调度差异，不是模型性能或参数变化实测。保留官方行为用于后续原生baseline；新方法若使用另一EMA日程必须单独标识。此前主要RAW结果不会因修正EMA而自动变好。

**发现二：CFG计算精度不同。** 官方安装版本Accelerate.prepare_model包装forward，将输出转FP32，然后官方compute_student_step组合CFG。现有SD3Backend.predict在autocast输出上组合CFG，最后才float。CPU构造的FP16输出[100,100.0625]、CFG4.5，分别得到100.28125与100.25；这是反例证明，不是SD3误差幅度。该差异可能影响动作/梯度，效应尚待真实GPU量化，不能称为所有退化根因。

最初的prepared验收只提取官方函数而未加Accelerate输出转换，会遗漏第二个差异。它未提交GPU。原准备包保留；修订包local_expert_v74_precision_revision1增加原生autocast+output-to-FP32包装，已有本地预测路径不改。不是通过修改原路径制造通过，也未放宽数值门槛。

**CPU准备失败与修复。** 一次CPU准备在Accelerator.unwrap_model自动发现DeepSpeed时，触发其NFS临时目录清理OSError39。单CPU无并行包装，因此改为断言单进程Conv2d后读取其state_dict，仍使用真实Accelerate累积。新增完整CPU回归检查，最终25项通过；无GPU失败或重试。

**真实GPU验收内容。** 从纯SD3.5M+fresh LoRA开始，3个固定TRAIN prompt、27实际状态，对相同初始参数分别执行原生包装路径与本地路径各1次RAW更新，比较教师目标、loss、梯度、Adam后参数以及单adapter导出后的完整10步latent重放。只做工程验收；不生成图片，不报告joint或五质量，不冒充效果实验。单L40S硬上限1800秒、无重试。

**资源。** 四分区实际probe：minilab最早预计10月10日21:23 EDT、scu为10月14日11:44 EDT、preempt为10月9日01:08 EDT。sablab当时运行6GPU、pending另4GPU；无本实验新增承诺，未干预其他项目。均未满足准备协议的一小时可启动条件，未产生submission intent。调度估计会变，后续真正提交前必须再次probe。

下一项是修订包真实GPU数值验收。通过后仍要补完整原生主循环与充分训练baseline；若不通过，先分解混合精度/CFG差异并独立设计修复，不开始短轨迹新方法。后续任务适配baseline可使用已冻结512训练prompt与128开发prompt；具体训练终点、有效batch和请求预算依据实际吞吐另行冻结，不能拿这次1更新硬件检查证明训练充分。

[CPU数值](../local_expert_v74/cpu_audit.json) · [修订验收协议](../local_expert_v74_precision_revision1/PROTOCOL.md) · [调度证据](../local_expert_v74/scheduling_20261008T204424Z.json) · [研究候选](../../docs/llm_opd_transfer_20261008.md)。
