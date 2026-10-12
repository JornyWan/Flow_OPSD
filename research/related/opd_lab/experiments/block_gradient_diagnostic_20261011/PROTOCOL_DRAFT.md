# 两步训练机制诊断草案：不训练、不生成图片、尚未GPU执行

依据：Phase79完成/审计，EMA joint50对原生49，16新增15损失；关系收益被颜色/多约束损失抵消，RAW43对45。EMA五质量差区间均跨0，训练耗时1.875倍。现有loss均有限且下降，但没有首步局部误差/梯度分量，不能判定优化停滞或Jacobian有害，也不支持盲目加fit。先填补此证据缺口。

研究问题：在相同起始状态，归一化两步终点监督是否掩盖首步教师误差？完整两步梯度相对原生一步的方向差，由首步/第二步直接梯度还是第二步输入Jacobian贡献？这比立即完整训练detach控制更便宜，也避免将整首步梯度被删误称Jacobian消融。

## 拟定固定设计（正式提交前须来源冻结）
- 仅训练数据：按四家族各取排序前4个训练prompt，共16，不根据开发成败选样。共同底座生成各自一次Flow10未解码轨迹，固定诊断噪声seed80080+排序索引。取j=0/3/6/8四anchor，共64固定状态；不是历史训练真实接受状态，不可冒称。
- 三个预指定权重：共同初始S0、Phase77最终EMA、Phase79最终EMA。全部从同一底座anchor开始重新计算各自两步；老师固定同外部GenEval专家/CFG1，学生CFG4.5、FP32组合/native AMP，same sigmas/normalization。它隔离同状态机制，不代表最终模型自己的全部on-policy分布。
- 只输出潜变量误差/梯度数值，不VAE decode，不OCR/五质量重评。此为机制诊断，不能作为新任务质量证据；已有全部任务和五质量结果继续引用Phase79。
- 预先报告所有64状态及按训练prompt聚类的汇总；首步误差、归一化终点误差、full/local/direct-first/direct-second/Jacobian norm与cosine、零梯度数量。零norm的cosine返回null，不用0充数。
- full对照：两步完整梯度；second-only：整个middle detach；no-input-Jacobian：保留首步直接项和第二步参数直接项。诊断采用cotangent分解，不偷偷修改训练图。三者固定θ/输入的前向要一致；teacher/anchor detach，autograd.grad不改变参数.grad、不更新optimizer。

## 工程进度与预算
独立core.py已实现代数分解；CPU合成非零Jacobian、常速度退化、endpoint cancellation、零梯度、无副作用/不合法输入检查。真实原生AMP、梯度累积尺度、权重/teacher切换与checkpoint重计算的集成尚未完成，不能把CPU通过当GPU已通过。
拟定GPU上限4500 L40S秒/1作业，但尚未预留/提交；正式冻结时需用实际操作计数估算时限，必要时先缩小预声明状态数且在任何结果之前记录，不在运行中挑子集。必须完整跑完三权重/所有状态，否则仅部分诊断，技术失败不自动重试。遵循campaign上限和提交前四分区probe。

## 如何据此作决定
这轮不设能“保证work”的数值阈值，不用训练状态梯度余弦选开发checkpoint。完整数据若支持稳定端点抵消且首步误差保留，则另立保持首步监督的单变量学习对照；若主要是输入Jacobian方向影响，则另立只去该项的控制；若未发现这些机制，不扫更多horizon/学习率，优先评估任务专家/OCR保持和计算匹配缺口。任何解释都需下游完整学习对照才能成为有效方法结论。

当前：只有CPU工具及草案，无GPU提交，无新学习收益。正式实施前仍须测试真实框架接线、冻结所有source/weights/prompt/noise/hash、研究问题/资源/停止规则，并reserve记账。
