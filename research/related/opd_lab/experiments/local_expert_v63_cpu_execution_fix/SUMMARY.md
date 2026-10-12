# 独立执行修复：随机组准备不再依赖计算节点git

原Phase63/Slurm1064571保留FAILED 1:0、185 L40秒的完整记录，不改档、不重提GPU。
故障是验证组训练成功后，普通`runtime.prepare`在计算节点重新快照源码时找不到git。
这是CPU编排依赖导致的GPU作业失败，不是已经观察到的算法训练或评测失败。

## 修复与验证

新增`runtime.prepare_matched_random(verified_run, destination)`：

- 首先验证原verified完整完成、count reference来源、冻结数据/源码/资产和训练合同。
- 只克隆framework/upstream/data/site/environment等输入，复制符号链接而非大模型资产。
- 随机臂仍从原发布学生和fresh Adam开始，不复制verified的训练权重、优化器或图片。
- 只改method、run内文件路径与count reference，重新生成manifest并检查比较合同。
- 不调用git或任何子进程；拒绝覆盖现有目录，拒绝未完成或被改动的source。

52项回归通过，包含禁用PATH且禁止subprocess/git的测试、输入克隆后真实toy训练、
source篡改/不完整/覆盖拒绝，以及原在线语义与完整resume测试。与Phase62的105项是重叠
测试集合，不将105+52报告成157个独立测试。

随后对真实已训练的Phase63 verified源码做同样检查：PATH设为不存在目录仍完成准备，
得到`runs/le63_semantic_random_cpu_prepared_002`，状态仅prepared、没有STARTED/训练结果。
它的框架内容digest与verified完全相同，runtime/data/site/参考校验通过，没有提交新GPU。
旧失败的`le63_semantic_random_001`部分目录保留不动。

## 已完成训练的CPU核对

verified32个查询覆盖全部0..8时间步，两轮分别接受4/16与1/16。
两个checkpoint均381/382个LoRA张量改变，全部有限；raw导出逐位等于training.pt中的student，
EMA导出等于记录的EMA及初始学生（未达更新间隔）。Adam所有状态分别step1/step2。
这些只证明训练与保存实际发生，不是修复兑现率、joint增益或质量保持。
两轮查询seed/time不同，不能把4→1单独解释成学生变强或变弱。

## 下一步边界

后续应另立有限执行协议，**只运行尚未运行的随机组与共同评测**，复用现有verified结果并
计入其185秒原账务。先验收两组逐批监督数量/状态起点、单adapter完整轨迹、A评分接口
parity及同8dev的Bjoint/OCR/CER和全部五质量。已准备目录本身不等于GPU授权或已经续跑。
不重训verified、不为寻找正结果换prompt/seed/最后checkpoint，不把小样本当论文主表。
若随后仍有失败，继续如实保留原状态和实际费用，不覆盖或假称完成。

CPU修复读图0、GPU0、定时器0；confirmation/final封存。正式共同起点强OPD比较尚未启动。
