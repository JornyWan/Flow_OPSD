# Phase57：学习完整选择策略，包括拒绝时的原动作

用户在Phase56审计后要求“请继续进行下一步”。落实已提出的独立全幅fallback问题；
本轮上限由执行计划保守设置为3×5400 L40秒（4.5小时），不是用户明说的数额，
不复用旧预算。Phase56半幅门槛未通过的结论不变，不再试幅度。

C2问题：明确拟合拒绝状态的anchor动作，是否减少质量漂移并保住验证组任务收益？
唯一变化：a_target=m*a_teacher+(1-m)*a_anchor，全部64状态用原transition loss/64。
原三个Phase55池、原mask、相同16次完整池更新、alpha1、LR1e-4、AdamW/scaler、
精度、Flow10/CFG4.5/512、query1部署与其余anchor查询均不改。直接恢复每池原
fresh_initial_training.pt的发布权重、空Adam/scaler/RNG，不承接训练过的权重。
每池random/verified各一臂，原masked结果只读复用，全部三池事先固定，不按效果停池。
没有新专家、质量权重、阈值、retention系数或在线刷新。随机组也完整监督，防止把
一般保持原模型的收益错误归因于验证选择。0教师接受时跳过optimizer，避免AdamW
对纯anchor目标引入变化；保留全量评测、不重抽。

这是完整选择目标的受控机制比较，不是同算力比较：每次64个反向查询，对照为
12/11/11。每池训练2048次反向查询，另重复首更新的审计128次；总2176/池，
另64个anchor动作与64教师动作精确回放、128个最终动作MSE。缓存教师动作不重采集。
同16步不能当成同GPU-hours，后续强算法比较须单独匹配实际成本。

每池新生成两臂×(64train+128dev)=384图，三池1152；每张完整task+五项质量，
5760新质量值。原候选、masked学生、共同dev anchor均复用原评价身份且核对哈希。
每池8历史task parity；全384轨迹/prefix、导出LoRA重载、128动作MSE审计；训练
teacher/anchor/behavior不变。确认512、最终768与第二噪声封存；人工标签不做主判定。
agent图片/截图/远程视觉读取0，不创建周期定时器。

主指标：128已暴露dev上verified_full−verified_masked的B joint，每池及三池等权均差。
次要固定对照：random_full−random_masked、verified_full−random_full、两新臂−anchor、
原verified_masked−random_masked。所有对照全报15项自动指标，包括joint/OCR/结构/
unknown/保护损伤与PickScore/26、CLIP、Aesthetic、HPSv2、ImageReward；按family/
domain/template分组，不把质量代理分数称为视觉真值。接受/拒绝子集、直接修复兑现、
原教师MSE/anchor漂移只用于解释，不替换dev主指标。

单池family分层配对prompt bootstrap20000，seed57057+池序号(+1用于dev)。跨池
bootstrap20000、seed57057，先抽run再抽共享prompt；只有128独立dev题，不是384。
三池固定且旧数据已暴露，区间探索性、未作多重比较校正，不是独立确认或正式非劣。
只有主差均值及三池点估计均>=0、verified_full−verified_masked五质量均差都>=0且
至少一项质量区间下界>0，才认为值得另立同算力强对照；否则记录取舍或不确定，不扩大。
即便通过也不证明筛选贡献；verified_full−random_full须独立解释，不改主指标。

资源：三个串行L40作业，各Slurm90分钟/内部5340秒；失败计费，不自动重试、换池
或扩预算。技术失败停止剩余池；完整GPU结果可在独立CPU记录审计，但真实失败
不能改写。GPU只写小完成标志，哈希/统计/HTML/CSV/ZIP在CPU完成。CPU错误另建
修复记录，不重跑GPU。管理器有一次性启动与提交意图锁；仅管理本轮三池，不定期
唤醒agent，不扩展下一轮。旧完成档案不修改。
