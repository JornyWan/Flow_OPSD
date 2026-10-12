# Phase81 frozen execution contract (2026-10-11)

The design text below is retained as pre-execution rationale. Its earlier 'not integrated' status is superseded by this contract; GPU qualification remains pending until execution.
Single L40S,4500s hard limit, no requeue/retry, no optimizer updates.256 principal endpoints and1280 quality values; all64 training prompts/four arms required. No automatic subsequent training or radius/timestep sweep.
Freeze rho0.10 and nearest positive Flow40 sigma to0.278; no change based on clipping frequency/outcomes. Primary bounded-minus-matched-random joint; also compare noop/full, all task metrics/five qualities/families/gains/losses. Small CLIP changes alone do not veto, but no omitted harms. No target success implies learning success.
Reuse exact Phase77 single-default student deployment and finalEMA. All student prefixes/noops precede expert loading;64 GenEval query predictions only; remove expert before student suffixes. Every query requires exact saved-student prediction and full noop latent replay; first teacher transition must match pinned native deterministic Flow kernel maxabs<=1e-6. Student weights checked before and after. Bounds/random displacement norms audited per query (rtol1e-5,atol1e-7); inactive clipping must not be claimed as clipping efficacy.
Two existing Phase77 EMA dev images (first2 sorted IDs) are regenerated solely for byte-exact deployment parity, outside the64 training cohort; no new dev comparisons/selection. This clarifies the draft's 'no dev' as no new efficacy experiment on dev. Eight old quality references and existing task reference rows use identical pinned scorer identities; max quality difference<=1e-4, task fields exact. These automated replay/decode operations are not agent image reads.
36 CPU tests cover targets, nonlinear rollout, complete scoring matrix and temporary expert lifecycle. Real GPU validation still required. Scorer/runtime/model inventories and all code/config/adapter inputs hashed; large backbone files only metadata. Failure or incomplete matrix stops; report repair must use independent directory. Submission must first probe4partitions, obey sablab allocated+pending<=6, reserve4500s from existing campaign, and fit the remaining time window. No new timer/agent.

---

# 受限专家方向：目标机会诊断草案

2026-10-11，用户同意继续参考OPSD，先验证目标，再验证拟合。
状态：目标构造核心已实现/CPU通过；真实模型接线、环境/权重快照、完整评分与GPU验收尚未完成。未分配正式phase、未reserve、未提交，不是已冻结运行协议。

## 研究问题及历史反证

在已训练好的原生OPD学生附近，外部GenEval专家的受限clean-output修正，是否比同幅度随机修正带来更多终端联合成功？
这比在未训练底座上证明教师有用更接近当前论文缺口。Phase77最终EMA为固定行为/后缀，不能把强baseline的收益算作新方法。

Phase56已测过三池192状态、Flow10 query1、固定半幅教师动作，未观察到联合任务与五质量同时改善，已停止该幅度搜索。本草案不复用其门槛、不继续扫描alpha。
Phase73的64训练prompt显示GenEval整体较强不等于单步介入收益；OCR局部提高文字却损坏结构，不扩大该路由。
本诊断改变的是研究场景：强原生EMA的真实低噪声状态、部署Flow40、OPSD启发的相对clean-anchor半径、等幅随机方向控制。不能与Phase56跨轮差异做因果比较。

## 拟固定设置

- 学生/前后缀：Phase77最终EMA，精确SHA与单物理adapter部署路径，固定外部GenEval专家；学生CFG4.5、专家CFG1保持已定义配方，不据结果改CFG。
- Cohort：现有512训练prompt四家族各按ID取前16，共64；固定新噪声seed规则，不依据成功/失败或dev选择。保存完整case与seed文件。不是新独立测试集。
- 部署：512px、text128、Flow40、FP32 CFG，沿原生已验收部署；选择实际sigma最接近0.278的唯一非终点query并记录实际sigma/index，tie按更早索引。所有臂同query，不扫描时步。
- 相对clean-anchor半径rho=0.10，来自固定公开OPSD配方，不依据本项目dev选择；不扫其他rho。此值可能根本不触发裁剪，必须报告，不能预称限幅有效。
- 每个prompt先采集同一冻结behavior前缀；四臂只替换该query动作，后缀均由同一冻结behavior在各自新状态上重算。
- 4×64=256主要终点，全部joint/OCR/结构/CER/unknown及1280项五质量。另评分固定历史参考，完整评分成本在提交前单独核算。
- 本轮零optimizer更新、零agent读图，不动dev128、confirmation512/final768；不是学习收益实验。

## 四臂及因果边界

令yB=z−sigma*vB，d=−sigma*(vT−vB)，r=rho*norm(yB)。

|臂|clean displacement|作用|
|---|---|---|
|noop|0|原生EMA自身后缀|
|full_teacher|d|当前状态直接使用外部教师动作|
|bounded_teacher|d * min(1,r/norm(d))|方向保留，仅上界裁剪|
|matched_random|随机单位方向 * norm(bounded displacement)|逐样本严格同幅度控制|

实际介入velocity为vB−displacement/sigma。不能把clean目标当下一状态写入，也不能把负号写反。
noop直接返回原vB，full直接返回原vT，未裁剪teacher直接返回vT；避免inverse round-trip导致本应相同的臂出现数值差异。
随机方向使用独立命名seed，不消耗采样主RNG。teacher相同/半径0时，bounded和random均保持noop。
此次没有构造reward-gradient目标、没有负目标反射loss，不能称原生OPSD。教师来源未变，仍是外部专家OPD候选。

## 主要终点和停止解释

主比较：全部64题的bounded_teacher−matched_random joint配对差；同时必须报告bounded_teacher−noop、bounded_teacher−full_teacher及全部任务/质量、家族和新增/损失。
以prompt为单位做描述性配对区间，不将四臂或多个指标当更多独立样本。人工仅辅助，不用视觉人工筛选通过样本。
如果方向优于随机但不优于noop，不能称有净改进机会。如果裁剪未生效，明确说测试的是低噪声教师方向，不是半径策略优势。
目标改善也不能直接触发学生训练成功结论；下一阶段需要独立冻结同目标拟合对照。
小幅CLIP下降单独不否决，但全部五质量、任务损伤及unknown要报告。不会用调rho/时步或只看新增案例挽救本次结论。
技术失败不重试；cohort/评分不完整则不作收益结论，CPU报告错误另立修复记录。

## 剩余执行阻断项与预算

目标核心`python/opdlab/bounded_clean_targets.py`已完成18项CPU检查：逐样本半径/等幅、Euler转换符号、精确noop/full、零方向、零半径、batch隔离、detach/参数不变、随机数状态、不合法输入与低精度提升。
尚缺真实Flow40模型/权重/teacher接口接线、标准评分/参考桥接、四臂完整记录及report、真实GPU首query parity合同和运行时估计。CPU通过不等于GPU验收。
须以Phase77原生部署为参考验证noop轨迹一致，验证full动作与原生teacher接口一致；保存前缀、latent、动作、权重/源指纹与评分身份，保证后缀不误用teacher或behavior旧缓存。
现有campaign仍20700/48600请求秒，只有1个新增作业名额、单项至多16200秒。这个草案不预留额度；提交前必须冻结实际时限/输入/完整评分预算，经runner reserve和四分区probe。若准备或队列不能支持完整执行，在截止前停止新增提交。
本轮若占用最后名额，后续学习只能在新授权预算内进行，不暗中塞入未声明训练。PREFIX监督草案保留，但不与本诊断叠加。
