# 前瞻研究合同：paired-harm-veto-v1

2026-10-01，三臂有限机制比较。研究问题、比较与本页判据先于GPU结果固定；执行入口、
全部输入指纹、集成资格和预算reservation尚未完成，当前不可提交。与已存4fit合同独立。

## 为什么先做这个问题

原严格正例目标只有51/1024，OPD监督支持被大幅删减；更多fit不会补回缺失支持。
原任务/模板随机又未匹配时间权重；若直接补时间分层，56个已知损伤中51个没有随机
替代空间。新日程先解决可交换性，再检验**明确负向终端证据是否比无信息的同量屏蔽
更有用**。这不是对原始“只学确认正例”的同义实现，而是保留OPD基础监督的一种修正。

## 三臂与唯一算法差别

- all：新日程上所有外部教师目标都拟合，本地单查询OPD估计器，不叫完整原生OPD。
- veto：对valid训练A证据，任一声明子句true→false就不学习该教师目标；其他查询保留。
  false→true不会抵消另一项损伤。unknown表示验证器弃权，沿用基础监督，不标为安全。
- random：从同一纯底座/fresh LoRA和空Adam重新训练，每批按任务族×模板×时间匹配
  已完成veto的保留数量，固定hash均匀挑选，自己的valid A标签不决定名单。

三者共同使用64训练prompt、1024查询/16外轮、每批64、1fit、固定实际lr3e-4、原AdamW
betas/eps/decay与clip1、loss除以全部64查询。无学习率搜索或接受率缩放。只有屏蔽名单
不同；all保留数更多，因此不是相同监督量。三臂都做相同候选评分以核查来源，其完整
GPU开销分别报告，不宣称查询或总算力最优。空保留批跳过Adam/EMA；如出现则报告实际
更新数和训练量，不伪造16更新。veto/random更新与实际LR序列必须一致。

教师GenEval CFG1，学生CFG4.5，Flow10、512px、text128、fp16、rank32/alpha64；纯SD3.5-M
加原共同初始化的fresh LoRA，教师/模型/评分资源内容指纹固定。候选noop和全幅teacher
共享旧behavior前缀及后缀；每外轮采自己新轨迹，目标detach，1fit后下一轮刷新behavior。
无第二推理adapter、奖励梯度、critic、新anchor或shadow update。

## 独立的新采样日程

版本balanced_family_template_paired_time_v3：每个任务族×模板每批恰好两个不同prompt，
保留旧顺序与逐查询noise种子，两者共享时间索引。每个9轮区块内，每一分层的时间索引
是固定seed独立洗牌的0..8排列。16轮内各索引访问1或2次（每次两条查询）。

这改变了批内时间相关性及跨轮相关性，不能称独立均匀时间抽样，也不宣称更新后的状态
分布下全OPD目标仍严格无偏。三臂共享同一新日程。不会把旧时间点A标签搬到新时间点。
若同一对的两项均应保留/屏蔽，random与veto不可避免相同；如果只屏蔽一项，则有两个
候选。最终须报告实际mask交集/不同数，不能人为强制random选择相反目标来制造差异。

## 终点与判定

仅最终epoch16 RAW，完整单学生Flow10。同128个已暴露开发prompt与固定noise，三新臂
各128终点，共384新开发图、1920新五质量值。原base与Phase66 native及旧all/Phase68
只读作背景；base复用必须核对初始权重、prompt/noise、生成与评分合同。旧native不是
与新日程逐状态匹配，其计算与采样差别完整标注。不得据此称超过完整MOPD论文配方。

主比较：完整dev的veto−random joint。共同必要比较veto−all joint，以及各臂−base。
全部报告joint、OCR exact、结构、CER、unknown、各任务族及PickScore/26、CLIP、Aesthetic、
HPSv2.1、ImageReward的绝对值和配对差。置信区间沿用按prompt、族内20000次bootstrap、
seed66066，反复使用开发集所得区间仅描述性，不计作多训练seed证据。

只有同时满足以下条件，才值得另立独立重复协议；不是论文有效性确认：

1. veto−random和veto−all的joint描述区间下界均>0，且veto−base的joint点估计>0。
2. veto−random、veto−all、veto−base的五质量均值差全部≥0；完整区间仍照报，这不证明
   质量非劣。OCR和结构分项不得省略，即使joint通过。
3. 来源/配方/权重链/评分合同与完整结果通过，veto/random实际更新与学习率匹配。

未通过不自动扩大veto、开放封存集或改变门槛。两种屏蔽都优于all但相互不分胜负，
只能支持减少部分监督或随机性有帮助，不能证明负向验证信息贡献。veto只减缓退化但
仍低于base，不叫方法成功。all也不改善base时，重新核对教师任务适配和训练充分性，
不能宣布OPD无效。若两臂mask完全相同，明确没有选择器对照差异。

本轮不新加接受状态前向诊断；先把有限预算用于完整三臂比较。训练A的损伤/未知、
时间覆盖、目标数、实际LR、梯度norm和RAW权重导出检查作解释，不冒称训练修复兑现。
confirmation512/final768封存，agent图片/截图/远程视觉读取0，人工仅辅助。

## 成本和执行前置

拟最多三个1×L40S作业，各10800秒硬上限，总32400请求秒；占用本12小时窗口已有上限，
不是额外授权，不返还失败或未用额度。all/veto可独立并行，random须在veto完整成功后
才准备/执行，其他项目不动。每次sbatch前四分区probe，sablab allocated+pending≤6，
无requeue/重试，先runner.py reserve。到17:05:15 UTC不再新增预留/提交。

提交前仍需：独立三臂控制器、manifest、原RAW权重与实际LR核查、完整报告与冻结判据
集成测试；分离veto的新A_decision语义和旧正例规则，验证评分cohort、拒绝残缺/错绑定
输出；严格核对旧base复用与来源。旧Phase68报告不能直接复用为本轮报告。若完整比较
成本或队列不可行，继续CPU准备，不先消耗一臂再声称比较完成。
