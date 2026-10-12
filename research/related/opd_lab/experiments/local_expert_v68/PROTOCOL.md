# Phase68：接受比例缩放实际学习率的匹配稳定性诊断

## 问题与唯一算法变量
Phase66的verified和同量random都只学习22/1024目标并严重退化。本轮检验：
保持既有严格语义选择与全部loss定义，显式缩小稀疏批次的Adam实际步幅，
能否减轻共享退化，并使verified出现超出random的完整学生收益？
每个非空64查询批接受m个：effective_lr = 3e-4 * m/64。
仍按sum(mask*原OPD transition-mean loss)/64反向；Adam betas/eps/decay、
clip、fp16、scaler、EMA时序不变。m=0时跳过optimizer/EMA，不执行零lr更新。
缩小lr也缩小AdamW衰减，属于整个optimizer步幅规则，不是单独梯度尺度因果实验。
这不是修正官方OPD的bug、不是充分训练OPD，也不保证按接受比例缩放理论最优。

## 固定起点、数据与比较
verified和random均从Phase66同一纯SD3.5-M＋fresh LoRA重新开始，fresh Adam。
同64训练文本、原1024 prompt/noise/time日程、16轮、每批64、每批拟合1次。
同GenEvalTeacher CFG1、student CFG4.5，Flow10、512px、text128、rank32/alpha64。
只评价最终epoch16 RAW，不挑中间/EMA/seed，不增加teacher/保持项/新gate。
random必须读取本次verified完整的每批/任务族/模板接受数量，并独立训练；不复用
Phase66的22个计数、名单或训练权重。两臂逐批lr及更新数匹配，时间分布不保证匹配。
规则改变会使在线状态、接受数量和目标改变；对旧Phase66是整套在线规则比较，
不是相同历史目标下的纯lr因果效果。同seed不表示默认反向逐位确定。
首轮64查询/128候选hash与Phase66核对，因为此时尚未发生任何参数更新。

## 观测与固定评价
额外记录每次实际lr、接受数、裁剪前梯度范数和参数位移；在更新前保存实际接受
状态、target、context、对应冻结behavior及候选hash。保存不修改训练目标或RNG，
CPU测试检查观测开关不改变Toy训练。此轮不增加训练状态终点评测预算，保存状态
是后续修复兑现核验的前提，不能将其或MSE当成已经测得兑现。
每臂完整旧128开发题，同prompt/noise、同五质量模型和任务评价器，256新开发终点、
1280质量值；最多4096训练候选PNG。初始/native/all/旧verified/random只读复用Phase66。
两臂完整单adapter部署、RAW一致性、资源SHA和A接口重评分检查继续执行。
confirmation512/final768封存；人类标签仅辅助；agent/远程/子agent读图0。

## 预声明判定，不挑正结果
主描述比较为新verified−新random的开发B joint。稳定性对照为每个新臂减去自己
Phase66同名臂；另完整报告新两臂相对base/native/all的joint、OCR exact/CER、
结构全正确、A unknown、四任务族，以及PickScore/26、CLIP、Aesthetic、HPSv2.1、
ImageReward五项质量。所有差异都报告，不仅报告通过门槛者。
族内按prompt配对bootstrap20000次/seed66066给描述性95%CI；单训练seed，旧开发集，
不计作确认性推断或训练seed不确定性，不声称同算力。
稳定性信号：两个新臂相对各自旧臂joint/OCR点估计都严格增加、五质量点估计均不负、
且每臂至少一项质量CI下界>0。即使通过也只支持缓解损伤，不等于筛选有效。
进一步机制/独立复现实验的必要门槛：verified对新random及旧native的joint CI
下界都>0，五质量对base/native均没有明确负向CI（上界不能<0）。
无明确下降不等于质量等效；接受状态兑现尚需另测。只恢复初始能力判作稳定性改善，
不得称论文方法成立。不通过则停止扩大本稀疏配方，不自动扫lr/加epoch/换teacher。
后续回到广覆盖OPD监督的设计需独立协议与预算，不能自动叠加损伤屏蔽机制。

## 历史依据和局限
Phase48曾在发布Student固定状态续训中将lr从3e-4降至1e-4，减轻退化但未证明
verified优于random；本轮是pure-base在线、与接受数量绑定的实际步幅规则。
Phase57保持项、Phase59/60 CFG已提示任务/质量取舍，本轮均不同时改变。
代码正确性与方法有效性分开；共享稀疏失败不是证明Adam是唯一根因。

## 独立预算与调度
用户本轮“好的请进行下一步”授权本次必要匹配实验。独立上限两个作业，各1×L40S、
7200秒，总请求14400 GPU秒（4 GPU小时）；包含加载、训练、观测验证、生成与全部评分。
Phase66两稀疏臂各约89分钟，新在线接受数可能增加，120分钟作为硬上限，不无限延长。
Slurm请求类型l40，实际设备必须NVIDIA L40S。verified先行，random以afterok依赖，
失败取消依赖。每个真实提交前检查minilab-gpu/sablab-gpu/scu-gpu/preempt_gpu；
sablab跨项目allocated＋pending预留不超6。不取消/移动其他项目任务。
允许preempt_gpu；中断/技术失败即停止，--no-requeue、自动重试0，新定时器0。
队列可启动估计超过1小时暂不提交；随机的估计另受verified实际完成限制。
独占intent与receipt阻止重复提交；预算按请求计、不回收旧余额或失败预算。
完成档案不可修改；CPU报告错误另立修复记录。无自动下一轮实验。
