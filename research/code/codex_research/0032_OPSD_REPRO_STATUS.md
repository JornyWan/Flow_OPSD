# 0032 两篇论文的复现状态:一篇完整,一篇只有 1/5 列(2026-08-30)

用户问"我们是否有 baseline 结果、基于什么、与官方复现是否一致"时,
我第一次回答把 OPSD 说得太轻("可比列吻合")。**准确的说法是两篇差别很大。**

## ① DiffusionOPD(Student 那篇):**8/8 列完整复现**

官方发布的 Student LoRA 在本集群重评 vs 论文正文 "Ours" 行:

| 列 | 论文 | 我们 | 相对差 | /2σ |
|---|---|---|---|---|
| GenEval | 0.96 | 0.9615 | +0.16% | +0.05 |
| OCR | 0.94 | 0.9273 | −1.35% | −0.65 |
| PickScore | 23.99 | 23.9850 | −0.02% | −0.16 |
| ClipScore | 0.297 | 0.2969 | −0.03% | −0.05 |
| Aes | 6.15 | 6.1506 | +0.01% | +0.02 |
| HPS | 0.342 | 0.3423 | +0.09% | +0.21 |
| ImgRwd | 1.50 | 1.5047 | +0.31% | +0.34 |
| UniRwd | 3.50 | 3.4997 | −0.01% | −0.01 |

**八列全部在 ±1 个 2σ 内**,七列相对差 <0.35%。
项目笔记记为三重验证:官方 Student 重评 == 自训 mopd == 论文 Ours 行。

⇒ **我们全部 baseline 的锚点是可靠的。**

## ② DiffusionOPSD:**只复现了 1/5 列**

| 列 | 我们 | 论文 OPSD | 状态 |
|---|---|---|---|
| **PickScore** | **25.2133 ±0.054** | **24.940** | ✅ **唯一可比,+0.27(+1.1%)** |
| ClipScore | 0.2548 | 0.340 | ❌ 论文用 clip 专训 checkpoint |
| HPSv2.1 | 0.3108 | 0.390 | ❌ 论文用 hps 专训 checkpoint |
| Aes | 6.2761 | 12.080 | ❌ 论文用 aes 专训 checkpoint |
| ImageReward | 1.1654 ±0.0229 | 1.76 | ❌ 论文用 imagereward 专训 checkpoint(2026-08-30 已补测) |

论文 OPSD 行是 **reward-specific** 的(每列用该列 reward 训到的 checkpoint),
而**我们只训了 `pickscore` 一个臂**。要补到 4/5 需要再训 clip/hps/aes 三个臂。

协议已逐项核对无误:DrawBench 1000 条(`pickapic/test.txt` 与 `drawbench/test.txt` md5 相同,
仓库脚本注释写明"论文的 held-out 评测集是 DrawBench,故复制为 test.txt")、
flow 采样 40 步、gs=1.0、512²、noise 0.7、seed 42。

## ③ 解决了一个悬而未决的红旗:Aes 12.08 vs 6.28

`compare_opsd_repro.py` 尾部原有一句**未经验证的猜测**:
"Aes 论文 12.08 而我们约 6.1,**评测器版本可能不同**,差异大时先查版本"。

**已查清,不是评测器问题:**

- 两个仓库的 aesthetic scorer **逐行相同** ——
  同一个 `openai/clip-vit-large-patch14` + 同一个 `sac+logos+ava1-l14-linearMSE.pth`;
- 决定性判据是**两篇论文报告的同一个基线模型**:
  OPSD 论文 base(gs=1.0)= **5.13**,我们 base w/o-CFG(gs=1.0)= **5.18** ⇒ **尺度一致**;
- 12.08 是**直接把 aesthetic scorer 当 reward 训练**的效果 ——
  ReFL 同样到 12.09、DiffusionNFT 到 9.11,而 base 只有 5.13。

我们的 pickscore 臂 Aes 停在 6.28(比 base 高约 1.1),**因为我们没有训 aesthetic**。
⇒ "❌不可比"的标注是对的,但理由是 **checkpoint 不同**,不是评测器不同。**红旗撤销。**

## ④ 对我们 baseline 命名的一处更正

我一直把 `crd` 叫 "**OPSD-direct**",容易被读成"复现了 OPSD"。**它不是。**

- 论文 OPSD:从 **base SD3.5-M** 出发做 reward 专训;
- 我们的 `crd`:从**发布的 DiffusionOPD Student** 出发做 direct continuation。

**起点不同,论文数值本就不是 `crd` 的对标目标。**
`crd` 的作用是**机制 control**(展示不加保护会怎样),不是"论文方法基线" ——
论文方法基线是 `mopdc`(MOPD-continue),那一条对标的是 DiffusionOPD,而那篇复现完整。

后续文档提到 `crd` 时应写 "**OPSD 式 direct continuation(自 Student)**",不简称 OPSD。

## 结论

| 问题 | 答案 |
|---|---|
| 有 baseline 结果吗 | 有,五臂全部跑完(200ep, exit=0),六点评测齐 |
| 基于什么 | 官方发布的 DiffusionOPD Student,本集群同代码同协议 |
| 与官方一致吗 | **DiffusionOPD 8/8 列一致;DiffusionOPSD 仅 1/5 列,其余缺 reward 专训臂** |


## ⑤ ImageReward 已补测(2026-08-30 13:13)

原记录"⛔ 缺 openai clip 包"不成立(见 0033 §⑤)。实测 `ImgR = 1.1654 ± 0.0229`(n=1000),
位于 FlowGRPO(1.01) 与 ReFL(1.28) 之间,远高于 base(−0.58) ——
**pickscore 专训带来的附带增益**,与论文 1.76 不可比(那一格是 imagereward 专训 checkpoint)。
可比列数**仍是 1/5**(只有 Pick)。

**这次补测同时是一次脚本自检:** 新结果与既有 `opsd_eval_ck100_raw.json` 的
4 个公共列**逐位相同**(差恰为 0.000e+00,协议参数 prompts/steps/gs/seed 亦一致)
⇒ 对 `opsd_eval.slurm` 的改动(ARM 标识、REWARDS 透传)没有触动评测路径,
且评测在不同 A40 节点间可复现。
