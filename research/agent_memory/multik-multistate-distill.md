---
name: multik-multistate-distill
description: "multik4(把单个 OPSD 目标铺到多个噪声态)ck50 早读 = 本项目第一个显著正向:+1.46×MDE t=+3.30,五轴同向"
metadata:
  type: project
---

2026-09-29:`multik4` 臂(`RPI_MULTIK_STATES="5,6,7"`)在 **ck50 早读**上给出本项目
**第一个显著的目标端正向结果**:`multik4 − baserep` 报告复合分 **+0.00483(+1.46×MDE, t=+3.30)**,
六条 reward 轴里五条同向为正、四条显著,其中两条 **held-out** 轴最强
(aesthetic t=+8.79、imagereward t=+5.87)。形状与 [[opsd-mirror-degeneracy-bug]] 那类
reward-hacking(clip 升 / held-out 降)**明确不同**。配对是三轴全同(A100/np=3/seed20260830/A40)。

**机制(已逐行核实 L2637-2641)**:它**不是**在高噪处重新构造目标。`_ypm is None` ⇒ `_tp = y_plus`,
即 **kq=8(σ=0.278)处算出的那一个目标**,被当作回归目标用在 k=5,6,7(σ=0.707/0.602/0.465)上;
每个态只算自己的一步去噪再回归到同一个 `_tp`。**没有额外 reward 调用、没有额外 VAE 解码。**
一句话:*OPSD 只在一个态上用它的目标,但那个目标对同一条轨迹上每个噪声态都是合法监督。*

**Why 这条比"高噪选择/hybrid"线更该做**:它绕开了 ∇R 在任何 σ 都是白噪谱、从不携带 layout
信息这个障碍(见 EXPERIMENT_LOG §4.604);高噪态上根本不求 reward 梯度。
同时正面回答导师方向①"高噪区如何 distill":**把低噪目标铺到高噪态,而不是去高噪找 reward 信号。**

**How to apply / 三条未决限制**:
- ⚠️ **+1.46×MDE 落在噪声地板的含糊带里**:canonical 臂两两差最大 1.23×MDE
  (见 [[opsd-noise-floor-measured]]),+1.46 只是它的 1.19× ⇒ **单 seed 判不了**,
  必须第二 seed 同号。
- **aesthetic +8.79 作废**(见 [[aesthetic-not-a-heldout-axis]]);
  **imagereward +5.87 保留**,且两个欠训参照的 imagereward 都是负的,符号相反 ⇒ 不是欠训假象。
- **ck50 不是判据**;§4.588 的 M/N 轴锚在 **ck100**,且 [[opsd-100-updates-not-converged]] 记着
  ck50→ck100 仍有 5×MDE 的移动。
- t=+3.30 是对**图**的(200 prompts × 5 图、单一训练 seed),而本项目 run-to-run 尺度是 ~1×MDE
  且曾符号相反 ⇒ 见 [[small-n-significance-flips]]、[[per-image-pattern-is-seed-dominated]]。
  第二 seed(20260905,对照 `bases2`)已排。
- 判定一律用 `judge.py`,配对键是 `r["index"]`;**n 必须是 1000**,n=200 是撞键的指纹
  (见 [[reuse-the-correct-impl-dont-rewrite]] 的 2026-09-29 补)。

同族臂:`multikF4`("3,6" 更高噪)、`multikALL`("1..7" 单调极值)、`multik2`("7" 单态)、
`multik4s`(snr 加权)。对照缺口已补:`baseline`(H200/np4)此前**从未做过官方四维评测**。

## 2026-09-30 重大更新:真正的结果在**少步数**上,不在满步数

**满步数(flow-40)的 +1.46×MDE 站不住**:(a) 贴着噪声地板(canonical 两两差最大 1.23×MDE,
见 [[opsd-noise-floor-measured]]);(b) **不迁移到第二个 prompt 集** —— `unseen1k` 上是
−0.15×MDE(t=−0.44),形状变成 clip 升 / pick 显著降(−4.20)/ hps 平的置换。

**但把采样步数砍到 4 步,优势在两个集上都很大且高度一致**(`multik4@ck50 − baserep@ck50`,
三轴全同 A100/np3/seed20260830,A40,n=1000):

| 集 | flow-40 | **flow-4** |
|---|---|---|
| drawbench | +1.46×MDE (t=+3.30) | **+7.05×MDE (t=+17.07), 胜率 75.1%** |
| unseen1k | **−0.15×MDE (t=−0.44) ⇒ 零** | **+6.09×MDE (t=+18.08), 胜率 74.3%** |

绝对退化(40→4 步):drawbench multik4 −0.0284 vs baserep −0.0468(**少掉 39%**);
unseen1k −0.0218 vs −0.0423(**少掉 48%**)。

**两个显而易见的反驳都被挡住**:unseen1k 上 flow-40 两臂起点几乎相同(1.6048 vs 1.6053),
终点 1.5831 vs 1.5630 ⇒ 既不是"训练更弱所以退化更少",也不是地板/回归均值效应。

**⇒ 论文论断应当是**:*把 OPSD 在单个查询态构造的目标铺到多个噪声态,**在满采样预算下不付代价**,
却让模型对**采样预算削减**显著更稳健。* 这绕开了地板问题与不迁移问题,
且少步/加速这个口径本项目已自证可发表(EXPERIMENT_LOG §4.589–§4.596)。

**未决的关键混淆**:`lr8x`(flow-40 上 +17.95×MDE)在 flow-4 上是否也大幅领先?
若是,"少步稳健"就不是 multi-state 监督独有的(R 轴,§4.620 §六 已登记判据,评测已排)。

**操作**:flow-4 已升为**主要**评测轴 —— watcher 的默认 `EVAL_CELLS` 改为四格
`drawbench:flow:40 drawbench:flow:4 unseen1k:flow:40 unseen1k:flow:4`。
判 σ 覆盖的形状(multik2/F4/HI/ALL/S3/SY)现在要在 **flow-4** 上看,不是 flow-40。

## 2026-09-30 01:00 证据已相当完整(EXPERIMENT_LOG §4.618–§4.625)

**步数斜坡(drawbench, ck50, multik4 − baserep):**
| 步数 | 40 | 20 | 10 | 6 | 4 | 2 |
|---|---|---|---|---|---|---|
| ×MDE | +1.46 | +1.10 | +1.43 | +2.75 | **+7.05** | **+23.64** |
| 胜率 | 53.0% | 53.3% | 53.0% | 58.7% | **75.1%** | **90.7%** |

**训练 checkpoint 无关(C1)**:flow-4 上 ck20/30/50/80 = +8.16 / +5.43 / +7.05 / **+9.15**×MDE,
t 全部 >13、胜率全部 >69% ⇒ 不依赖早停点(回答了 [[opsd-100-updates-not-converged]] 的质疑)。

**退化阶梯(40→4 步的绝对退化)**:未训练底模 −40.2×MDE → canonical OPSD −14~−19 →
**multik4 −8.6**;而 `lr8x` 是 −18.0(与 canonical 相同)⇒ **调 LR 抬绝对分但完全不改善少步稳健**。
flow-2 时 canonical 崩到 1.0831,**比未训练底模在 4 步的 1.1069 还低**,multik4 是 1.1611。

**十条替代解释全部被实测挡住**(见 §4.625 §四 的清单),其中最关键三条:
(1) unseen1k 上 flow-40 两臂起点相同(1.6048/1.6053)⇒ 不是"训练更弱所以退化更少";
(2) `lr8x` 的 Δ₄−Δ₄₀ 只有 +0.91/−1.50×MDE ⇒ 不是"训练更强"的通性;
(3) 少步轴对 lr8x/qstrain9 的一般放大是 ×0.93–1.20,multik4 是 **×4.83**,
    且 unseen1k 上从 −0.15(零)跳到 +6.09 ⇒ 不是轴的动态范围。

**评测口径**:watcher 默认四格 `drawbench/unseen1k × flow-40/flow-4`。
`unseen1k` 是有效轴(校准臂 lr8x +17.95→+20.97、qstrain9 −35.46→−36.02,比值 1.02–1.17),
且它 1000 个 prompt 全唯一(drawbench 只有 200 唯一 × 5 重复)⇒ 功效更高。
**σ 覆盖的族形状(multik2/F4/HI/ALL/S3/SY)必须在 flow-4 上读,不是 flow-40。**

## 2026-09-30 03:35 主臂更替:`multikF4`(2 个**高噪**态)取代 `multik4`

`multikF4` = `RPI_MULTIK_STATES="3,6"`(σ=0.858/0.602,**只有 2 个**额外态)。
对同硬件同 ck 的 canonical(`baseline`,H200/np4),**ck100**:

| 集 | NFE=40 | NFE=4 |
|---|---|---|
| drawbench | **+5.62×MDE** (t=+14.21, 70.9%) | **+10.70×MDE** (t=+23.60, 82.7%) |
| unseen1k | **+3.34×MDE** (t=+11.03, 64.4%) | 评测中 |

**与 `multik4` 的关键差别**:后者在 unseen1k/NFE=40 上是 **−0.15×MDE(零)**,
`multikF4` 是 **+3.34×MDE**。少步上两者**等效**(ck50 flow-4:+7.52 vs +7.05,差在 0.55 余量内)。
两臂只共享 σ=0.602;`multikF4` 另加**更高**的 .858,`multik4` 另加**更低**的 .707/.465
⇒ **更高的 σ 更好,而且 2 个态就够(更省算力)。**

**论断升级为**:*满步在训练目标上显著更好 + 少步上大幅更好*。
⚠️ **但有一格必须标出**:unseen1k 的**满步**格上,已验证的 held-out 轴 `imagereward`
只有 **+1.18(t<2)**,训练内三轴 >5.98 ⇒ **该格增益由训练目标扛着,held-out 未确认**。
少步格(drawbench NFE=4)的 imagereward 是 **+11.24**,强确认。
⇒ **两句话要分开写,不能合并成"全面更好"。**

效应随训练增长(ck50→ck100:+3.19→+5.62、+7.52→+10.70、unseen1k 满步 +1.22 未判→+3.34 判正向)。
seed 复现 `mkF4_s2`(seed 20260905,A100/np3,对照 `bases2`)已在 sablab 链首。

## 2026-09-30 11:50 seed 复现成立,"单 seed"这条保留解除

第二个 seed(20260905,A100/np3,对照 `bases2`)在 ck50 上:
**NFE=4 +8.45×MDE(t=+19.84,胜率 77.5%)、NFE=40 +4.04×MDE(t=+9.41)**,
均**同号且略强于**主 seed(+7.52 / +3.19)。逐轴四条在两个 seed 上全正。
两 seed 的 Δ 差(+0.93 / +0.85×MDE)**小于 canonical 重跑散布**(最大 1.23×MDE,见
[[opsd-noise-floor-measured]])⇒ 效应对 seed 稳健。
⇒ **此前每节都带的"仍为单 seed"限定,在 ck50 上解除。**
