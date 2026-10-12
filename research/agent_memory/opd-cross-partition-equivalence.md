---
name: opd-cross-partition-equivalence
description: "CA-OPD 可以跨 scu/minilab/sablab 三个账号派活,但臂间对比必须保证硬件等效"
metadata:
  type: feedback
---

用户 2026-08-27 22:40:"注意 scu, minilab, sablab 我们都能用,但要保证他们是等效的即可。"

**Why:** 队列常被他人占满,只盯 `sablab-gpu` 会让作业排到第二天;三个账号一起用能立刻解锁空闲卡
(当晚从 1 个作业在跑变成 5 个)。但 CA-OPD 的结论几乎全是**臂间配对检验**,
换硬件等于在配对里引入一个未受控变量。

**How to apply:**
- 派活时把 `scu-gpu` / `minilab-gpu` / `sablab-gpu` / `preempt_gpu` 一起看(见 [[cluster-usage-conventions]] 的分区表)。
- **评测(eval)的既定基准是 A40**:`ev*` 作业历史上全部跑在 ai-gpu01/02/03/04。
  新臂的 10 点评测要继续用 A40(ai-gpu05 也是 A40,在 preempt_gpu 里),否则与所有既往臂不可比。
- 训练/测量换硬件时,**加一个同参数的复现点**去量化差异,而不是假设可忽略。
  例:ρ 扫描从 H200 挪到 H100 时,补跑一个 ρ=0.100 的 `WFC_TAG=h100rep`,与 H200 上的同 ρ 点直接对比。
- 换卡型时注意显存:A40/L40=48GB,A100/H100=80GB,H200=141GB。gcprobe 在 48GB 上结构性 OOM。

**2026-08-29:这条规则被违反了两次,原因是两套评测模板并存。**
`eval_all.slurm`(sablab-gpu + **A40**)与 `eval_opsdck.slurm`(minilab-gpu + **L40**)同时存在,
新写监视脚本时沿用了手边的那个,就把跨臂硬件搞不一致了:
- **GN 线**:GN 臂十点全 L40,对照臂 rtcraw 全 A40;
- **warmup 种子 43**:warmup 臂 A40,对照臂 rtcraw43 L40;
- (CR 三臂全 A40,幸免。)

**已实测的 A40→L40 硬件效应**(同一 checkpoint `rtcraw-421` 各评一次,单位为该轴 2σ):

    pick +0.13 | aes 0.09 | hps 0.30 | clip 0.45 | geneval 0.33 | ocr 0.75

**六轴全部 <0.75×2σ,即都在同硬件重测噪声内。** 所以两条结论都不受影响
(GN 的落后是 46/11.7/16.6 个 2σ,大一到两个数量级;warmup 种子 43 校正后判定不变)。
**但这是运气**:若硬件效应大一个数量级,两条结论都会毁掉。

**追加做法:**
- 新建评测脚本时**显式核对 partition 与 gres**,不要沿用手边模板;
- 提交后用 `sacct -j <id> --format=Partition,ReqTRES` **回查**实际硬件,不要只看提交命令;
- 发现跨臂不一致时**直接测量**(同一 checkpoint 在两种卡上各评一次),
  不要用"效应很大所以应该没事"搪塞。

Related: [[opd-caopd-project]], [[cluster-usage-conventions]], [[phase0-scheduling-preferences]]

**2026-09-05 实际踩到:** 审计 `arm_provenance.json` 的 `gpu` 字段发现两个**已写进结论**的
对比是跨硬件的 —— inner2(A100)vs basenp3(H200)、base200(H100)vs basenp3(H200)。
lr2x 的 +0.0282 不受影响(同为 H200 np=3)。已派同 seed 同 np 只换卡的 A/A 对照来量硬件项。
**每次建对比前先 `grep gpu arm_provenance.json` 对一遍,不要假设同一批实验跑在同一种卡上。**

**2026-10-11 用户放开:单 reward 的方法 run 可以用 preempt 跑**(先一个 seed)。做法:`--no-requeue`,被抢占就**从零重投**而不是续训
(MSP 系续训 = EMA 重启,会污染,见 §4.811);preempt 副本与 owner 分区后备副本共用 OUTPUT_DIR,先完成者生效。
卡数:有效批固定 48×24 = 1152/更新,3/4/8 卡只改微批切分;我们 4 卡的 open3 300 步 canonical 与论文 25.51/0.333/0.389 吻合(§4.896)。
