---
name: reuse-the-correct-impl-dont-rewrite
description: 现成脚本里已解决并写明注释的坑,另写一次性统计命令时会再踩一遍;要复用函数而不是重写
metadata:
  type: feedback
---

2026-09-09:为了看有没有空闲 GPU 派活,现写了一条 `squeue -h -w $n -o "%b" | grep -o "gpu:[a-z0-9]*:[0-9]*"`
的一次性统计,得出「gpu14/16/17/18 各 4 张全空」,据此向用户报了一张空闲表并派了三个训练臂 —— **全是错的**,
集群当时几乎满载(gpu17 已占 4、gpu14 已占 4)。

`wait_and_launch.sh` 里的 `free_on()` **早就正确实现了**,而且开头注释就写着这个坑的原文:
> 空闲量必须读 scontrol 的 AllocTRES:squeue 里别人的作业可能报无类型的 gres/gpu:1,
> 用 'gpu:类型:N' 的正则会漏算占用、高估空闲。

**Why:** 别人的作业申请 `--gres=gpu:1`(不带类型)时,`%b` 里就是 `gpu:1`,类型正则匹配不到 ⇒ 占用被算成 0。
这个坑我自己踩过一次、修好、还写进了注释,然后**因为图快另写了一条一次性命令,又踩了同一个坑**。

**How to apply:** 需要某个量时先 `grep` 现有脚本有没有函数已经算过它(这里是 `free_on`);
有就直接 `source`/调用,不要为了「一行就够」重写。凡是涉及 SLURM 资源占用,一律读
`scontrol show node` 的 `AllocTRES=...gres/gpu[^=]*=N`,不要读 squeue 的 `%b`。
参见 [[cluster-usage-conventions]]、[[print-the-series-first]]。

**2026-09-10 补(同一坑的第三种形式)**:判断"有没有空闲卡"必须**同时看三样**:
1. `AllocTRES` 的占用数;2. **`State` 是否含 `RESERVED`/`DRAIN`**;3. **该节点在不在我够得着的分区**。
本轮只看第 1 样,连续误判两次:提交到 sablab-gpu 却发现 gpu10 只在 scu-gpu;
改到 scu-gpu 后才发现 gpu10 有 `prj0129` 保留(到 2026-09-26,用户名单不含我)⇒ 16 天都用不上。
`wait_and_launch.sh` 的注释里早写过 reservation 这条,但只用在**训练**派活上,**评测**派活没防。

**2026-09-10 补(同一坑的第四种形式:复用的是**数据**,不是代码)**:
为 Q-core 的方向对照登记了一个新探针(§4.517-P),打算等空卡去跑。
写完才发现 `outputs/qsig_kq*/gateQ1_probe.jsonl` **已经有 kq=1..9 共 9 格、每格 n=2304 的同一测量**,
其中还包含跨两种硬件、两个 seed 的三次复现。零 GPU 就把主结果拿到了(cos(d_clean,d_Q)≈0.08)。
**How to apply:** 登记新探针前,先 `grep -rl '<字段名>' outputs/` 找有没有已经落盘的同名指标;
本项目探针一律写 `outputs/<tag>/gate*_probe.jsonl`,字段名就是最好的检索键。

**2026-09-10 补(第五种形式:漏检既有**结论**,差点白跑 25 GPU-小时)**:
从探针数据推出"trust-region 半径过冲 ⇒ 登记新臂 qcorerho",launcher 都起好了,
才发现同类干预的**更强版本**(逐样本回溯)早已实现成 `qopsd` 臂跑完,
且 `outputs/TABLE_qopsd_vs_np3_ck100.txt` 显示官方 held-out **0/5 轴更好、2/5 轴更差**。
**How to apply:** 登记任何新臂前,两条固定检索:
`grep -n '<机制关键词>' EXPERIMENT_LOG.md`(找既有登记/判定)+ `ls outputs/TABLE_*`(找已落盘的对比表)。
探针上的正结果**不是**派活理由,先确认同一机制没有已跑完的 held-out 反例。

**2026-09-11 补(安全结论会随时间失效)**:§4.528 核过"提交到 sablab-gpu 不会抢占任何人"并据此升过优先级。
6 小时后想再用同一招,**重跑检查发现 gpu02/gpu04 上已各有一个别人的 QOS=low 作业** ⇒ 会被杀掉,遂放弃。
**How to apply:** "先证明没人会被杀"这类**关于集群瞬时状态**的安全结论**必须每次重跑**,
不能因为上次核过就沿用。核的是 `squeue -w <节点> -o %q` 里有没有会被自己 QOS 抢占的作业。

**2026-09-29 补(第六种形式:逐图配对键)**:判 OPSD 臂时为了"快"现写了一个 20 行配对脚本,
键用 `(prompt, seed)`。但 `matched_pi_*.json` 每个 prompt 有 **5 张图**、`seed` 字段整个文件同值
⇒ dict 静默覆盖,**n 从 1000 塌成 200,所有 t 低估 √5 倍**。我据此在 EXPERIMENT_LOG 写下过
`qstrain7−basenp3 t=+1.00`(真值 −0.93)、`qcore−basenp3 t=−4.59`(真值 −9.29),20 分钟后才发现并更正。
更糟的是同一轮里 `multik4` 的早读被算成 `+1.09×MDE t=+1.09`(不显著),真值是
**`+1.46×MDE t=+3.30`(显著)** —— 差点把本项目第一个正向结果当成噪声放过。

**正确键是 `r["index"]`**,而 `judge.py` 的 `load()`/`paired()` **一直就是对的**。
**How to apply:** 任何 OPSD 逐图配对一律 `import judge.py` 调 `J.load(file, axis)` + `J.paired(a, b)`,
不要另写。要批量对比就写个薄壳(scratchpad 里的 `pair.py` 就是:`importlib` 加载 judge,
存好 `sys.argv[1:]` 再把 `sys.argv` 清空以免触发它的 main)。
判定前先核 **n 是不是 1000** —— n=200 就是撞键的指纹。
