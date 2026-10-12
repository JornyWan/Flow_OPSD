# 0028 same-image 评测(先于数据写死判读)

## 为什么(codex 0024 §6 列为第一优先)

Self-OPD 论文的关键论点是:**task images 上的 preference** 与
**held-out DrawBench/Pick-a-Pic** 是两个不同问题 ——
它自己在 task images 上强过 DiffusionOPD,但在 held-out 上 Pick/HPS 反而低 0.56/0.0314。

我们现在只知道 `crg` 的 **DrawBench** HPS 明显下降(−13.4×2σ),
却**不知道它在自己正在保护的 OCR/GenEval task images 上质量如何**。
这两种情况的含义完全相反:

- 若只在 held-out 退化 ⇒ 守卫保住了它承诺保护的东西,代价落在分布外 —— 可接受且可陈述;
- 若连 task images 也退化 ⇒ 守卫在**它自己的地盘上**就损害了质量,主张要大幅收窄。

**这个诊断不需要新训练**,比继续研究 `G_fit` 更直接。

## 实现(最小改动)

原 `evaluation.py` 把 scorer 按数据集**写死**(geneval→只 geneval,ocr→只 ocr),
只有 drawbench 分支支持 `EVAL_REWARDS`。已把该覆盖**统一到所有数据集**:
不设环境变量时行为逐位不变,设了就完全按它来。

于是可以在**同一批生成图像**上同时算任务指标与 PickScore/HPS,
避免"每个指标各采一批图"引入的采样噪声 —— 这正是 codex 强调的点。

**作业**(tag 加 `SI` 后缀,不覆盖既有结果):

| tag | 数据集 | EVAL_REWARDS |
|---|---|---|
| `crdSI_181` | ocr | `ocr,pickscore,hpsv2` |
| `crdSI_181` | geneval | `geneval,pickscore,hpsv2` |
| `crgSI_181` | ocr | `ocr,pickscore,hpsv2` |
| `crgSI_181` | geneval | `geneval,pickscore,hpsv2` |

同 checkpoint(step 181)、同 guidance(4.5)、同样本数(256)、同硬件(A40)。

## 先于数据写死的判读

比较 `crgSI` 与 `crdSI` 在**同一批 task images** 上的 PickScore/HPS:

| 观测 | 结论 |
|---|---|
| task images 上 crg 的 Pick/HPS **不低于** crd | 守卫的质量代价**只在 held-out 分布上**;可主张"保住了承诺保护的能力,代价在分布外" |
| task images 上 crg 的 Pick/HPS **也明显低于** crd | 守卫在**自己的地盘上**就损害质量 ⇒ 主张必须收窄,并需解释为何只保护 KL 不足以保住感知质量 |
| 两者都低于发布 Student | 说明是"继续优化 PickScore"本身的代价,不是守卫独有 —— 需把 Student 的同批 same-image 分数一并报 |

**必须一并跑的第三个对照**:发布 Student 的 same-image 分数。
没有它,前两行都无法区分"守卫造成的"与"继续训练本身造成的"。
(Student 的 SI 作业待 A40 释放后补;当前 4 个作业已排在 preempt/A40 上。)

## 限度

- 256 个样本,单次生成,无独立 seed ⇒ 只作诊断,不作 paper-level 显著性;
- OCR/GenEval 的 prompt 分布与 DrawBench 不同,**不能**把两处的 Pick/HPS 直接横比,
  只能在**同一 prompt 集内**做 crg-vs-crd 的配对比较。

---

## 04:30 首轮作业 exit=0、有产出,但**算的不是我要的东西**

四个 SI 作业都 COMPLETED 并写出了 `average_scores.json`,内容却是:

```json
{"ocr": 0.9404, "avg": 0.9404}
```

**只有任务指标,没有 pickscore/hpsv2** —— 而这两个正是 same-image 诊断的全部目的。

**真因:SLURM 的 `--export` 把逗号当分隔符。**
我提交时写的是

    --export=ALL,SI_TSV=...,SI_DS=ocr,EVAL_REWARDS=ocr,pickscore,hpsv2

SLURM 解析成 `EVAL_REWARDS=ocr` 外加两个**未定义的变量名** `pickscore`、`hpsv2`。
日志里 `[same-image] EVAL_REWARDS 覆盖 -> ['ocr']` 就是铁证。

**修法**:先在提交环境里 `export EVAL_REWARDS="ocr,pickscore,hpsv2"`,
再用**纯 `--export=ALL`** 提交 —— 当前环境整体继承,值里的逗号不会被拆。

**这是今晚第 N 次"exit=0 且有产出 ≠ 算对了"**。
这次能发现,是因为我按预注册去**逐项核对要的字段是否存在**,
而不是看到 `average_scores.json` 存在就当成功。
`eval_status.sh` 的 3/3 判据也拦不住这一类 —— 文件确实写出来了,只是内容不对。

**补充的检查项(写进本预注册,后续 SI 作业必须过)**:
产出的 JSON 必须**同时包含**任务指标与 `pickscore`、`hpsv2` 三个键;
缺任何一个即视为该作业失败,不得进入分析。

---

## 05:10 结果出齐(crd/crg),但**缺第三个对照,现在还不能解读**

| prompt 集 | 指标 | crd(无保护) | crg(守卫) | 差 /2σ |
|---|---|---|---|---|
| **ocr** | ocr | 0.7338 | **0.9286** | **+4.87** |
| | pickscore | 0.9591 | 0.8672 | **−31.06** |
| | hpsv2 | 0.3099 | 0.2671 | **−19.46** |
| **geneval** | geneval | 0.6849 | **0.9521** | **+4.50** |
| | pickscore | 0.9531 | 0.8924 | **−20.51** |
| | hpsv2 | 0.3119 | 0.2761 | **−16.27** |
| held-out drawbench | pickscore | 0.9503 | 0.9369 | −4.52 |
| | hpsv2 | 0.3146 | 0.3132 | −0.64 |

**守卫在 task images 上的 Pick/HPS 比无保护臂差 20~31 个 2σ,远大于 held-out 的 −4.5/−0.6。**

### ⚠️ 但这**不能**直接读作"守卫损害了质量"

`crd` 正在**猛推 pickscore**,它在**所有** prompt 上的 pickscore 都会更高 ——
包括 OCR/GenEval 的 prompt。所以 `crg` 低于 `crd` 有两种完全不同的原因:

- (A) `crd` **涨上去了**,`crg` 只是没跟着涨(两者都 ≥ Student)⇒ 这是**预期的权衡**,不是损害;
- (B) `crg` **跌下去了**(低于 Student)⇒ 守卫在自己的地盘上**真的损害了质量**。

**判据必须是 `crg` 相对起点 Student,不是相对 `crd`。**
这正是本预注册开头写的"必须一并跑的第三个对照:发布 Student 的 same-image 分数;
没有它,前两行都无法区分"。

**已提交** `StudentSI` 的 ocr/geneval same-image 评测(job 1036918/1036919)。
**在它回来之前,上表只登记数据,不下任何结论。**

(顺带:held-out 与 task images 的 pickscore 绝对值差异很大 ——
crd 在 ocr prompt 上 0.9591、在 drawbench 上 0.9503 ——
所以**两处的分数不可横比**,只能各自在同一 prompt 集内做臂间比较。这也写在预注册限度里。)

---

## 🔴 05:40 我引入了一个回归,污染了 10 个评测(如实记录)

为做 same-image 诊断,我把 `EVAL_REWARDS` 的覆盖**从 drawbench 分支推广到所有数据集**。
当时判断"不设该变量时行为逐位不变",**这个判断是错的** ——
`eval_all.slurm` 与 `eval_opsdck.slurm` 的第 35 行本来就有

    export EVAL_REWARDS=pickscore,clipscore,aesthetic,hpsv2

那是**只给 drawbench 分支准备的**。我的改动让它对 ocr/geneval 也生效,
于是这两个模板派出的所有 ocr/geneval 评测**改算 drawbench 的 scorer,任务指标丢失**:

    mopdc_{31,61,91}_ocr      mopdc_{31,61,91,121}_geneval
    crkl5_31_ocr              crkl5_{31,61}_geneval

**正好是用户要求补的两条新基线臂。** 老的 crd/crg/crm 评测在改动之前完成,未受影响。

**发现方式**:`eval_status.sh` 报 3/3、文件也确实存在,但我按基线表逐轴取值时
`ocr`/`geneval` 取到 `None` —— 于是去看 JSON 的**实际键名**,才看到里面是 drawbench 的四个 scorer。
**"3/3 且文件非空"仍然不等于"算的是要的东西"。**

**修法**:改用**独立变量名** `SI_EVAL_REWARDS`,`EVAL_REWARDS` 恢复原语义(仅 drawbench)。
已删除 10 个被污染的评测目录并取消在飞的相关作业,由监视进程用修复版重派。

**教训**:改一个"看起来只是放宽"的开关前,要先 `grep` 全仓看**谁已经在设这个变量**。
我当时只检查了"不设时默认值不变",没检查"已经有人设了它"。

### 06:20 StudentSI 首轮失败:LoRA 路径多了一层

    FileNotFoundError: LoRA directory not found at .../Student/lora/lora

评测脚本会**自己追加 `/lora`**,所以 targets 表里应写**父目录** `.../DiffusionOPD/Student`,
而我写成了 `.../Student/lora`。项目里既有的 `StudentIR` / `StudentUNIrep*` 目标表用的都是父目录
—— 查一下就能避免。已按既有约定改正重交(job 1036937/1036938)。

**顺带确认了一件事**:此前担心"改了 slurm 脚本会不会影响已排队的作业"。
不会 —— SLURM 在 **submit 时快照脚本**,1036918 打印的 `rewards=ocr,pickscore,hpsv2`
就是用提交时那版(仍读 `EVAL_REWARDS`)跑出来的。所以我 05:40 的改动没有破坏在飞作业,
它们失败另有原因(路径)。这一点值得记住:**编辑 slurm 不影响已提交作业,但会影响之后的提交**。

---

# ✅ 定稿:第三对照到位,先前的"守卫损害质量"警报被推翻

`StudentSI` 的 ocr/geneval 两半均已完成。按预注册,判据是 **crg 相对起点 Student**,
不是相对 `crd`。

## 同一批 task images 上(256 张,step 181)

| prompt 集 | 指标 | Student | crd(无保护) | crg(守卫) | crd−Stu | **crg−Stu** |
|---|---|---|---|---|---|---|
| **ocr** | ocr | 0.9347 | 0.7338 | 0.9286 | **−5.02** | **−0.15** |
| | pickscore | 0.8672 | 0.9591 | 0.8672 | +31.03 | **−0.03** |
| | hpsv2 | 0.2701 | 0.3099 | 0.2671 | +18.08 | −1.38 |
| **geneval** | geneval | 0.9310 | 0.6849 | 0.9521 | **−4.14** | **+0.36** |
| | pickscore | 0.8923 | 0.9531 | 0.8924 | +20.54 | **+0.03** |
| | hpsv2 | 0.2784 | 0.3119 | 0.2761 | +15.26 | −1.00 |

**两个 prompt 集给出完全一致的结论**:守卫臂的 pickscore 与 Student **逐位相同**
(−0.03 / +0.03 ×2σ),hps 在噪声内(−1.38 / −1.00),任务指标保持(−0.15 / +0.36)。

⇒ 落在预注册判定表的**第一支**:
**守卫保住了它承诺保护的东西,质量代价不在 task images 上。**

先前(05:10)看到的"crg 比 crd 差 20~31 个 2σ"**完全来自 crd 涨上去**,不是 crg 跌下去。
**当时坚持不解读、先补第三对照,是对的** —— 只看 crd-vs-crg 会得出完全相反的结论。

## crd 那一列是教科书式的 reward hacking,而守卫阻止了它

无保护臂在**task images** 上:pickscore **+31.03 / +20.54**、hps **+18.08 / +15.26**,
而任务指标 **−5.02 / −4.14**。
**图更"好看"、偏好分都涨,但字是错的、物体是错的。**

这比"守卫保住了 OCR"强一档:**它保住的是"不要用画质换任务正确性"这件事**,
且有**同一批图像**上的直接证据 —— 不是两批图各测一个指标推出来的。

## held-out 对照给出完整图景

| DrawBench | Student | crd | crg | crd−Stu | crg−Stu |
|---|---|---|---|---|---|
| pickscore | 0.9222 | 0.9503 | 0.9369 | +9.47 | **+4.94** |
| hpsv2 | 0.3427 | 0.3146 | 0.3132 | −12.75 | **−13.38** |

- **task images**:三指标全部保持在 Student 水平;
- **held-out**:pickscore 上升(+4.94),hps 下降(−13.38)。

**HPS 的损失只发生在 held-out 分布上,不在被保护的 task 分布上。**
而由论文方法基线(`0030`)可知,该损失来自 **reward 优化本身**
(MOPD-continue 继续训练同等步数只掉 0.41×2σ),不是守卫造成的。

## 可主张的措辞(收窄且逐条有证据)

> 在从已蒸馏的多能力 Student 继续优化 PickScore 时,one-sided same-state retention 守卫
> 在**被保护的 task 分布上**同时保持了任务能力与感知质量(三指标相对 Student 均在噪声内),
> 并阻止了无保护 continuation 那种"提升偏好分、破坏任务正确性"的 reward hacking;
> 代价是 held-out 分布上的 HPS 下降,而该下降主要来自 reward 优化本身而非守卫。

## 限度

- 256 样本、单次生成、无独立 seed ⇒ **诊断级证据**,paper-level 需多 seed;
- OCR/GenEval 与 DrawBench 的 prompt 分布不同,**两处分数不可横比**,
  只能各自在同一 prompt 集内做臂间比较;
- `crkl5`/`mopdc` 的 same-image 分数尚未测 —— 若要在论文里写这条,
  应把这两条基线也纳入同一批 same-image 评测。

---

# ✅ 四臂 × 两个 prompt 集全齐(2026-08-30 上午)

## task images(同批 256 张,相对起点 Student,单位 = 该轴重测 2σ)

| 臂 | ocr | pick | hps | ‖ | geneval | pick | hps |
|---|---|---|---|---|---|---|---|
| MOPD-continue(论文方法) | +0.04 | −1.14 | −1.07 | ‖ | +0.36 | −0.46 | −1.58 |
| OPSD-direct(无保护) | **−5.02** | **+31.03** | **+18.08** | ‖ | **−4.14** | **+20.54** | **+15.26** |
| 朴素 frozen-Student KL | +0.37 | −2.13 | −3.37 | ‖ | +0.56 | −0.96 | −1.48 |
| **one-sided hinge (w=10)** | **−0.15** | **−0.03** | −1.38 | ‖ | **+0.36** | **+0.03** | **−1.00** |

## held-out DrawBench(相对 Student)

| 臂 | pick | hps |
|---|---|---|
| MOPD-continue(论文方法) | −0.15 | **+0.08** |
| OPSD-direct(无保护) | +9.47 | −12.75 |
| 朴素 KL | −0.58 | **−18.65** |
| **one-sided hinge (w=10)** | **+4.94** | −13.38 |

## 三条结论

**① 守卫在 task images 上与论文方法基线几乎不可区分,但在 held-out 上多拿 pick +4.94。**
三项差值全部落在 ±1.4 个 2σ 内(ocr +0.04 vs −0.15、pick −1.14 vs −0.03、hps −1.07 vs −1.38)。
⇒ **在被保护的分布上,守卫把模型维持在"等同于继续跑论文方法"的状态,同时在分布外取得了 reward 进展。**

**② 无保护臂是教科书式 reward hacking,两个 prompt 集一致。**
task images 上偏好分 **+31.03 / +20.54**、hps **+18.08 / +15.26**,
而**任务指标 −5.02 / −4.14** —— 图更好看、但字错了/物体错了。
守卫与论文方法都没有发生这种交换。

**③ 朴素 KL 付了代价却没换到东西。**
task images 上 pick **−2.13 / −0.96**、held-out pick **−0.58**,三处都是负的;
而 held-out hps **−18.65**(四臂最差)。
⇒ 与六点主表的"守卫严格支配朴素 KL"一致,且在 same-image 上再次复现。

## 🔴 对我先前一句陈述的更正

我在 §0 曾写"**HPS 的损失只发生在分布外**"。**这句太绝对。**
准确的是:**hps 在 task images 上也有小幅下降**(hinge −1.38 / −1.00、朴素 KL −3.37 / −1.48),
只是比 held-out(−13.38 / −18.65)**小一个数量级**。

正确措辞应为:
> HPS 的损失**主要**发生在 held-out 分布上;在被保护的 task 分布上仅有约 1 个 2σ 的下降,
> 与论文方法基线(−1.07 / −1.58)处于同一水平。

**"只在分布外"与"主要在分布外、task 上也有约 1σ"是不同的主张**,后者才是数据支持的。
