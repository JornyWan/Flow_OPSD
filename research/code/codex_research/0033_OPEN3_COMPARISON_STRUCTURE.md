# 0033 OPSD 论文的 Open3 comparison 是什么,以及我们缺什么(2026-08-30)

用户要求"训 OPSD 的 multi-reward 最终版,而不是单 reward 的"。照做时查清了
论文 baseline 的真实结构 —— 这改变了对"OPSD 复现不完整"的理解。

## ① 论文其实有两套互不相同的结果

| | Main Results 表(README L133-146) | Open3 comparison(`opd/README.md`) |
|---|---|---|
| checkpoint | **每列一个 reward 专训的** | **一个 checkpoint 打全部三列** |
| 原文措辞 | "Reward-specific checkpoints are evaluated on their **matched** held-out objective" | "three teacher-distillation baselines reported in the paper's **SD3.5-M Open3 comparison**" |
| OPSD 行 | Pick 24.94 / CLIP 0.340 / HPS 0.390 / Aes 12.08 / ImgR 1.76(**100 updates**) | **Pick 25.51 / CLIP 0.333 / HPS 0.389**(**300 updates**,见下 §⑭更正) |

**⚠️ 本节原写"Open3 没有可对标的 published 数字" —— 该结论错误,已由 §⑭ 更正。**
数字在**论文正文**(`sections/experiments.tex` L46),不在 README。我当时只查了 README。

## ② Open3 comparison 的完整臂集(= 我们要用的论文 baseline)

论文在这个对比里放的是:

1. **DiffusionOPSD-Open3** —— 直接用 joint 目标训(`sd35_open3`)。
2. **三条 teacher-distillation baseline**(`opd/`):**DanceOPD / DiffusionOPD / FlowOPD**。
   三条都把**三个冻结的 OPSD 专家**(pickscore / clipscore / hpsv2 的 **checkpoint-100** LoRA)
   按 1:1:1 蒸馏进**同一个 student**,300 updates,reward model 不进第二阶段 loss。

**关键认识:DiffusionOPD 不是"另一篇论文",它是 OPSD 论文 Open3 对比里的一条 baseline。**
我们一直当起点用的那个发布版 Student,就是这条臂的产物。
⇒ 我们和论文的接口比原先以为的紧密得多。

各方法的目标(`opd/README.md`):
- DanceOPD:在**一个**低噪 query state 上,三教师 velocity-MSE 等权平均。
- DiffusionOPD:在**全部 10 步**上,三教师 transition-mean MSE 等权平均。
- FlowOPD:随机 SDE rollout,教师 transition log-prob 做 per-step reward + clipped PPO。

## ③ 我们现在缺的,和为什么它一举两得

已有:`pickscore` 专家(`outputs/opsd_repro_sd35_pickscore/checkpoint-100`)。
缺:**`clipscore` 与 `hpsv2` 专家**。

补上它们同时解决两件事:
1. OPSD Main Results 复现从 **1/5 列 → 3/5 列**(补 CLIP 0.340、HPS 0.390);
2. 凑齐 Open3 comparison 需要的**三个教师**,使 DanceOPD/FlowOPD 这两条
   论文 baseline 变得**可训**(DiffusionOPD 那条我们已有官方发布权重且 8/8 列吻合)。

## ④ 本次已启动(2026-08-30 12:17 EDT)

| job | 臂 | config | 预算 | 资源 | 状态 |
|---|---|---|---|---|---|
| 1037080 | **OPSD-Open3** | `public.py:sd35_open3` | 300 upd, β=0.1 | 3×H200 | RUNNING,已进 epoch 1 |
| 1037094 | OPSD-clipscore 专家 | `public.py:sd35_clipscore` | 100 upd, β=1.0 | 3×H200 | RUNNING(ai-gpu17) |
| 1037095 | OPSD-hpsv2 专家 | `public.py:sd35_hpsv2` | 100 upd, β=1.0 | 3×H200 | PENDING(afterany:1037094) |

> 1037081/1037082 是这两个臂的**第一次**提交,已取消重投,原因见下。
> 三个专家臂(pickscore 历史跑 / clipscore / hpsv2)**world size 均为 3**,配置一致。

**SLURM gres 降级(本次踩了三回,记法要改):**
`scontrol show job` 的 `TresPerNode` 是**申请**,`AllocTRES` 才是**实际授予**,两者会不一致。
1037079 与 1037081 都申请 h200:3 却只拿到 `AllocTRES=...h200=2`,作业内 `nvidia-smi` 只见 2 张,
被脚本守卫拦下 requeue —— 守卫行为正确,但它会**占着那 2 张卡空睡 900s** 再 requeue,
最坏 40 轮 ≈ 10h 双卡空转。改为申请 `h200:4`(4 也是合法分组值)后,
SLURM 无法把它塞进只剩 2 张的节点;实测 1037094 申请 4 被降级为 3 —— **3 仍合法,直接跑起来**。
⇒ 判据一律看 `AllocTRES`,不看 `TresPerNode`,更不看 squeue 的 `%b`(会截断)。

运行时已核对 1037080 真的在跑 Open3,不是"没崩就算对":
`[OPA] opa=1 reward=open3 dir_mode=grad dual_neg=1 rho=0.1 margin=0.005 n_ascent=2 query_sigma=0.278 mb=6`
且 `reward_fn={clipscore:1, hpsv2:1, pickscore:1}`、`beta=0.1`、batch 8×3×48=1152/epoch(论文 48×24 分组)。

## ⑤ 修正 0032 的一处错误记录

0032 写 ImageReward "⛔ 缺 openai clip 包" —— **不成立**。
实测 env 里 `clip` / `ImageReward` / `open_clip` 三个包都能 import,
且 OPSD 的 scorer 能实例化成功(`ImageRewardScorer(dtype=float32, device='cpu')` → `SCORER_OK`,
权重走本地 `~/.cache/huggingface/ImageReward/ImageReward.pt`,无需联网)。
ImgRwd 这一列**没有技术障碍**,只是还没排评测。

## ⑥ 顺带修掉的一个真实缺口

OPSD repo **根本没有 `reward_ckpts/` 目录** —— 此前只跑 pickscore,而 pickscore 走 HF hub,
所以从没触发。open3 需要 HPS 的 `open_clip_pytorch_model.bin`(ViT-H-14, 3.94GB)与
`HPS_v2.1_compressed.pt`,`check_reward_setup.py --reward open3` 直接报缺失。
已按项目既有惯例(0015 slurm 也这么做)软链到 DiffusionOPD 侧已有的同名目录:
`DiffusionOPSD-main/reward_ckpts -> DiffusionOPD-main/reward_ckpts`。
运行日志确认三个 rank 都从该路径加载了 ViT-H-14。

## ⑦ 运行时实测(不只看 exit=0)

指标由 `install_wandb_jsonl_tee` 落到 `outputs/opsd_repro_sd35_open3/metrics.jsonl`
(stdout 只有 `Advantages mean`,不足以判定三路是否都活)。

| epoch | reward_clipscore | reward_hpsv2 | reward_pickscore | reward_avg |
|---:|---:|---:|---:|---:|
| 0 | 0.2417 | 0.2233 | 0.7622 | 1.2272 |
| 3 | 0.2663 | 0.3061 | 0.8301 | **1.4026** |

- `reward_avg` 恰等于三者之和 ⇒ **1:1:1 聚合已验证生效**;
  `reward_pickscore≈0.83` 是 /26 归一口径(README 写明 "paper-matched PickScore /26")。
- **三路同步上升**,没有"一路涨、两路塌"的多目标退化。
- 节奏 ~3.6 min/epoch(12:20:39→12:24:15→12:27:50)⇒ 300 updates ≈ **18h**,48h 墙钟充裕。

## ⑧ 一处必须讲清的对齐问题(不要含糊)

**Open3 臂从 base SD3.5-M 出发**(`lora_path=None`),论文的 Open3 对比里
DanceOPD/DiffusionOPD/FlowOPD 三条 baseline 也都是从 base 训一个新 LoRA student。
**而我们自己的 CA-OPD/RTC 各臂是从发布版 Student 继续训的。**

⇒ Open3 臂**不是**我们各臂的同起点对照,它是"论文对同一多目标问题的另一条技术路线"。
写论文时两者要分开摆:
- 同起点内部对照(自 Student):`mopdc` / `crd` / `crkl5` / guard 等;
- 论文方法对照(自 base):OPSD-Open3、以及三条 teacher-distillation baseline。
把它们混进一张表并按行比大小是错的。

## ⑨ 排期与自动化

- 已建 `codex_research/watch_opsd_arm_eval.sh`:等训练臂结束 → 自动按 **OPSD 原生协议**
  (`cross_eval.py`, DrawBench 1000, flow 40 步, gs=1.0, seed 42)排 A40 评测。
  注意这与 `auto_eval_on_finish.sh` 的 DiffusionOPD 协议(gs=4.5, n=256, 三数据集)**不是一回事**,
  与论文主表对标只能用前者。三个臂的监视进程均已挂上。
- `slurm/opsd_eval.slurm` 改了两处:
  (a) 输出名加 `ARM` 标识 —— 原来 `opsd_eval_ck100_raw.json` 不含臂名,
      不同臂的同号 checkpoint 会命中"已存在→跳过",作业 exit=0 却没产出新数据;
  (b) 新增 `REWARDS` 透传 `--rewards` —— 官方默认集 `DEFAULT_Y_REWARDS` 不含 imagereward。
- 正在跑一次自检:对 **pickscore 臂 ck100** 用 5 scorer 重评(job 1037093, A40)。
  其中 4 个公共列应与既有 `opsd_eval_ck100_raw.json` 吻合;若吻合即证明上述改动
  没有动到评测路径,imagereward 是干净的增量。**吻合与否会如实记录。**

## ⑩ 一个**不能**当健康信号用的指标(以及两个能用的)

`metrics.jsonl` 里有 `opa_acc_plus` / `opa_acc_minus`(目标采纳率)。看着像"构造的目标是否被接受"
的理想判据,但对照**已跑完的 pickscore 臂 102 条记录:两者全部恒为 1.0**。
⇒ 它是官方实现的**常量**,不随训练/reward/配置变化,**不含任何信息**。
引用它说"目标被采纳了 ⇒ 跑对了"是自欺 —— 与之前 `cr_active_frac ≡ 1.0` 是同一类错误。

**能用的判据(open3 实测):**
- `reward_clipscore` / `reward_hpsv2` / `reward_pickscore` 各自的轨迹(会动、方向可判);
- `zero_std_ratio = 0.0` —— 没有出现"组内 reward 标准差为 0"导致 advantage 归零的退化组;
- `opa_delta_res ≈ 0.071` —— 非零残差,说明构造步确实在改动目标。

clipscore 臂(单 reward)同样非退化:`reward_clipscore` 0.2465 → 0.2528(epoch 0→1)。

**顺带第 4 次栽在 NFS 属性缓存上**:`[ -f metrics.jsonl ]` 在文件已写入后仍返回假,
读任何计算节点刚写的文件前必须先 `ls` 其父目录。

## ⑪ 与 codex `0028a_METHOD_GENEALOGY_AND_STARTING_POINTS` 的关系

codex 同日(12:22)独立写了同一主题,**结论一致、内容互补,没有冲突**:

- codex 那份确认了 `base → OPSD specialists → frozen teachers` 与
  `base → shared student + frozen teachers → DanceOPD/DiffusionOPD/FlowOPD`,
  并明确否定了"DiffusionOPD → DiffusionOPSD"的误读 —— 与本文 §② 一致;
- codex 多给了**命名规范**(主表每行写全 `initialization → method`,
  官方点估计单列 `Reported original-paper results` 只作背景,不进同一 ranking)
  —— 与本文 §⑧ 是同一要求的两种表述,**采用 codex 的表格命名**;
- codex 多给了一条我们尚未考虑的近邻 baseline:**Self-OPD**(arXiv 2608.26872),
  且指出其 mixed-reward 版存在 from-scratch / warm-start 两条路径,
  v1 论文 Table 2 未标注最终点来自哪条 ⇒ **复现主表时不能擅自假设**;
- 本文多给了 codex 未涉及的两点:(a) 论文**两套结果口径**的区分,
  (b) ~~Open3 没有 published 数值~~ **该判断错误,见 §⑭** —— 数值在论文正文而非 README。

**索引撞号已处理:** 我 11:28 建的 `0028_SAMEIMAGE_EVAL_PREREG.md` 与 codex 12:22 建的
`0028_METHOD_GENEALOGY_AND_STARTING_POINTS.md` 同号。不改文件名(会断双向引用),
只在 `INDEX.md` 里以 `0028a` / `0028b` + 文件名后缀消歧。

## ⑫ OPSD 原生协议下的锚点表(2026-08-30 13:19 实测)

协议:DrawBench 1000 条、flow 40 步、gs=1.0、512²、seed 42、`pickscore_scale=raw`。

| 模型 | Pick | CLIP | HPSv2.1 | Aes | ImgR |
|---|---:|---:|---:|---:|---:|
| **base SD3.5-M(无 LoRA)** | 20.5868 | 0.2408 | 0.2070 | 5.1248 | −0.5481 |
| 论文 base(`main_results.tex`) | 20.51 | 0.237 | 0.204 | 5.13 | −0.58 |
| DiffusionOPD Student(发布版) | 21.5644 | 0.2570 | 0.2512 | 5.9508 | 0.3355 |
| OPSD-pickscore 专家 ck100 | 25.2133 | 0.2548 | 0.3108 | 6.2761 | 1.1654 |
| 论文 OPSD(reward 专训,每列不同 ckpt) | 24.94 | 0.340 | 0.390 | 12.08 | 1.76 |

### base 行验证了整条协议链

**五列全部对上**:相对差 Pick +0.37% / CLIP +1.60% / HPS +1.47% / Aes **−0.10%** /
ImgR +0.032(绝对,分母近零不看相对)。这不依赖任何训练 ——
采样器、步数、gs、分辨率、prompt 集、seed、五个 scorer 的尺度**全部正确**才可能同时落在这个范围。

**诚实标注:** CLIP 与 HPS 差约 1.5%(+2.4 / +2.5 个我们自己的 SE),比项目既有的
"重测 2σ = 0.61%"大。但那个 0.61% 是**六轴聚合比值**的口径,不是单列;单列重测方差本就更大,
且我们与论文是**不同硬件**(fp16 kernel 经 40 步采样必然分岔)。
⇒ 结论写成"**在预期范围内吻合,协议链无误**",**不写成"复现到位"**。

### 一个看起来很强、但不能这样写的对比

OPSD-pickscore ck100 相对 DiffusionOPD Student:**4/5 列更高**
(Pick +3.65、HPS +0.060、Aes +0.325、ImgR +0.830;**CLIP −0.0021 是唯一更低的**)。

**不能据此说"OPSD 打败 DiffusionOPD"**,理由:
1. 这套协议**只测 reward 类指标**,而 Student 的强项(GenEval / OCR 的构图与文字准确率)
   **在这张表里一个都没测**。用只测对方弱项的尺子量,赢了不说明问题。
2. Student 的训练目标是多能力保持,不是抬高这五个 reward;
   OPSD-pickscore 则是**直接把 pickscore 当 reward 优化**,Pick 那一列本就该赢。
3. 唯一有信息量的是 **CLIP 那一列 Student 反而更高** —— 说明 reward 专训并非全面占优。

正确用法:这张表是 **Open3 对比的锚点**(该对比中各臂同样从 base 出发、同样用这套协议),
等 Open3 臂(1037080)与 clip/hps 两个教师臂跑完后填入,再做同协议的横向比较。

## ⑬ 到早晨的排期(2026-08-30 13:21 起算)

实测节奏:open3 **3.76 min/epoch**、clipscore **2.7 min/epoch**。

| 时刻 | 事件 |
|---|---|
| ~17:05 | clipscore 专家(100 upd)完成 → 监视自动派其评测 |
| ~17:05 | hpsv2 专家(1037095)解除依赖开始训练 |
| ~18:33 | open3 **checkpoint-100** 落盘 → 立即评测(约 19:15 出数) |
| ~21:35 | hpsv2 专家完成 → 自动评测 |
| ~00:47 | open3 **checkpoint-200** → 评测(约 01:30 出数) |
| ~07:05 | open3 训练结束,**checkpoint-300** → 评测(约 07:50 出数) |

**改用增量监视的理由:** 原 `watch_opsd_arm_eval.sh` 等训练**整体结束**才派评测,
open3 要跑 19h,之后再串行评 3 个点(41 分钟/点)= 再等 2h,早上看不到数。
新的 `watch_ckpt_eval.sh` 改成**每个 checkpoint 一落盘就派该点评测**,
ck100 / ck200 在夜里就出数,早晨只差 ck300。
(轮询里每次都先 `ls` 父目录刷 NFS 缓存 —— 不刷会出现"文件已写入但判为不存在",本项目已栽 4 次。)

**顺带补齐 pickscore 臂的 5 列轨迹**(ck20/40/60/80,等间距**预先指定**,不是事后挑点),
用 17:05 前的空闲 A40。目的是看 ImgR 在 pickscore 专训过程中的走势,
以及给唯一完整复现的那条臂一条完整曲线。


---

## ⑭ 【更正】Open3 **有** published 数值,在论文正文里(2026-08-30 深夜)

**本文 §① 与 §⑪ 原写"Open3 只有图、无数值表、没有可对标的 published 数字"。这是错的。**

我当时只查了 README(`assets/robustness.png` 确实只是图),**没有查 arXiv 源码的正文**。
数值在 `arXiv-2608.24646v1/sections/experiments.tex` L46:

> "The jointly trained **300-update** \name{} policy obtains PickScore **25.51**,
> CLIPScore **0.333**, and HPSv2.1 **0.389**, compared with 23.62, 0.294, and 0.340
> for DiffusionNFT. ... Compared with separate **100-update** \namew{} specialists at
> 24.94, 0.340, and 0.390, the shared policy retains **113.0%, 92.9%, 99.6%** of their
> gains over the base model."

### 由此确定的两件事

| 项 | 值 |
|---|---|
| **论文 Open3(300 updates)** | Pick **25.51** / CLIP **0.333** / HPS **0.389** |
| **论文 reward-specific 专家的预算** | **100 optimizer updates**(与我们同预算,明确) |

⇒ **"复现 Open3"是可以谈的**,只要跑满 300 updates 并按这三个数对表。
⇒ **训练步数**不是 clipscore 差距的解释 —— 论文专家也是 100 updates。

### 同一段还给出一个与我们观察一致的事实

> "All three rewards continue improving through update 300"

**与我们实测的 ck100 未饱和(四轴仍在 +6~21×2σ 上升,0045 §⑥)一致。**

### 教训

**"README 里没有"不等于"论文里没有"。** 涉及"某个数字是否存在"的判断,
必须查 arXiv 源码正文,不能只查代码仓库的 README。
本文 §① 的结论建立在只查 README 之上,导致一个明确的错误结论流传了半天,
并被 0040/0045 等下游文档引用。**下游引用已一并更正。**
