---
name: cluster-usage-conventions
description: "SLURM cluster layout, GPU partitions/QoS, conda env conventions, and storage/dataset locations for chw4026's server"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 2357a1e3-c947-4954-8a20-1e05037f0663
---

SLURM cluster, 21 GPU nodes `ai-gpu01..21`, **4 GPUs max per NODE but multi-node jobs ARE allowed** (`--nodes=2 --gres=gpu:a100:4` = 8 GPUs passes `--test-only`; 3 nodes/12 GPUs too; no account-level MaxTRES cap) — the old "max 4 GPUs per job" note was wrong. All partitions 72h walltime. Live cheat-sheet: `project3/research_hub/00_server_and_env.md` (re-verified 2026-08-23). `med_vicl/CLAUDE.md` is behavioral guidelines, NOT a server doc — the server tables are in `med_vicl/CLAUDE_project_memory.md` §2 (also partly stale).

- **三个分区都可用(用户 2026-08-31 明确)**,但**账户/QoS 互不兼容,不能用一次 `--partition=a,b,c` 跨提交**,
  必须分别提交(实测 `scontrol show partition`):

  | 分区 | AllowAccounts | AllowQos | 节点 | 其中 ≥80GB 的 |
  |---|---|---|---|---|
  | `sablab-gpu` | sablab,scu | **仅 `sablab-high`** | gpu01-04,06-09,17-18 | a100 06-09 / h200 17-18 |
  | `minilab-gpu` | scu,minilab,paetzollab | high,normal | gpu12-14,17-18 | **h100 14** / h200 17-18 |
  | `scu-gpu` | **ALL** | high,normal | gpu10-11,19-20 | **a100 10-11** |

  我们的关联:`sablab`(含 sablab-high)与 `minilab`(high/low/normal)。
  ⇒ sablab-gpu 用 `--account=sablab`(qos 留空或 sablab-high);
    minilab-gpu 用 `--account=minilab --qos=high`;scu-gpu 用 `--account=minilab --qos=high`。
  **三者合计的 ≥80GB 节点:a100 gpu06-11、h100 gpu14、h200 gpu17-18(共 8 个)。**
  只盯 sablab-gpu 会白白放弃 gpu10/11/14 —— 2026-08-30 夜就因此让 A/B 干等了数小时。

- **Partitions (account / qos to use):** `sablab-gpu` (sablab, omit --qos → sablab-high; a40 gpu01-04, **a100 gpu06-09**, **h200 gpu17-18**) = first choice. `minilab-gpu` (minilab, qos high; l40 gpu12-13, h100 gpu14, h200 gpu17-18). `scu-gpu` (AllowAccounts=ALL → submit with sablab/minilab + qos=high; a100 gpu10-11, l40 gpu19-20) — **non-preemptible, no longer needs preempt_gpu**. `preempt_gpu` (**requires explicit `--qos=low`** or you get "Invalid qos specification"). `sablab-gpu-low` (qos=sablab-low, prio 10).
- **`ai-gpu10` 看着有空卡但拿不到(至 2026-09-08)**:预留 `prj0129`
  (2026-08-18 → **2026-09-08 23:00**)独占该节点 **全部 128 CPU**(`TRES=cpu=128`,
  `Flags=IGNORE_JOBS,SPEC_NODES`),用户名单里没有 chw4026。
  于是 `sinfo` 会显示 `gpu:a100:1(IDX:0)` 即 3 张空闲,实际提交必得
  `Reason=ReqNodeNotAvail,_May_be_reserved_for_other_job`。
  **⇒ 统计"可用 A100"时要把 gpu10 整个排除**;2026-08-31 为此白提交过一次。
  过了 09-08 需重新 `scontrol show reservation` 确认是否续期。
- **提交"部分节点"作业(不带 `--exclusive`)前,必须先查全集群任一节点的最大空闲卡数。**
  2026-09-01 实测教训:全集群每节点最多空 2 张,我却提交了 `--gres=gpu:h200:3`(无 `--exclusive`)。
  SLURM **照样把作业调度上去并只给 2 张**(`AllocTRES=...h200=2`),于是作业内的
  `EXPECT_NPROC` 守卫每轮拦下 → `scontrol requeue` → 再被调上去 → **无限循环,
  每轮各占 2 张卡空转**,既浪费又挡住别人。
  **正确做法**:先
  `for n in ...; do scontrol show node $n | grep AllocTRES; done` 求最大空闲数;
  凑不出就**别提交部分节点作业**,或加 `--exclusive`(要整节点,SLURM 不会半给)。
  **2026-09-03 又栽了一次**(inner4 申请 h200:3、只拿到 2,循环 24 轮才被发现)。
  两条补充:
  (a) **守卫的 requeue 上限要小**。`ab_open3.slurm` 原设 40 次 × sleep 300 = **3.3 小时空转**,
      而且每轮都**持有**那几张卡挡着别人。已收到 6 次(30 分钟),到顶就 `exit 1` 释放显卡。
  (b) **`--exclusive` 不一定是对的解**:它要求**整节点空闲**。实测 h200 上
      `--exclusive` 排到次日 13:46,而不带它的 `h200:3` 当晚 23:40 就能起 —— 差 14 小时。
      更快且同样安全的写法是**挂依赖**:`--dependency=afterany:<自己占着卡的作业>`
      + `--nodelist=<该节点>`,自己的作业一结束就精确接上,不存在部分授予的窗口。
  (c) **依赖也不够**:2026-09-04 实测,`--dependency=afterany:qcore` + `--nodelist=ai-gpu17`,
      依赖满足那一刻 gpu17 已被别人 4 个单卡作业占满 —— 依赖只保证"我的卡释放",
      保证不了"别人不抢"。
  (d) **⭐ 目前最可靠的写法:申请整节点的卡数(4),不加 --exclusive。**
      同一时刻实测排期:`h200:3`(非 exclusive)会被部分授予;`h200:3 --exclusive` 排到 09-06;
      而 **`h200:4` 非 exclusive 排到当天 10:42**。要 4 张就等于要整节点的 GPU,
      SLURM 不会半给,又不像 `--exclusive` 那样还要等 CPU/内存也全空。
      ⇒ **优先把臂设计成 nproc=4**;只有必须保持 nproc=3(如续跑已有 np3 run)时才用 `--exclusive`。
      注:`public.py` 的 48×24 分组只允许 nproc ∈ {3,4,6,8},且 B_eff=1152 与 nproc 无关,
      实测 np3 与 np4 五轴全打平(配对 t=−0.58),所以换 nproc 不影响可比性。
- **OPSD(DiffusionOPSD-rpi)训练在 80GB 卡上贴边,带探针只能上 H200(141GB)。**
  2026-09-01 实测 ai-gpu11(A100 80GB):训练基线已占 **79.21/79.25 GiB**,
  RTC 探针多要 32MiB 就 `OutOfMemoryError`。探针的 `except` 会吞掉异常保证训练继续,
  于是**作业正常 exit=0 但指标一条不产** —— 只看退出码会误判为"跑成了"。
  ⇒ (a) 探针作业一律投 H200(gpu17/18);(b) 监视脚本必须把"零记录"当失败处理。
  H100(gpu14/16)也是 80GB,同样不够。
- **⚠️ 不要原地重写正在被长跑进程执行的 shell 脚本。**
  bash 是**按文件偏移增量读取**脚本的,`open(p,"w")` 原地截断会让运行中的实例
  从错位的字节处继续读 ⇒ 执行出乱码/半截命令。
  2026-09-01 改 `auto_table_on_finish.sh` 时有 4 个监视进程正在跑它,已按精确 PID 停掉重挂。
  ⇒ **在 NFS 上"写临时文件后 `mv`"同样不安全** —— 2026-09-09 实测:`mv` 之后旧 inode 被 unlink,
  运行中的 bash 再去读脚本的后续命令时得到 `error reading input file: Stale file handle`,**exit 2**
  (baseH100 跑满 6 小时、产物齐全,只因这个而被 sacct 记成 FAILED)。
  "运行中的实例保留旧 inode" **只对本地文件系统成立**。
  ⇒ **正确做法:有实例在跑时根本不要动那个文件**(原地写 = 读到错位内容;`mv` = ESTALE),
  要加新内容就**用一个新文件名**提交新作业,老实例继续用老文件,之后择机合并。
  副作用:**退出码不再可信**。判完整性要用 epoch 覆盖无缺口 + ck 序列齐 + 末个 ck 的 lora 权重存在。
  **停进程时,"用精确 PID"还不够 —— 取 PID 的那条 `ps | grep` 本身也会匹配到自己的 shell。**
  2026-08-31 用 `pkill -f <脚本名>` 杀掉过自己的 shell;
  2026-09-01 改用 `ps -o pid=,cmd= | grep "[a]uto_table..." | awk '{print $1}'` **又杀了一次**
  —— 因为 shell 快照的命令行里也含该字符串(显示为 `arm=source`)。
  ⇒ **安全写法**:取 PID 后按"命令行是否以脚本绝对路径开头"二次过滤,例如
  `ps -o pid=,args= -u $USER | awk -v p="$SCRIPT" '$2=="/bin/bash" && $3==p {print $1}'`;
  或者干脆**先确认要不要停** —— 多数情况下"改完只影响以后新起的实例"就够了,不必重启。
- **判断"要等多久"不能只看占用作业的 `TimeLeft`,还要看它们能否被抢占。**
  2026-09-01:三个 A40 节点全满、最短 `TimeLeft=19:45`,我据此说"评测要等 20 小时";
  但占用者 lus4017 的 5 个作业都在 **`preempt_gpu`/QOS=low**,可被普通 QOS 抢占 ——
  `sbatch --test-only` 实测 `sablab-gpu` 路径会 `Preempts: 1038525`,
  `sablab-gpu-low` 甚至报"40 分钟后可起"。**差点把 40 分钟说成 20 小时。**
  ⇒ 查 `scontrol show job <占用者>` 的 `Partition`/`QOS`,再用 `--test-only` 实测。
- **`squeue`/`scontrol` 的 `StartTime` 估计是按前序作业的 `TimeLimit` 算的,不是按实际进度。**
  2026-08-31 实测:certp 的 `est_start=2026-09-01T11:17:39` **恰好等于**占着同一节点的 n2 的
  `EndTime`(= 提交时刻 + 24h 时限),而 n2 实际再跑 15 分钟就结束。
  ⇒ **看到"要等到明天/三天后"先别急着改方案**,先比对
  `scontrol show job <占用作业>` 的 `RunTime` vs `TimeLimit` 与真实进度;
  前序作业一结束,估计会立刻坍缩。差点为此白做一个单卡变通方案。
- **`scu-gpu` is the cluster DEFAULT partition** — always write `--partition` explicitly or jobs silently land there.
- **Every partition is `PreemptMode=CANCEL`** — preemption kills the job, so `--requeue` + checkpoint-resume is mandatory on preempt_gpu.
- **GPU inventory:** H200-141GB ×9 (gpu17,18 ×4 each; gpu15 ×1 admin-only), A100-80GB ×24 (gpu06-09 sablab, gpu10-11 scu), H100-80GB ×8 (gpu14 minilab, gpu16 yiwanglab), L40-48GB ×18, A40-48GB ×20. H200 nodes have 256 cores; the rest 128. Verify free cards with `sinfo -N -p admin -O "NodeHost:10,Gres:14,GresUsed:20,StateLong:12" | sort -u`, and dry-run with `sbatch --test-only`.
- **Quirks learned the hard way:** SLURM 会**超额分配** gres:2026-08-31 实测 ai-gpu14(物理 4 卡)上同时挂着 我们的 `gres/gpu:3` + 另一用户的 2 个 `gres/gpu:1` = **5 > 4**,于是我们的作业只看到 2 张。此前记作"降级(ask 4, get 3)"其实是同一现象。⇒ **必须在作业内用 `nvidia-smi -L | wc -l` 核实并 requeue**;且若某节点长期被小作业占住 N 张,大作业永远排不进去 —— 要把该节点 `--exclude` 掉,否则每轮 requeue 都白占那几张卡(既浪费又影响别人)。 **从作业外部判定要比对 `scontrol show job` 的 `AllocTRES=...gres/gpu:<type>=N`(实际授予) 与 `TresPerNode=gres/gpu:<type>:M`(申请);只看 TresPerNode 或 squeue 的 %b 会被截断/误导。** 2026-08-30 实测:同时提交的两个 h200:3 作业,一个 AllocTRES=3(正常)、一个 AllocTRES=2(被降级),降级发生在节点已满却仍被调度上去时。 A requeue TERM-trap eats scancel's first signal → `scancel` twice. `torch.compile` inductor thread-pool breaks on DataLoader fork → `TORCHINDUCTOR_COMPILE_THREADS=1`. Check `scontrol show reservation` before trusting an idle node.
- **Conda:** `~/.condarc` points envs_dirs/pkgs_dirs at `projects/side_project/generative/conda_envs/envs|pkgs`, so `conda create -n X` lands there automatically. Activate by absolute path, or `export PATH="$ENV/bin:$PATH"; export CONDA_PREFIX="$ENV"` (maintenance-safe). Nothing goes in `$HOME`. ~~Login node has internet; compute nodes are offline~~ → **2026-08-28 实测推翻**：CPU 节点(ai-cpu01)与 GPU 节点(ai-gpu16)**都有外网**(api.wandb.ai TCP 443 可达, 0.067s)，wandb online 已验证可用(entity `jorny-cornell-university`)。`HF_HUB_OFFLINE=1` 仍保留(权重已预下载，避免运行时拉取)。
- **Storage:** `/midtier/minilab/scratch/chw4026/` = main (20T pool, **only 5.4T free / 74% used as of 2026-08-23** — was 9.8T in July; watch it). `/midtier/sablab/scratch/` = 780T free, holds shared `data/`. `$HOME` = config only. `~/.bashrc` already redirects TMPDIR/HF_HOME into scratch `.cache`.
- **Data:** ImageNet-1k `/midtier/sablab/scratch/data/ImageNet/ILSVRC2012/`; project copy `project3/data/imagenet_256/`; FID real-image sets `project3/data/reals_*_10k/`. COCO + many medical sets under `/midtier/sablab/scratch/data/`.

- **⭐ `sbatch` 整体故障时,`srun`/`salloc` 仍可用(2026-09-08 实测)。**
  报错 `Batch job submission failed: I/O error writing script/environment to file`
  是 slurmctld **保存 batch 脚本**那一步失败;**分配路径完好**。同一时刻实测:
  `srun --gres=gpu:a40:1 -p sablab-gpu` 与 `srun --gres=gpu:h200:1 -p minilab-gpu` 都成功,
  `sbatch --wrap="true"` 失败。已排除的两个错误假设:`--export=ALL` 环境过大、脚本过大
  (换成最小 export、换成空 `--wrap` 都照样失败)。
  ⇒ **绕行**:`salloc --no-shell` 拿分配 → 核对 `AllocTRES` → `srun --jobid=<J> bash <原 slurm 脚本>`。
  原脚本用 `torch.distributed.run --standalone` 启动,**没有嵌套 srun**,可直接这样跑;
  但脚本里的 `scontrol requeue` 在非 batch 作业上无效,必须自己在外面核对卡数。
  故障**间歇且窗口极短**:12 次密集重试(72 秒)一次没抓到,而两分钟前刚成功过
  ⇒ **"某次成功"不能用来判断已恢复**。
  另:`codex_research/watch_matched_ckpt.sh` 在 sbatch 失败时**只 echo 不重试**,
  停机期间每次触发都永久丢一个评测点 —— 停机时必须换成带重试的 srun 版 watcher。
- **`ai-gpu10` 的 `prj0129` 预留**已经**连续续期三次**(09-08 → 09-26 → **2026-10-09T23:00**,2026-09-28 复查)。
  ⇒ **默认当它不可用**,每次只用 `--test-only` 确认一次即可(实测 `allocation failure:
  Requested node configuration is not available`,即使把 `--cpus-per-task` 降到 8 也一样 ——
  预留吃掉全部 128 CPU)。`Users=` 名单里始终没有 chw4026。
  (此前记的 "EndTime=2026-09-08 之后可用" 已作废 —— 预留是会续的,**每次都要重查**。)
  gpu10 在 `sinfo` 里显示 `IDLE+RESERVED`、`AllocTRES` 里 gpu 占用为 0,
  ⇒ **任何按 `Gres − AllocTRES` 算空闲卡的脚本都会把它算成 4 张可用**。
  统计可派卡数时必须 `scontrol show res` 交叉核对用户名单。

Related: [[project3-research-hub]], [[phase0-scheduling-preferences]]

**2026-09-10 ★ 评测分区/QOS(此前一直用错)**:A40 同时在 `sablab-gpu`(gpu01–04)与 `preempt_gpu`(gpu01–05)。
我长期用 `preempt_gpu + qos=low`(**最低优先级**),导致评测排队 25 小时。
账户实际还有 **`sablab-high`/`sablab-low`** QOS 从未用过。
改成 `--partition=sablab-gpu --qos=high` 后:优先级 1281→2208,**StartTime 从 +25h 变成即刻**
(因为 sablab-gpu 可抢占 preempt_gpu 上别人的作业)。
**用法:判定点等关键评测用 sablab-gpu+high;日常补密评测仍走 preempt,避免频繁打断别人。**


**2026-09-28 ★★ 用户给的两条硬约束 + 提交前必做的四分区扫描**

1. **`sablab-gpu` 上我同时最多跑 6 张卡**(用户明确)。提交前先算已用:
   `squeue -u chw4026 -h -t R -o "%P|%b" | awk -F'|' '$1=="sablab-gpu"{n=$2;gsub(/.*:/,"",n);s+=n}END{print s+0}'`
2. **每次提交 GPU 作业都要扫四个分区**,挑空闲的或最快能排到的,别只盯 sablab:

   | 分区 | AllowQos | 节点 | 提交方式 |
   |---|---|---|---|
   | `sablab-gpu` | **仅 sablab-high** | gpu01-04, 06-09, 17-18 | `--account=sablab`(qos 留空) |
   | `minilab-gpu` | high, normal | gpu12-14, **17-18** | `--account=minilab --qos=high` |
   | `scu-gpu` | high, normal | **gpu10-11**, 19-20 | `--account=minilab --qos=high` |
   | `preempt_gpu` | **仅 low** | gpu01-14, 16-21 | `--qos=low`(可被抢占,CANCEL) |
   | `sablab-gpu-low` | sablab-low(prio 10) | gpu01-04,06-09,17-18 | 抢占不了 sablab-high |

   **H200(gpu17/18)同时在 sablab-gpu 与 minilab-gpu** ⇒ 两条路都要试。
   **H100 gpu14 只在 minilab-gpu**;**A100 gpu10-11 只在 scu-gpu**。

3. **用 `sbatch --test-only` 取各路径的最早起跑时间来选**,不要靠"哪个看起来空"。
   2026-09-28 实测示例(全部 ≥80GB 满载时):scu-gpu A100 = +27.5h、minilab-gpu H200 = +43h、
   sablab-gpu H200 = +44h ⇒ **只盯 sablab 会白等 16 小时**。
   注意该估计按前序作业 **TimeLimit** 算(偏悲观),前序一结束会立刻坍缩。
4. **长等待优先用"直接 sbatch 排队"而不是轮询 salloc**:排队能在别人早退时立刻接上,
   轮询只在看到空卡那一刻才抢、会输掉竞争。部分授予的风险由作业内 `EXPECT_NPROC` 守卫 + `--requeue` 兜住。

## 2026-09-30 补:`afterany` 依赖链的取消陷阱 + 部分授予的正确请求方式

**(1) `afterany:<job>` 在该作业被 `scancel` 时**也**算满足** ⇒ 取消链上任一作业,
它的后继会**立刻释放抢跑**。本轮因此瞬间破了 sablab 的 6 卡上限(3+2+2=7),
且连锁把第三个作业的依赖清成 `(null)`(9 张的隐患)。
**正确顺序:先 `scontrol update JobId=<后继> Dependency=...` 改挂或 hold,再 `scancel`。**
每次 `scancel` 之后**必须重新核一遍实得卡数**(`squeue -p sablab-gpu -u <me>` 逐条读 `AllocTRES`,
**不加项目过滤** —— 跨项目作业也占同一个配额)。

**(2) 训练臂一律 `--gres=<type>:4` + `EXPECT_NPROC="3 4"`。**
本集群会把请求降级:申请 4 得 3(`multik4_a4`)、申请 3 得 **2**(`mkF4_s2`)、
申请 h200:3 得 2(`qcoreQ7`)。而 **2 不在任何 baseline 的 world_size 里**,
守卫只能无限 requeue。申请 4 并接受 {3,4} 是唯一稳的组合
(np4 vs np3 的口径偏移实测仅 +0.17×MDE,见 EXPERIMENT_LOG §4.614)。
**为了"配对 np=3"而申请 3 是错的** —— 它换来的是 2,配不上任何东西。
