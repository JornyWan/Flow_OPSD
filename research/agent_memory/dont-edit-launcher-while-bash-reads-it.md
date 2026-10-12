---
name: dont-edit-launcher-while-bash-reads-it
description: "smoke 用 `bash ab_open3_v6.slurm` 直接调启动脚本时,脚本被逐段读取;运行中改它 ⇒ 读到错位行(unbound variable 等)、job FAILED、afterok 链断"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe
  modified: 2026-10-10T03:33:17.084Z
---

2026-10-09:gate smoke 训练已跑完,收尾时报 `line 483: OPA_RESUME_DIR: unbound variable` —— 因为我在它运行期间往
`slurm/ab_open3_v6.slurm` 加了两个新臂,bash 按旧字节偏移继续读,执行了错位的行。afterok 依赖它的两条正式臂随之"永不满足"。

**Why:** bash 不是一次读完脚本;sbatch 提交的作业读的是假脱机副本(安全),但 smoke 脚本里 `bash $OPD_ROOT/slurm/ab_open3_v6.slurm` 是直接读盘上文件。

**How to apply:**
- 改 `ab_open3_v6.slurm`(或任何被 `bash` 直接调用的脚本)前,先 `squeue` 确认没有 smk_* 作业在跑;有就等它结束,或把改动写进新文件。
- smoke 的 exit 码不可信时(守卫放弃走 exit 0、或收尾报错),以 metrics.jsonl 的读数人工判定,并手动处理依赖链。
- 同一机制的另一坑(2026-10-11):启动脚本里的守卫 `scontrol requeue $SLURM_JOB_ID` 在 smoke 里 requeue 的是 **smoke 作业本身**
  (smk_l40 因显存守卫被重排 40 次、空耗 5 h)。smoke 前先确认会不会撞守卫(显存 / nproc),必要时用 MIN_GPU_MIB / EXPECT_NPROC 放宽。
- 另一守卫(2026-10-11):启动脚本只收 nproc ∈ {3,4,6,8}(48×24 分组),单卡探针走启动脚本会 sleep 900 + requeue 循环 ⇒ 经启动脚本的探针至少要 3 卡。
- 相关:[[no-temp-files-in-managed-repo]]、[[self-kill-by-pattern-match]]
