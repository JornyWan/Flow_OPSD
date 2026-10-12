---
name: no-temp-files-in-managed-repo
description: OPSD 仓库有 preflight fork-clean 检查;任何遗留的 .bak/.orig 会让训练作业 exit 42 并带死 launcher
metadata:
  type: feedback
---

2026-09-10:改 `train_opsd_ri_sd3.py` 前用 `cp` 在**仓库内**留了个 `train_opsd_ri_sd3.py.bak_0910_1624`。
两小时后 `inner2` launcher 成功拿到 3 张 H100,`ab_open3_*.slurm` 的
`preflight_check.py --expect-fork-clean` 判定"仓库有 1 处未预期改动"⇒ `exit 42`,
launcher 按设计 scancel 后**自身退出**,ai-gpu14 随后空转约 2 小时 —— 那是集群唯一一段有空卡的时间。

**Why:** preflight 只禁止**未预期的新文件**;对派生脚本本身的内容改动是允许的
(实测移走 .bak 后 preflight 退出码 0,而我的 flow-后缀改动仍在)。所以致命的不是"改了代码",是"多了个文件"。

**How to apply:**
1. 备份一律写到 scratchpad,**绝不放进 `CA-OPD/code/DiffusionOPSD-rpi/`**;
2. 改完仓库、起训练**之前**,本地跑一遍(不需要 GPU,几秒):
   `preflight_check.py --repo <repo> --config sd35_open3 --nproc 3 --seed <s> --expect-updates <u> --expect-fork-clean <--allow-env ...>`;
3. launcher 一旦因 step 失败退出就**不会自己回来**,派活后要复查它是否还活着。
参见 [[opsd-flag-chain-completeness]]、[[dont-drop-safety-checks-when-simplifying]]。

**2026-09-10 补(同一条臂的第三次起跑)**:因 nproc 不对而 scancel 重起时,
上一次留下的 `checkpoints/checkpoint-N` 会让新副本**自动 resume**,得到一条混合臂。
`run_attempts.jsonl` 的 `"resumed"` 字段是唯一能看出来的地方(0=全新,≥1=续跑)。
**判断一个 run 干净要看四样**:preflight 通过、`nproc` 与对照臂一致、**`resumed` 为 0**、
metrics 从 `_step=0` 起。前两样对而第三样错时,表面上完全看不出来。
重起前要把旧输出目录 `mv` 走(别删,便于核查),并清 `ablock_*` 锁(内含 `jobid` 文件,`rmdir` 会失败)。

**2026-09-10 补(watcher 会静默消失)**:`watch_srun_eval_preempt.sh` 的等待循环是
`for _ in $(seq 1 1440)` × `sleep 60` = **24 小时上限**,超时后打一行「ckN 超时未出现」就正常退出。
一条臂反复起跑失败拖过 24 小时后,watcher 全都没了,而 `ps` 里 launcher 还在、日志无 ERROR
⇒ 训练会跑完却**没有任何评测被提交**。
**How to apply:** 每个巡检周期都跑一次 `ps -u chw4026 -o args= | grep '[w]atch_'` 核对 watcher 数量;
训练**重起**后必须重挂(旧 watcher 的 24h 窗口从它自己启动时算,不随训练重置)。

**2026-09-11 补(watcher 的 NFS 刷新深度)**:`watch_srun_eval_*.sh` 里 `ls "$DIR"` 只刷一层,
而检查的是 `$DIR/checkpoints/checkpoint-N/lora/adapter_config.json`。
NFS 缓存了 `checkpoints/` 下该 ck 的**否定 dentry** ⇒ watcher 盲了 **12 分 50 秒**,
直到我的巡检 `ls -1v "$DIR/checkpoints"` 恰好刷开它,下一轮才触发。
**没有旁人巡检时,它可能一直不触发。**
**How to apply:** 轮询 NFS 上的新文件时,`ls` 必须覆盖**要检查的路径的每一层**
(`ls "$DIR" "$DIR/checkpoints" "$CK"`),不能只刷最外层。
这是本会话第三次 watcher 类静默失效(另两次:24h 超时、训练重起后未重挂)。

**2026-09-11 补(改了正在被执行的 slurm 脚本)**:job 1047436 从 19:45 跑到次日 10:34,
我在 21:xx 与 23:44 往 `ab_open3_v4.slurm` 里插了两条新臂(行 190、234,都在 epilogue 之前)。
**bash 按字节偏移逐段读脚本**,训练命令返回后用旧偏移继续读 ⇒ 读到移位后的内容,
epilogue 错乱、报 `_RC: unbound variable`、**退出码 1**(而训练产物完整)。
我在 §4.539 明明写过这条规矩,却只用在 watcher 上,没想到**正在被 SLURM 执行的 slurm 脚本是同一回事**,
而且它的执行窗口长达 15 小时。
**How to apply:** 任何可能正在被执行的脚本(watcher / slurm 作业 / launcher)一律**复制出新版本再改**;
加新臂写 `_v5.slurm`,不原地编辑。另:产物完整但 exit≠0 时,先查是不是 epilogue 被这样搞坏了。
