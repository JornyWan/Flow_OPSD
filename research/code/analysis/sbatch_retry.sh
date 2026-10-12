#!/bin/bash
# sbatch 的重试封装。
# 2026-08-25 遇到过 slurmctld 的 StateSaveLocation 写不进去(headnode 上 /var/spool/slurm/slurmctld),
# 表现为**任何**提交都报 "I/O error writing script/environment to file" —— 连两行的空脚本也一样。
# 这是控制器侧的故障,与作业内容无关,通常几分钟到几十分钟自愈。
# 提交失败若不重试,自动化链路(比如 arm 跑完自动排评测)会静默断掉,所以这里统一兜底。
# 用法: sbatch_retry.sh [--tries N] [--sleep S] <原样传给 sbatch 的参数...>
TRIES=60; SLEEP=60
while [ $# -gt 0 ]; do
  case "$1" in
    --tries) TRIES=$2; shift 2;;
    --sleep) SLEEP=$2; shift 2;;
    *) break;;
  esac
done
for t in $(seq 1 "$TRIES"); do
  OUT=$(sbatch --parsable "$@" 2>&1)
  if [ $? -eq 0 ] && [ -n "$OUT" ] && [ -z "${OUT//[0-9_]/}" ]; then
    [ "$t" -gt 1 ] && echo "[sbatch_retry] 第 $t 次尝试成功" >&2
    echo "$OUT"; exit 0
  fi
  echo "[sbatch_retry] 第 $t/$TRIES 次失败: $OUT" >&2
  sleep "$SLEEP"
done
echo "[sbatch_retry] $TRIES 次全部失败,放弃" >&2
exit 1
