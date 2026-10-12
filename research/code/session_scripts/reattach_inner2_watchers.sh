#!/bin/bash
# inner2H100 的两个评测 watcher 因 24h 超时自然退出(日志:「ck200 超时未出现」)。
# 训练已重起并在 ep16,ck200≈6.3h、ck400≈13.8h 后出现,新的 24h 窗口能覆盖。
set -u
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
S=$(dirname "$0"); SELF=$$
D="$B/outputs/ab_open3_inner2H100_s20260830_u200"
for ck in 200 400; do
  # 已存活就不重复挂
  dup=0
  while read -r pid args; do
    [ "$pid" = "$SELF" ] || [ "$pid" = "$PPID" ] && continue
    case "$args" in *watch_srun_eval_preempt.sh\ inner2H100\ *\ $ck) dup=1;; esac
  done < <(ps -u chw4026 -o pid=,args=)
  [ "$dup" = 1 ] && { echo "  ck$ck: 已有 watcher,跳过"; continue; }
  nohup bash "$S/watch_srun_eval_preempt.sh" inner2H100 "$D" "$ck" \
    > "$S/se_inner2H100_ck${ck}_r2.log" 2>&1 &
  echo "  ck$ck: 已挂 pid=$!  -> se_inner2H100_ck${ck}_r2.log"
done
