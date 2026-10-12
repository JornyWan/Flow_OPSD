#!/bin/bash
set -u
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
S=$(dirname "$0"); SELF=$$
D="$B/outputs/ab_open3_inner2H100_s20260830_u200"
k=0
while read -r pid args; do
  [ "$pid" = "$SELF" ] || [ "$pid" = "$PPID" ] && continue
  case "$args" in *watch_srun_eval_preempt.sh\ inner2H100\ *) kill "$pid" 2>/dev/null && k=$((k+1));; esac
done < <(ps -u chw4026 -o pid=,args=)
echo "  停旧版 $k 个"
for ck in 200 400; do
  nohup bash "$S/watch_srun_eval_long.sh" inner2H100 "$D" "$ck" > "$S/se_inner2H100_ck${ck}_long.log" 2>&1 &
  echo "  ck$ck 已挂 pid=$!"
done
