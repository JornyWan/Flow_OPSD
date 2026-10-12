#!/bin/bash
# 脚本形式:避免 pkill -f 匹配到调用者自己(已栽四次)
set -u
SELF=$$; SD="$(dirname "$0")"
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$B" || exit 1
k=0
while read -r pid args; do
  [ "$pid" = "$SELF" ] && continue
  case "$args" in *wait_and_launch_v3*) ;; *) continue ;; esac
  case "$args" in *lr8xH100*) kill "$pid" 2>/dev/null && k=$((k+1)) ;; esac
done < <(ps -u chw4026 -o pid=,args=)
echo "  停掉旧 launcher: $k 个"
SCRIPT=slurm/ab_open3_v3.slurm REQ=4 ACCEPT="3 4" \
  nohup setsid bash "$SD/wait_and_launch_v3.sh" inner2 20260830 200 \
  "$B/outputs/ab_open3_inner2H100_s20260830_u200" h100 4 minilab-gpu minilab ai-gpu14 \
  > "$SD/wl_inner2H100_u200.log" 2>&1 &
echo "  改派 inner2 u200 @ H100/ai-gpu14 (pid=$!)"
