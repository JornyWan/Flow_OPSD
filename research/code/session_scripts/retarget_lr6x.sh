#!/bin/bash
set -u
SELF=$$; SD="$(dirname "$0")"
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$B" || exit 1
K=0
while read -r pid args; do
  [ "$pid" = "$SELF" ] || [ "$pid" = "$PPID" ] && continue
  case "$args" in *wait_and_launch_v3*) ;; *) continue ;; esac
  case "$args" in *ab_open3_lr6x*) kill "$pid" 2>/dev/null && K=$((K+1)) ;; esac
done < <(ps -u chw4026 -o pid=,args=)
echo "  停掉旧 lr6x launcher: $K 个"
SCRIPT=slurm/ab_open3_v3.slurm REQ=4 ACCEPT="3 4" \
  nohup setsid bash "$SD/wait_and_launch_v3.sh" lr6x 20260830 100 \
  "$B/outputs/ab_open3_lr6x_s20260830_u100" a100 3 sablab-gpu sablab "ai-gpu09 ai-gpu07 ai-gpu08 ai-gpu10 ai-gpu11" \
  > "$SD/wl_lr6x_a100.log" 2>&1 &
echo "  重派 lr6x -> A100 池(gpu09/07/08/10/11),阈值 3 张  pid=$!"
