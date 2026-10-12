#!/bin/bash
set -u
SD="$(dirname "$0")"; W="$SD/watch_srun_eval_preempt.sh"
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$B" || exit 1
arm(){ local tag=$1 dir=$2 ck=$3
  if ps -u chw4026 -o args= | grep -F watch_srun_eval | grep -v grep | grep -qE " $tag .* $ck\$"; then
    echo "  已挂 $tag ck$ck,跳过"; return; fi
  nohup setsid bash "$W" "$tag" "$dir" "$ck" > "$SD/se_${tag}_ck${ck}.log" 2>&1 &
  echo "  派出 $tag ck$ck (pid=$!)"; }
arm inner2gacc2s2 "$B/outputs/ab_open3_inner2gacc2_s20260905_u100" 80
arm lr4x          "$B/outputs/ab_open3_lr4x_s20260830_u100"        20
arm gacc4         "$B/outputs/ab_open3_gacc4_s20260830_u100"       80
