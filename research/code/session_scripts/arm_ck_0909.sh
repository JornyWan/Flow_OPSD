#!/bin/bash
set -u
SD="$(dirname "$0")"; W="$SD/watch_srun_eval_preempt.sh"
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$B" || exit 1
have(){ ps -u chw4026 -o args= | grep -F "watch_srun_eval" | grep -v grep | grep -qE " $1 .* $2\$"; }
arm(){ local tag=$1 dir=$2 ck=$3
  if have "$tag" "$ck"; then echo "  已挂 $tag ck$ck,跳过"; return; fi
  nohup setsid bash "$W" "$tag" "$dir" "$ck" > "$SD/se_${tag}_ck${ck}.log" 2>&1 &
  echo "  挂上 $tag ck$ck (pid=$!)"; }
arm inner2lrhalf   "$B/outputs/ab_open3_inner2lrhalf_s20260830_u100" 100
arm inner2gacc2s2  "$B/outputs/ab_open3_inner2gacc2_s20260905_u100"  200
