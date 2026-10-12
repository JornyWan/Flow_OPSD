#!/bin/bash
set -u
SD="$(dirname "$0")"; W="$SD/watch_srun_eval_preempt.sh"
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$B" || exit 1
arm(){ nohup setsid bash "$W" "$1" "$2" "$3" > "$SD/se_${1}_ck${3}.log" 2>&1 & echo "  派出 $1 ck$3 (pid=$!)"; }
arm base200  "$B/outputs/ab_open3_baseline_np3_s20260830_u200" 100
arm lr2x200  "$B/outputs/ab_open3_lr2x_s20260830_u200"         100
arm inner2b  "$B/outputs/ab_open3_inner2_s20260830_u200"       200
