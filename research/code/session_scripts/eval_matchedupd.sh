#!/bin/bash
set -u
SD="$(dirname "$0")"; W="$SD/watch_srun_eval_preempt.sh"
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$B" || exit 1
D="$B/outputs/ab_open3_baseline_np3_s20260830_u100"
for ck in 40 60; do
  [ -d "$D/checkpoints/checkpoint-$ck/lora" ] || { echo "  ck$ck 缺 checkpoint,跳过"; continue; }
  [ -f "$B/outputs/matched_pi_basenp3_${ck}_default_drawbench_flow40.json" ] && { echo "  basenp3_$ck 已有,跳过"; continue; }
  nohup setsid bash "$W" basenp3 "$D" "$ck" > "$SD/se_basenp3_ck${ck}.log" 2>&1 &
  echo "  挂上 basenp3 ck$ck (pid=$!)"
done
