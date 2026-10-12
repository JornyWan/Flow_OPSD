#!/bin/bash
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
export EVAL_PART=preempt_gpu EVAL_QOS=low WAIT_MIN=5760
h(){ nohup "$O/codex_research/watch_matched_ckpt.sh" "$1" "$O/outputs/ab_open3_$1_s20260830_u100" 50 100 > "$O/outputs/mw_$1.log" 2>&1 & echo "  挂 $1 pid=$!"; }
h base4
h base6
