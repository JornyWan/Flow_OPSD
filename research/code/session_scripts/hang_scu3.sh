#!/bin/bash
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
export EVAL_PART=preempt_gpu EVAL_QOS=low WAIT_MIN=5760
h(){ nohup "$O/codex_research/watch_matched_ckpt.sh" "$1" "$O/outputs/$2" 50 100 > "$O/outputs/mw_$1.log" 2>&1 & echo "  挂 $1 pid=$!"; }
h multik2  ab_open3_multik2_s20260830_u100
h multik4s ab_open3_multik4s_s20260830_u100
h hybgrad  ab_open3_hybgrad_s20260830_u100
