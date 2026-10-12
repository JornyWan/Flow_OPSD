#!/bin/bash
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
W=$O/codex_research/watch_matched_ckpt.sh; L=$O/outputs
export EVAL_PART=preempt_gpu EVAL_QOS=low WAIT_MIN=5760
h(){ local t=$1 d=$2; shift 2; nohup "$W" "$t" "$L/$d" "$@" > "$L/mw_${t}.log" 2>&1 & echo "  挂 $t pid=$!"; }
h multikHI ab_open3_multikHI_s20260830_u100 50 100
h multik4H ab_open3_multik4H_s20260830_u100 50 100
