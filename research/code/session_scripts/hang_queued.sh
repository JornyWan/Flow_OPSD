#!/bin/bash
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
W=$O/codex_research/watch_matched_ckpt.sh; L=$O/outputs
export EVAL_PART=preempt_gpu EVAL_QOS=low WAIT_MIN=5760   # 96h:覆盖排队等待
h(){ local t=$1 d=$2; shift 2; nohup "$W" "$t" "$L/$d" "$@" > "$L/mw_${t}.log" 2>&1 & echo "  重挂 $t pid=$!"; }
h hybridw   ab_open3_hybridw_s20260830_u100   50 100
h hybridtr  ab_open3_hybridtr_s20260830_u100  50 100
h multikALL ab_open3_multikALL_s20260830_u100 50 100
h mk4seed2  ab_open3_multik4_s20260905_u100   50 100
h baseA4    ab_open3_baseA4_s20260830_u100    100
