#!/bin/bash
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
W=$O/codex_research/watch_matched_ckpt.sh; L=$O/outputs
for pid in 748881 748882 752728 752729 752730 752731 752732 1302121 1302122 1401781; do
  kill $pid 2>/dev/null && echo "  杀 $pid"
done
sleep 1
export EVAL_PART=preempt_gpu EVAL_QOS=low WAIT_MIN=5760 \
       EVAL_CELLS="drawbench:flow:40 unseen1k:flow:40"
h(){ local t=$1 d=$2; shift 2; nohup "$W" "$t" "$L/$d" "$@" > "$L/mw_${t}.log" 2>&1 & echo "  挂 $t pid=$!"; }
h multik4   ab_open3_multik4_s20260830_u100   100
h multikF4  ab_open3_multikF4_s20260830_u100  50 100
h multikALL ab_open3_multikALL_s20260830_u100 50 100
h multikHI  ab_open3_multikHI_s20260830_u100  50 100
h multik4H  ab_open3_multik4H_s20260830_u100  50 100
h mk4seed2  ab_open3_multik4_s20260905_u100   50 100
h baseA4    ab_open3_baseA4_s20260830_u100    100
h hybrid    ab_open3_hybrid_s20260830_u100    50 100
h hybridw   ab_open3_hybridw_s20260830_u100   50 100
h hybridtr  ab_open3_hybridtr_s20260830_u100  50 100
