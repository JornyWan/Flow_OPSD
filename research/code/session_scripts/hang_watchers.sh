#!/bin/bash
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
W=$O/codex_research/watch_matched_ckpt.sh
L=$O/outputs
export EVAL_PART=preempt_gpu EVAL_QOS=low     # A40 评测不吃 sablab 的 6 卡预算
hang() {  # TAG DIR steps...
  local tag=$1 dir=$2; shift 2
  nohup "$W" "$tag" "$L/$dir" "$@" > "$L/mw_${tag}.log" 2>&1 &
  echo "  挂 $tag ($dir) ck=$* pid=$!"
}
hang multik4    ab_open3_multik4_s20260830_u100   100
hang multikF4   ab_open3_multikF4_s20260830_u100  50 100
hang hybrid     ab_open3_hybrid_s20260830_u100    50 100
hang hybridw    ab_open3_hybridw_s20260830_u100   50 100
hang hybridtr   ab_open3_hybridtr_s20260830_u100  50 100
hang multikALL  ab_open3_multikALL_s20260830_u100 50 100
hang mk4seed2   ab_open3_multik4_s20260905_u100   50 100
hang baseA4     ab_open3_baseA4_s20260830_u100    100
