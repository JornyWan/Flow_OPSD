#!/bin/bash
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
export EVAL_PART=preempt_gpu EVAL_QOS=low WAIT_MIN=5760
nohup "$O/codex_research/watch_matched_ckpt.sh" hybrid "$O/outputs/ab_open3_hybrid_s20260830_u100" 50 100 \
  > "$O/outputs/mw_hybrid.log" 2>&1 & echo "  重挂 hybrid pid=$!"
