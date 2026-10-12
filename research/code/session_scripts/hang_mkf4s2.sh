#!/bin/bash
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
export EVAL_PART=preempt_gpu EVAL_QOS=low WAIT_MIN=5760
nohup "$O/codex_research/watch_matched_ckpt.sh" mkF4seed2 "$O/outputs/ab_open3_multikF4_s20260905_u100" 50 100   > "$O/outputs/mw_mkF4seed2.log" 2>&1 & echo "  挂 mkF4seed2 pid=$!"
