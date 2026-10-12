#!/bin/bash
# RB-LT 三臂的 watcher。四格(drawbench/unseen1k × flow-40/flow-4),ck50 + ck100。
# 判定点是 ck100(§4.780 §八);ck50 只看轨迹,不判(§4.743:ck50 会误判 ck100)。
# WAIT_MIN 调大:三臂在等 H200 整节点,起跑可能要几小时。
# EVAL_PART=preempt_gpu 以免吃掉 sablab 的"同时 6 卡"预算。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SP=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad
for a in poseuler rblt rbltd0; do
  D=$O/outputs/ab_pickscore_${a}_s20260830_u100
  T=$([ "$a" = poseuler ] && echo rblposeuler || echo $a)
  nohup env WAIT_MIN=2880 EVAL_PART=preempt_gpu EVAL_QOS=low \
    bash "$O/codex_research/watch_matched_ckpt.sh" "$T" "$D" 50 100 \
    > "$SP/mw_${T}.log" 2>&1 &
  echo "  watcher $(printf '%-12s' $T) pid=$! dir=$(basename $D)"
done
