#!/bin/bash
# §4.773 两条退火臂的 watcher。默认四格(drawbench/unseen1k × flow-40/flow-4)。
# ck250 也挂:纯 MSP 在 200→250 之间越过零点,250 能看出恢复是立刻发生还是滞后。
# WAIT_MIN 调大 —— 两臂挂在 smoke 的 afterok 依赖后面,起跑可能要几小时。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SP=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad
for a in anneal200 resumectl200; do
  D=$O/outputs/ab_open3_${a}_s20260830_u300
  nohup env WAIT_MIN=2880 EVAL_PART=preempt_gpu EVAL_QOS=low \
    bash "$O/codex_research/watch_matched_ckpt.sh" "$a" "$D" 250 300 \
    > "$SP/mw_${a}.log" 2>&1 &
  echo "  watcher $a pid=$! log=$SP/mw_${a}.log"
done
