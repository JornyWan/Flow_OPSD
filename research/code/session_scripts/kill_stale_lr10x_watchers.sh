#!/bin/bash
# 杀 lr10x 的残留 watcher(该臂 §4.346 因发散停止,ck100 永不出现)。
# 用 ps 快照 + **整条 args 精确匹配**(§4.319:子串匹配会撞上父 shell 的 args)。
set -u
S=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
D="$B/outputs/ab_open3_lr10x_s20260830_u100"
T1="bash $S/watch_a100.sh lr10x $D 100"
T2="bash $B/codex_research/watch_matched_ckpt.sh lr10x $D 100"
k=0
while read -r pid args; do
  if [ "$args" = "$T1" ] || [ "$args" = "$T2" ]; then
    kill "$pid" 2>/dev/null && { echo "  killed pid=$pid"; k=$((k+1)); }
  fi
done < <(ps -u chw4026 -o pid=,args=)
echo "  共杀 $k 个"
