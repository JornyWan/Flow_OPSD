#!/bin/bash
set -u
SP=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad
SELF=$$; k=0
while read -r pid a1 a2 rest; do
  [ "$pid" = "$SELF" ] && continue
  case "$a1" in */bash|bash) ;; *) continue;; esac
  [ "$a2" = "$SP/wait_and_launch_v4.sh" ] || continue
  case "$rest" in multik4*) kill "$pid" 2>/dev/null && k=$((k+1));; esac
done < <(ps -u chw4026 -o pid=,args=)
echo "  停掉 $k 个 multik4 轮询器"
