#!/bin/bash
set -u
SP=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad
SELF=$$
echo "  真进程(命令行以 /bin/bash <脚本绝对路径> 开头):"
while read -r pid a1 a2 rest; do
  [ "$pid" = "$SELF" ] && continue
  case "$a1" in */bash|bash) ;; *) continue;; esac
  case "$a2" in
    "$SP/launch_qcoreQ7.sh")        echo "    ✅ launcher pid=$pid" ;;
    "$SP/wait_and_launch_v4.sh")    echo "    ✅ wait_and_launch pid=$pid  args=$rest" ;;
    "$SP/watch_srun_eval_long2.sh") echo "    ✅ watcher pid=$pid  args=$rest" ;;
  esac
done < <(ps -u chw4026 -o pid=,args=)
