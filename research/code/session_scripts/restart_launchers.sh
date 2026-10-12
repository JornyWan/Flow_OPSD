#!/bin/bash
# .bak 已移出仓库、preflight 已本地验证通过 ⇒ 重启被 exit 42 带死的 launcher。
set -u
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
S=$(dirname "$0"); SELF=$$
cd "$B" || exit 1
alive(){ # tag -> 0 若已有活着的 launcher
  while read -r pid args; do
    [ "$pid" = "$SELF" ] || [ "$pid" = "$PPID" ] && continue
    case "$args" in *wait_and_launch_v4.sh\ $1\ *) return 0;; esac
  done < <(ps -u chw4026 -o pid=,args=); return 1; }

start(){ # arm seed updates outdir typ ngpu part acct nodes
  local arm=$1; shift
  if alive "$arm"; then echo "  $arm: launcher 仍存活,跳过"; return; fi
  REQ=4 ACCEPT="3 4" SCRIPT=slurm/ab_open3_v4.slurm \
    nohup bash "$S/wait_and_launch_v4.sh" "$arm" "$@" > "$S/wl_${arm}_restart.log" 2>&1 &
  echo "  $arm: 重启 pid=$! -> scratchpad/wl_${arm}_restart.log"
}
echo "== 重启 =="
start inner2 20260830 200 "$B/outputs/ab_open3_inner2H100_s20260830_u200" h100 3 minilab-gpu minilab "ai-gpu14"
start q1flow 20260830 6   "$B/outputs/ab_open3_q1flow_s20260830_u6"        h200 3 sablab-gpu  sablab  "ai-gpu17 ai-gpu18"
start lr6x   20260830 100 "$B/outputs/ab_open3_lr6x_s20260830_u100"        a100 3 sablab-gpu  sablab  "ai-gpu09 ai-gpu07 ai-gpu08 ai-gpu06"
