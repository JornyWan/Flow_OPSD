#!/bin/bash
set -u
SELF=$$; SD="$(dirname "$0")"
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$B" || exit 1
K=0
while read -r pid args; do
  [ "$pid" = "$SELF" ] || [ "$pid" = "$PPID" ] && continue
  case "$args" in *wait_and_launch_v3*) ;; *) continue ;; esac
  case "$args" in *ab_open3_lr6x*|*inner2H100*) kill "$pid" 2>/dev/null && K=$((K+1)) ;; esac
done < <(ps -u chw4026 -o pid=,args=)
echo "  停掉旧 launcher: $K 个"
go(){ local a=$1 s=$2 u=$3 o=$4 t=$5 p=$6 c=$7 n=$8 lg=$9
  SCRIPT=slurm/ab_open3_v3.slurm REQ=4 ACCEPT="3 4" \
    nohup setsid bash "$SD/wait_and_launch_v4.sh" "$a" "$s" "$u" "$o" "$t" 3 "$p" "$c" "$n" \
    > "$SD/wl_${lg}_v4.log" 2>&1 &
  echo "  v4 重派 $lg -> $n  pid=$!"; }
go lr6x   20260830 100 "$B/outputs/ab_open3_lr6x_s20260830_u100"       a100 sablab-gpu  sablab  "ai-gpu09 ai-gpu07 ai-gpu08 ai-gpu10 ai-gpu11" lr6x
go inner2 20260830 200 "$B/outputs/ab_open3_inner2H100_s20260830_u200" h100 minilab-gpu minilab "ai-gpu14" inner2H100_u200
