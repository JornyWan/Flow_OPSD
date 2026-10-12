#!/bin/bash
# 分两步:本脚本由**另一条命令**执行,所以父 shell 的命令行里不含待匹配模式
set -u
SELF=$$; SD="$(dirname "$0")"
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$B" || exit 1
K=0
while read -r pid args; do
  [ "$pid" = "$SELF" ] || [ "$pid" = "$PPID" ] && continue
  case "$args" in *wait_and_launch_v3*) ;; *) continue ;; esac
  kill "$pid" 2>/dev/null && K=$((K+1))
done < <(ps -u chw4026 -o pid=,args=)
echo "  停掉旧 launcher: $K 个(阈值 4,拿不到 3 张的节点)"
go(){ local a=$1 s=$2 u=$3 o=$4 t=$5 p=$6 c=$7 n=$8 lg=$9
  SCRIPT=slurm/ab_open3_v3.slurm REQ=4 ACCEPT="3 4" \
    nohup setsid bash "$SD/wait_and_launch_v3.sh" "$a" "$s" "$u" "$o" "$t" 3 "$p" "$c" "$n" \
    > "$SD/wl_${lg}.log" 2>&1 &
  echo "  重派 $lg -> $n (阈值改为 3 张) pid=$!"; }
go lr8x   20260830 200 "$B/outputs/ab_open3_lr8x_s20260830_u200"     h200 sablab-gpu  sablab  ai-gpu17 lr8x200
go lr6x   20260830 100 "$B/outputs/ab_open3_lr6x_s20260830_u100"     h200 sablab-gpu  sablab  ai-gpu18 lr6x
go inner2 20260830 200 "$B/outputs/ab_open3_inner2H100_s20260830_u200" h100 minilab-gpu minilab ai-gpu14 inner2H100_u200
