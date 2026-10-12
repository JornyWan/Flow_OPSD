#!/bin/bash
# 写成脚本:避免命令行里出现臂名被后续 pgrep/pkill 模式误伤(已栽三次)
set -u
SD="$(dirname "$0")"; WL="$SD/wait_and_launch_v3.sh"
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$B" || exit 1
go(){ # arm seed upd outdir typ part acct node logname
  local a=$1 s=$2 u=$3 o=$4 t=$5 p=$6 c=$7 n=$8 lg=$9
  SCRIPT=slurm/ab_open3_v3.slurm REQ=4 ACCEPT="3 4" \
    nohup setsid bash "$WL" "$a" "$s" "$u" "$o" "$t" 4 "$p" "$c" "$n" \
    > "$SD/wl_${lg}.log" 2>&1 &
  echo "  派出 $lg -> $n ($t)  pid=$!"
}
go lr8x 20260830 200 "$B/outputs/ab_open3_lr8x_s20260830_u200"     h200 sablab-gpu  sablab  ai-gpu17 lr8x200
go lr6x 20260830 100 "$B/outputs/ab_open3_lr6x_s20260830_u100"     h200 sablab-gpu  sablab  ai-gpu18 lr6x
go lr8x 20260830 100 "$B/outputs/ab_open3_lr8xH100_s20260830_u100" h100 minilab-gpu minilab ai-gpu14 lr8xH100
