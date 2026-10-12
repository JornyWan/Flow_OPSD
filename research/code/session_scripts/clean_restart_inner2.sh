#!/bin/bash
# np4 run 落了 checkpoint-10,np3 run 从它续跑 ⇒ 混合臂。清干净重起纯 np3。
# 目录用 mv 保留(不删),便于事后核查。
set -u
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
S=$(dirname "$0"); SELF=$$
D="$B/outputs/ab_open3_inner2H100_s20260830_u200"
LOCK=/midtier/minilab/scratch/chw4026/tmp/ablock_ab_open3_inner2H100_s20260830_u200
cd "$B" || exit 1

echo "== 1. 停 launcher =="
while read -r pid args; do
  [ "$pid" = "$SELF" ] || [ "$pid" = "$PPID" ] && continue
  case "$args" in *wait_and_launch_v4.sh\ inner2\ *) kill "$pid" 2>/dev/null && echo "  killed $pid";; esac
done < <(ps -u chw4026 -o pid=,args=)

echo "== 2. scancel =="
scancel 1047433 2>&1 | sed 's/^/  /'
for i in $(seq 1 40); do squeue -j 1047433 -h 2>/dev/null | grep -q . || { echo "  已退出"; break; }; done

echo "== 3. 归档被污染的目录(mv,不删) =="
if [ -d "$D" ]; then
  T="${D}_np4hybrid_$(date +%H%M)"
  mv "$D" "$T" && echo "  $D -> $(basename "$T")"
fi
rmdir "$LOCK" 2>/dev/null && echo "  锁已清" || { rm -f "$LOCK/jobid" && rmdir "$LOCK" && echo "  锁已清(先删 jobid)"; }

echo "== 4. 纯 np3 重起 =="
REQ=3 ACCEPT="3" SCRIPT=slurm/ab_open3_v4.slurm \
  nohup bash "$S/wait_and_launch_v4.sh" inner2 20260830 200 "$D" h100 3 minilab-gpu minilab "ai-gpu14" \
  > "$S/wl_inner2_clean.log" 2>&1 &
echo "  launcher pid=$!"
