#!/bin/bash
# 把 inner2 u200 从 np4 重起为 np3,与对照臂 baseH100(world_size=3)及项目其它臂一致。
set -u
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
S=$(dirname "$0"); SELF=$$
LOCK=/midtier/minilab/scratch/chw4026/tmp/ablock_ab_open3_inner2H100_s20260830_u200
cd "$B" || exit 1

echo "== 1. 停掉存活的 inner2 launcher(避免它抢在前面又要 4 张) =="
while read -r pid args; do
  [ "$pid" = "$SELF" ] || [ "$pid" = "$PPID" ] && continue
  case "$args" in *wait_and_launch_v4.sh\ inner2\ *) kill "$pid" 2>/dev/null && echo "  killed launcher $pid";; esac
done < <(ps -u chw4026 -o pid=,args=)

echo "== 2. scancel np4 作业 =="
scancel 1047400 2>&1 | sed 's/^/  /'
for i in $(seq 1 20); do
  squeue -j 1047400 -h 2>/dev/null | grep -q . || { echo "  作业已退出(第 ${i} 次检查)"; break; }
done

echo "== 3. 清锁 =="
rmdir "$LOCK" 2>/dev/null && echo "  已删 $LOCK" || { [ -d "$LOCK" ] && echo "  ⚠️ 锁仍在(非空?)" || echo "  锁已不存在"; }

echo "== 4. 以 REQ=3 / ACCEPT=3 重起 =="
REQ=3 ACCEPT="3" SCRIPT=slurm/ab_open3_v4.slurm \
  nohup bash "$S/wait_and_launch_v4.sh" inner2 20260830 200 \
        "$B/outputs/ab_open3_inner2H100_s20260830_u200" h100 3 minilab-gpu minilab "ai-gpu14" \
  > "$S/wl_inner2_np3.log" 2>&1 &
echo "  launcher pid=$!"
