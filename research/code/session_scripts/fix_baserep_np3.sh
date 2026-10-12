#!/bin/bash
# baserep 实得 np4,但唯一对照 baseAA 是 np3 ⇒ 存在"全对"选择,按 §4.534 重起为纯 np3。
# 这与 §4.560 的 inner8 不同:那里两个对照 nproc 相反,无法全对。
set -u
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
S=$(dirname "$0"); SELF=$$
D="$B/outputs/ab_open3_baserep_s20260830_u100"
LOCK=/midtier/minilab/scratch/chw4026/tmp/ablock_ab_open3_baserep_s20260830_u100
cd "$B" || exit 1
echo "== 停 launcher =="
while read -r pid args; do
  [ "$pid" = "$SELF" ] || [ "$pid" = "$PPID" ] && continue
  case "$args" in *wait_and_launch_v4.sh\ baserep\ *) kill "$pid" 2>/dev/null && echo "  killed $pid";; esac
done < <(ps -u chw4026 -o pid=,args=)
echo "== scancel =="
scancel 1047850 2>&1 | sed 's/^/  /'
for i in $(seq 1 40); do squeue -j 1047850 -h 2>/dev/null | grep -q . || { echo "  已退出"; break; }; done
echo "== 清目录与锁(才跑 20 秒,无有效产物)=="
[ -d "$D" ] && { mv "$D" "${D}_np4_$(date +%H%M)"; echo "  目录已归档"; }
rm -f "$LOCK/jobid" 2>/dev/null; rmdir "$LOCK" 2>/dev/null && echo "  锁已清"
echo "== 以 REQ=3/ACCEPT=3 重起 =="
REQ=3 ACCEPT="3" SCRIPT=slurm/ab_open3_v5.slurm \
  nohup bash "$S/wait_and_launch_v4.sh" baserep 20260830 100 "$D" a100 3 \
        sablab-gpu sablab "ai-gpu06" > "$S/wl_baserep_np3.log" 2>&1 &
echo "  launcher pid=$!"
