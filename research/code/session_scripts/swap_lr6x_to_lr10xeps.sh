#!/bin/bash
# A100 槽从 lr6x 换给 lr10xeps(§4.541:信息量更高)。lr6x 的登记(§4.451 臂2)保留,只是降优先级。
set -u
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
S=$(dirname "$0"); SELF=$$
cd "$B" || exit 1
echo "== 停 lr6x launcher(watcher 保留,120h 版,无害) =="
k=0
while read -r pid args; do
  [ "$pid" = "$SELF" ] || [ "$pid" = "$PPID" ] && continue
  case "$args" in *wait_and_launch_v4.sh\ lr6x\ *) kill "$pid" 2>/dev/null && k=$((k+1));; esac
done < <(ps -u chw4026 -o pid=,args=)
echo "  已停 $k 个"

ARM=lr10xeps; SEED=20260830; UP=100
OUT="$B/outputs/ab_open3_${ARM}_s${SEED}_u${UP}"
echo "== 起 lr10xeps launcher(A100,申请 4 接受 3|4) =="
REQ=4 ACCEPT="3 4" SCRIPT=slurm/ab_open3_v4.slurm \
  nohup bash "$S/wait_and_launch_v4.sh" "$ARM" "$SEED" "$UP" "$OUT" a100 3 \
        sablab-gpu sablab "ai-gpu09 ai-gpu07 ai-gpu08 ai-gpu06" \
  > "$S/wl_${ARM}.log" 2>&1 &
echo "  launcher pid=$!"
echo "== 挂评测 watcher(CK_MULT=1 ⇒ ck = 外层 epoch) =="
for ck in 20 50 100; do
  nohup bash "$S/watch_srun_eval_long.sh" "$ARM" "$OUT" "$ck" \
    > "$S/se_${ARM}_ck${ck}.log" 2>&1 &
  echo "  ck$ck pid=$!"
done
