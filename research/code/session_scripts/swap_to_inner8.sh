#!/bin/bash
# A100 槽从 lr10xeps 换给 inner8(§4.544:信息量更高;§4.543 已降低 lr10xeps 的先验)。
# lr10xeps 的登记(§4.541/§4.542)保留,watcher 保留,只降优先级。
set -u
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
S=$(dirname "$0"); SELF=$$
cd "$B" || exit 1
echo "== 停 lr10xeps launcher(watcher 保留) =="
k=0
while read -r pid args; do
  [ "$pid" = "$SELF" ] || [ "$pid" = "$PPID" ] && continue
  case "$args" in *wait_and_launch_v4.sh\ lr10xeps\ *) kill "$pid" 2>/dev/null && k=$((k+1));; esac
done < <(ps -u chw4026 -o pid=,args=)
echo "  已停 $k 个"

ARM=inner8; SEED=20260830; UP=100
OUT="$B/outputs/ab_open3_${ARM}_s${SEED}_u${UP}"
echo "== 起 inner8 launcher(A100,申请 4 接受 3|4) =="
REQ=4 ACCEPT="3 4" SCRIPT=slurm/ab_open3_v4.slurm \
  nohup bash "$S/wait_and_launch_v4.sh" "$ARM" "$SEED" "$UP" "$OUT" a100 3 \
        sablab-gpu sablab "ai-gpu09 ai-gpu07 ai-gpu08 ai-gpu06" \
  > "$S/wl_${ARM}.log" 2>&1 &
echo "  launcher pid=$!"
echo "== 挂评测 watcher(⚠️ CK_MULT=8 ⇒ ck = 8 × 外层 epoch) =="
for pair in "160:ep20" "400:ep50" "800:ep100"; do
  ck=${pair%%:*}; lbl=${pair##*:}
  nohup bash "$S/watch_srun_eval_long.sh" "$ARM" "$OUT" "$ck" > "$S/se_${ARM}_ck${ck}.log" 2>&1 &
  echo "  ck$ck ($lbl) pid=$!"
done
