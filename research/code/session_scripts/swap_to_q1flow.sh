#!/bin/bash
# 把 H200 槽从 qcorerho(先验已被 qopsd 的 held-out 失败削弱)换成 q1flow 探针。
# 写成脚本再执行:ps/grep 的模式会命中调用它的父 shell 命令行(已栽五次)。
set -u
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
S=$(dirname "$0"); SELF=$$
cd "$B" || exit 1

echo "== 停掉 qcorerho 的 launcher 与 watcher =="
k=0
while read -r pid args; do
  [ "$pid" = "$SELF" ] || [ "$pid" = "$PPID" ] && continue
  case "$args" in
    *wait_and_launch_v4.sh\ qcorerho\ *|*watch_ab_ckpt_hipri.sh\ qcorerho\ *)
      kill "$pid" 2>/dev/null && { echo "  killed $pid"; k=$((k+1)); } ;;
  esac
done < <(ps -u chw4026 -o pid=,args=)
echo "  共 $k 个"

echo "== 起 q1flow 探针(UPDATES=6 足够攒 ~2304 行) =="
ARM=q1flow; SEED=20260830; UP=6
OUT="$B/outputs/ab_open3_${ARM}_s${SEED}_u${UP}"
REQ=4 ACCEPT="3 4" SCRIPT=slurm/ab_open3_v4.slurm \
  nohup bash "$S/wait_and_launch_v4.sh" "$ARM" "$SEED" "$UP" "$OUT" h200 3 \
        sablab-gpu sablab "ai-gpu17 ai-gpu18" \
  > "$B/outputs/wl_${ARM}.log" 2>&1 &
echo "  launcher pid=$! -> outputs/wl_${ARM}.log"
echo "  (探针不需评测 watcher:产出是 $OUT/gateQ1_probe.jsonl,不是 checkpoint 评测)"
