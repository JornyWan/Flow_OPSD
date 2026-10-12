#!/bin/bash
# §4.561 的第二次重复,放 H100(gpu14 4 张全空、IDLE、无其它作业 ⇒ 不抢占)。
# 对照 baseH100(H100/np3/seed-1/updates=100,已有 A40 评测)⇒ 与 A100 侧那次完全独立。
set -u
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
S=$(dirname "$0")
cd "$B" || exit 1
ARM=baserep; SEED=20260830; UP=100
OUT="$B/outputs/ab_open3_baserepH100_s${SEED}_u${UP}"
REQ=3 ACCEPT="3" SCRIPT=slurm/ab_open3_v5.slurm \
  nohup bash "$S/wait_and_launch_v4.sh" "$ARM" "$SEED" "$UP" "$OUT" h100 3 \
        minilab-gpu minilab "ai-gpu14" > "$S/wl_baserepH100.log" 2>&1 &
echo "  launcher pid=$!  -> $OUT"
for ck in 20 50 100; do
  nohup bash "$S/watch_srun_eval_long2.sh" baserepH100 "$OUT" "$ck" \
    > "$S/se2_baserepH100_ck${ck}.log" 2>&1 &
  echo "  watcher ck$ck pid=$!  (CK_MULT=1 ⇒ ck = 外层 epoch)"
done
