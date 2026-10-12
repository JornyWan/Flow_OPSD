#!/bin/bash
# §4.521 qcorerho:Q-core + opa_rho=0.05。写成脚本再执行,避免 ps/grep 命中自身命令行。
set -u
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
S=$(dirname "$0")
cd "$B" || exit 1
ARM=qcorerho; SEED=20260830; UP=100
OUT="$B/outputs/ab_open3_${ARM}_s${SEED}_u${UP}"
# 训练:H200(与 qcore/basenp3 同硬件,配对三轴之一);申请 4 张、接受 3 或 4
REQ=4 ACCEPT="3 4" SCRIPT=slurm/ab_open3_v4.slurm \
  nohup bash "$S/wait_and_launch_v4.sh" "$ARM" "$SEED" "$UP" "$OUT" h200 3 \
        sablab-gpu sablab "ai-gpu17 ai-gpu18" \
  > "$B/outputs/wl_${ARM}.log" 2>&1 &
echo "launcher pid=$! -> outputs/wl_${ARM}.log"
# 评测 watcher:CK_MULT=1(未改 inner/gacc)⇒ ck 编号 = 外层 epoch。A40 固定。
nohup bash "$B/codex_research/watch_ab_ckpt_hipri.sh" "$ARM" "$OUT" 100 20 50 \
  > "$B/outputs/watch_${ARM}.log" 2>&1 &
echo "watcher  pid=$! -> outputs/watch_${ARM}.log"
