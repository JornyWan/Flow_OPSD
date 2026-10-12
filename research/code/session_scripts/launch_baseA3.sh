#!/bin/bash
# A100/np3 组缺的 canonical 基线(§4.828 §四)。
# 显式 --gres=gpu:a100:3 + EXPECT_NPROC=3:确定性地拿到 np3,与该组三条臂匹配;
# 若只拿到 2 张,守卫会 requeue(§4.591)。不加 --exclusive —— 加了会变 np4,就不匹配了。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SEED=20260830
J=$(env EXPECT_NPROC=3 ARM=baseA4 SEED="$SEED" UPDATES=100 \
      AB_CONFIG=config/public.py:sd35_open3 \
      OUTPUT_DIR="$O/outputs/ab_open3_baseA3_s${SEED}_u100" \
    sbatch --parsable --job-name=baseA3 \
      --account=sablab --partition=scu-gpu \
      --gres=gpu:a100:3 --cpus-per-task=24 --mem=300G --time=24:00:00 \
      "$O/slurm/ab_open3_v6.slurm")
echo "  baseA3 job=$J"
nohup bash "$O/codex_research/watch_matched_ckpt.sh" baseA3 \
  "$O/outputs/ab_open3_baseA3_s${SEED}_u100" 50 100 \
  > "$O/outputs/watch_baseA3.log" 2>&1 &
echo "  watcher pid=$!"
