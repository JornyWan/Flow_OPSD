#!/bin/bash
# LT-OPSD 三条臂(§4.830)。smoke 1067882/1067883 五条判据全过。
# 硬件:minilab-gpu h200 整节点 4 张 --exclusive(与 poseuler/rbltg 同轨道)。CK_MULT=1。
# 提交顺序 = 优先级:主臂 → 匹配 canonical 对照(没有它主臂判不了)→ 插值零控制。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SEED=20260830
sub(){ local jn=$1 arm=$2
  local out="ab_pickscore_${arm}_s${SEED}_u100"
  local J=$(env EXPECT_NPROC=4 ARM="$arm" SEED="$SEED" UPDATES=100 \
      AB_CONFIG=config/public.py:sd35_pickscore \
      OUTPUT_DIR="$O/outputs/$out" \
    sbatch --parsable --job-name="$jn" \
      --account=minilab --partition=minilab-gpu --qos=high \
      --exclusive --gres=gpu:h200:4 \
      "$O/slurm/ab_open3_v6.slurm")
  nohup bash "$O/codex_research/watch_matched_ckpt.sh" "$arm" "$O/outputs/$out" 50 100 \
    > "$O/outputs/watch_${arm}.log" 2>&1 &
  echo "  $arm job=$J watcher=$!"; }
sub lt_main ltopsd
sub lt_qse  qseuler
sub lt_zero ltopsd0
