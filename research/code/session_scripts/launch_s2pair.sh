#!/bin/bash
# §4.841 seed-2 复现。用法: launch_s2pair.sh canon|msp
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SEED=20260905
case "$1" in canon) ARM=baseline; TAG=canon300s2;; msp) ARM=multikF4; TAG=msp300s2;; *) echo "canon|msp"; exit 1;; esac
OUT="$O/outputs/ab_open3_${ARM}_s${SEED}_u300"
J=$(env EXPECT_NPROC=4 ARM="$ARM" SEED="$SEED" UPDATES=300 \
      AB_CONFIG=config/public.py:sd35_open3 OUTPUT_DIR="$OUT" \
    sbatch --parsable --job-name="$TAG" \
      --account=minilab --partition=minilab-gpu --qos=high \
      --exclusive --gres=gpu:h200:4 "$O/slurm/ab_open3_v6.slurm")
nohup bash "$O/codex_research/watch_matched_ckpt.sh" "$TAG" "$OUT" 100 150 200 250 300 \
  > "$O/outputs/watch_${TAG}.log" 2>&1 &
echo "  $TAG job=$J watcher=$!  out=$(basename $OUT)"
