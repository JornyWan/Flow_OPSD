#!/bin/bash
# §4.844 第二阶段:qlook1,H100 np3,100 updates;对照 baseH100/baserepH100(同硬件同 np 同 seed)。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
OUT="$O/outputs/ab_open3_qlook1fz_s20260830_u100"
J=$(env EXPECT_NPROC=3 ARM=qlook1fz SEED=20260830 UPDATES=100 \
      AB_CONFIG=config/public.py:sd35_open3 OUTPUT_DIR="$OUT" \
    sbatch --parsable --job-name=qlook1fz \
      --account=minilab --partition=minilab-gpu --qos=high \
      --gres=gpu:h100:3 --nodelist=ai-gpu14 --cpus-per-task=24 --mem=300G \
      "$O/slurm/ab_open3_v6.slurm")
nohup bash "$O/codex_research/watch_matched_ckpt.sh" qlook1fz "$OUT" 50 100 > "$O/outputs/watch_qlook1fz.log" 2>&1 &
echo "  qlook1fz job=$J watcher=$!"
