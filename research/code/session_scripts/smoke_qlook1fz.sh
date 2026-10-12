#!/bin/bash
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SP=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad
rm -rf "$SP/smoke_qlook1fz_out"
env EXPECT_NPROC=3 ARM=qlook1fz SEED=20260830 UPDATES=2 \
    AB_CONFIG=config/public.py:sd35_open3 OUTPUT_DIR="$SP/smoke_qlook1fz_out" \
  sbatch --parsable --job-name=smk_qlook1fz \
    --account=minilab --partition=minilab-gpu --qos=high \
    --gres=gpu:h100:3 --nodelist=ai-gpu14 --cpus-per-task=24 --mem=300G --time=01:30:00 \
    --output=$SP/smk_qlook1fz_%j.out "$O/slurm/ab_open3_v6.slurm"
