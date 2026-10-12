#!/bin/bash
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$O" || exit 1
OUT="$O/outputs/ab_hps_baseline_s20260830_u100"
J=$(ARM=baseline AB_CONFIG="config/public.py:sd35_hpsv2" OUTPUT_DIR="$OUT" \
    sbatch --parsable --job-name="hpsv2base_h4" \
    --account=minilab --partition=minilab-gpu --qos=high --gres=gpu:h200:4 \
    "$O/slurm/ab_open3_v6.slurm" 2>&1)
echo "  单 reward(sd35_hpsv2, β=1.0 自带)-> job=$J"
