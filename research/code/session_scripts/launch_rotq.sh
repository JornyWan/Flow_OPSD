#!/bin/bash
# poseulerrot 真臂:主判定的**单变量对照**(与 rblt 只差目标构造)。
# smoke 1067542 已过(σ 轮转 0.960→0.913 与 σ 表吻合、dnorm>0、neg_loss=0、exit 0)。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SEED=20260830
env EXPECT_NPROC=4 ARM=poseulerrot SEED="$SEED" UPDATES=100 \
    AB_CONFIG=config/public.py:sd35_pickscore \
    OUTPUT_DIR="$O/outputs/ab_pickscore_poseulerrot_s${SEED}_u100" \
  sbatch --parsable --job-name=rbl_poseulerrot \
    --account=minilab --partition=minilab-gpu --qos=high \
    --exclusive --gres=gpu:h200:4 \
    "$O/slurm/ab_open3_v6.slurm"
