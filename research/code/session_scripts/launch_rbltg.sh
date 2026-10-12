#!/bin/bash
# RB-LT v2 两条臂。smoke 1067821 已过(policy/pos = 0.727/0.764,与预测逐位吻合)。
# 只需两条:
#   rbltg        = 方法 v2
#   poseulerrotg = 同状态集的 ROTQ 对照(_elig 与缩放都在共享路径上 ⇒ 自动同口径)
# **不重跑** rbltd0:d≡0 时 loss≈0,缩放与状态集都不改变它 ⇒ v1 的 B0 结果可沿用(§4.805)。
# **不重跑** poseuler:它不是 RBLT/ROTQ ⇒ v2 改动对它无影响 ⇒ 现有 ck100 四格可复用。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SEED=20260830
sub(){ local jn=$1 arm=$2 out=$3
  env EXPECT_NPROC=4 ARM="$arm" SEED="$SEED" UPDATES=100 \
      AB_CONFIG=config/public.py:sd35_pickscore \
      OUTPUT_DIR="$O/outputs/$out" \
    sbatch --parsable --job-name="$jn" \
      --account=minilab --partition=minilab-gpu --qos=high \
      --exclusive --gres=gpu:h200:4 \
      "$O/slurm/ab_open3_v6.slurm"; }
echo "  rbltg        job=$(sub rbl_rbltg        rblt        ab_pickscore_rbltg_s${SEED}_u100)"
echo "  poseulerrotg job=$(sub rbl_poseulerrotg poseulerrot ab_pickscore_poseulerrotg_s${SEED}_u100)"
