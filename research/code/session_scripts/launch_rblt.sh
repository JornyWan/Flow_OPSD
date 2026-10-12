#!/bin/bash
# RB-LT 三臂,官方 reward-specific 满预算(100 updates),PickScore。
# ⚠️ 用 --hold 而不是 --dependency=afterok:smoke —— afterok 只看退出码,
#    而 smoke 的锋利判据(d=0 ⇒ 第 1 步 loss≈0)是**数值**的。核过数再 scontrol release。
# 硬件:H200×4 整节点(与 §4.709 的 reward-specific 臂同族);--exclusive 防部分授予(§4.727)。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SEED=20260830
sub(){ # jobname ARM
  local jn=$1 arm=$2
  env EXPECT_NPROC=4 ARM="$arm" SEED="$SEED" UPDATES=100 \
      AB_CONFIG=config/public.py:sd35_pickscore \
      OUTPUT_DIR="$O/outputs/ab_pickscore_${arm}_s${SEED}_u100" \
    sbatch --parsable --job-name="$jn" \
      --account=minilab --partition=minilab-gpu --qos=high \
      --exclusive --gres=gpu:h200:4 \
      "$O/slurm/ab_open3_v6.slurm"
}
for a in poseuler rblt rbltd0; do echo "  $(printf '%-9s' $a) job=$(sub rbl_$a $a)  (HELD)"; done
