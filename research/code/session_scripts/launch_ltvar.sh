#!/bin/bash
# 用户原表格第 3/4 行:pcfm(全局插值)/ bridge((ε,x₊) 直线)。
# 对照共用已在跑的 poseulerrotg(同 _elig 状态集 + 同 σ*/σ 缩放)⇒ 不必另排对照。
# smoke:pcfm 已在 v2 代码下过(job 1067858);bridge 的 v1 smoke 过,
#        且它与 pcfm 共用同一套共享改动 ⇒ 下面也给它补一个 v2 smoke。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SEED=20260830
sub(){ local jn=$1 arm=$2 out=$3
  env EXPECT_NPROC=4 ARM="$arm" SEED="$SEED" UPDATES=100 \
      AB_CONFIG=config/public.py:sd35_pickscore \
      OUTPUT_DIR="$O/outputs/$out" \
    sbatch --parsable --job-name="$jn" \
      --account=minilab --partition=minilab-gpu --qos=high \
      --exclusive --gres=gpu:h200:4 "$O/slurm/ab_open3_v6.slurm"; }
echo "  rbltpcfm   job=$(sub rbl_pcfm   rbltpcfm   ab_pickscore_rbltpcfm_s${SEED}_u100)"
echo "  rbltbridge job=$(sub rbl_bridge rbltbridge ab_pickscore_rbltbridge_s${SEED}_u100)"
