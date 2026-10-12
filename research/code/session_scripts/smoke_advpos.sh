#!/bin/bash
# §4.839 multikF4pos 的 code-correctness smoke:同 seed 各 2 步,multikF4 对 multikF4pos。
# 判据:(A) ep0 两臂逐位相同(rollout/目标/query 态 loss 未被改动)
#       (B) ep1 两臂不同(额外态 r1 夹取确实改变了第一次更新)
#       (C) provenance/echo 里 MKADVPOS=1 只出现在 pos 臂;exit 0
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SP=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad
sub(){ local arm=$1 out=$2 jn=$3
  rm -rf "$SP/$out"
  env EXPECT_NPROC=4 ARM="$arm" SEED=20260830 UPDATES=2 \
      AB_CONFIG=config/public.py:sd35_open3 OUTPUT_DIR="$SP/$out" \
    sbatch --parsable --job-name="$jn" \
      --account=minilab --partition=minilab-gpu --qos=high \
      --exclusive --gres=gpu:h200:4 --time=01:00:00 \
      --output=$SP/${jn}_%j.out "$O/slurm/ab_open3_v6.slurm"; }
echo "  multikF4    job=$(sub multikF4    smoke_mkf4_out   smk_mkf4)"
echo "  multikF4pos job=$(sub multikF4pos smoke_mkpos_out  smk_mkpos)"
