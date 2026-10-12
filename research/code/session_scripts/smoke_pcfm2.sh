#!/bin/bash
# rbltpcfm 在 v2 代码下的 smoke。判据:
#  (A) 输入**不被扰动**:pcfm 分支里 z_in = z_k ⇒ 与 residual-bridge 的差别只在目标与输入
#      ⇒ 用 pos_loss 的量级区分(§4.802 §二十一:pcfm 在 ep0 是 rblt 的 6.46×)
#  (B) σ 轮转在 8 个 σ≥0.278 的态上(v2 的状态集);ep0=0.960
#  (C) policy/pos 带 σ*/σ 缩放 ⇒ ep0 处应 ≈ (pcfm 的 pos 比值) × 0.29
#  (D) neg_loss=0;三道守卫不触发;exit 0
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SP=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad
rm -rf "$SP/smoke_pcfm2_out"
env EXPECT_NPROC="3 4" ARM=rbltpcfm SEED=20260830 UPDATES=2 \
    AB_CONFIG=config/public.py:sd35_pickscore \
    OUTPUT_DIR="$SP/smoke_pcfm2_out" \
  sbatch --parsable --job-name=smk3_pcfm2 \
    --account=sablab --partition=preempt_gpu --qos=low \
    --gres=gpu:h100:3 --cpus-per-task=24 --mem=300G --time=02:00:00 \
    --exclude=ai-gpu01,ai-gpu03 --output=$SP/smk3_pcfm2_%j.out \
    "$O/slurm/ab_open3_v6.slurm"
