#!/bin/bash
# RB-LT v2(rbltg)的 code-correctness smoke。判据(§4.803 §四 的两处改动各一条):
#  (A) 状态集:rblt_sig_b 应轮转 σ≥σ*=0.278 的 8 个态 ⇒ ep0=0.960, ep1=0.913(**不再**出现 0.009)
#  (B) 缩放:opa_policy_loss / opa_pos_loss 的比值应从 v1 的 ~2.51 降到 ~0.73(= 2.51 × 0.278/0.960)
#      ⇒ opa_pos_loss 本身**不变**(~0.12),只有 policy_loss 被缩小
#  (C) neg_loss=0;三道守卫不触发;exit 0
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SP=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad
rm -rf "$SP/smoke_rbltg_out"
env EXPECT_NPROC="3 4" ARM=rblt SEED=20260830 UPDATES=2 \
    AB_CONFIG=config/public.py:sd35_pickscore \
    OUTPUT_DIR="$SP/smoke_rbltg_out" \
  sbatch --parsable --job-name=smk3_rbltg \
    --account=sablab --partition=preempt_gpu --qos=low \
    --gres=gpu:h100:3 --cpus-per-task=24 --mem=300G --time=02:00:00 \
    --exclude=ai-gpu01,ai-gpu03 --output=$SP/smk3_rbltg_%j.out \
    "$O/slurm/ab_open3_v6.slurm"
