#!/bin/bash
# rbltrand 的 code-correctness smoke。判据(§4.817 §五):
#  (A) ‖d‖:ep0 的 rblt_dnorm_b 与 rbltg 的 16.911 比较;差 >10% 必须披露为混淆
#  (B) σ 轮转在 8 个 σ≥0.278 的态上;ep0=0.960(与另三变体同)
#  (C) neg_loss≡0(opa_dual_neg=0 生效);三道消费端守卫不触发;exit 0
#  (D) provenance 里 extra_cfg 含 opa_dir_mode=rand
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SP=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad
rm -rf "$SP/smoke_rand2_out"
env EXPECT_NPROC="3 4" ARM=rbltrand SEED=20260830 UPDATES=2 \
    AB_CONFIG=config/public.py:sd35_pickscore \
    OUTPUT_DIR="$SP/smoke_rand2_out" \
  sbatch --parsable --job-name=smk3_rand2 \
    --account=sablab --partition=preempt_gpu --qos=low \
    --gres=gpu:h100:3 --cpus-per-task=24 --mem=300G --time=02:00:00 \
    --exclude=ai-gpu01,ai-gpu03 --output=$SP/smk3_rand2_%j.out \
    "$O/slurm/ab_open3_v6.slurm"
