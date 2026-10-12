#!/bin/bash
# poseulerrot(RPI_ROTQ=1)的 code-correctness smoke。判据:
#  (A) rblt_sig_b 必须等于 σ 表在下标 1+(epoch%9) 处的值(ep0→idx1≈0.960, ep1→idx2≈0.913)
#      ⇒ 证明轮转生效、且日志里的 σ 与实际用的 σ 是同一个。
#      同时 opa_sigma_q(canonical 自己的)仍为 0.278 ⇒ 两者确实不同。
#  (B) rblt_dnorm_b > 0(目标位移非零);opa_neg_loss = 0(positive-only)。
#  (C) opa_pos_loss 与 canonical 同量级(不是 0,也不是爆掉)。
#  (D) 三道守卫不触发;exit 0。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SP=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad
rm -rf "$SP/smoke_poseulerrot_out"
env EXPECT_NPROC="3 4" ARM=poseulerrot SEED=20260830 UPDATES=2 \
    AB_CONFIG=config/public.py:sd35_pickscore \
    OUTPUT_DIR="$SP/smoke_poseulerrot_out" \
  sbatch --parsable --job-name=smk3_rotq \
    --account=sablab --partition=preempt_gpu --qos=low \
    --gres=gpu:h100:3 --cpus-per-task=24 --mem=300G --time=02:00:00 \
    --exclude=ai-gpu01,ai-gpu03 \
    --output=$SP/smk3_rotq_%j.out \
    "$O/slurm/ab_open3_v6.slurm"
