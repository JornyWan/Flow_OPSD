#!/bin/bash
# RB-LT 的 code-correctness smoke。判据(锋利,端到端验符号约定):
#   (A) rbltd0(opa_dir_mode=noop ⇒ d≡0):**第 1 个 update 的 opa_policy_loss 必须 ≈ 0**
#       —— 此时 student 与 old adapter 相同,y_pos=ŷ_θ(z_k,σ)、y_tar=ŷ_old(z_k,σ) 完全相等。
#       若不为 0,说明 z_in / y_tar / σ_k 的某处对不上。
#   (B) rblt(d≠0):第 1 个 update 的 opa_policy_loss 必须 > 0 且量级 ≈ σ_k·|d|。
#   (C) 回显里 RBLT=1;三道 RuntimeError 守卫都不触发;exit 0。
# ⚠️ 独立 OUTPUT_DIR 在 scratchpad,不落 outputs/。H100 够用(只验代码路径,不比数值)。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SP=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad
sub(){ # jobname ARM
  local jn=$1 arm=$2
  rm -rf "$SP/smoke_${arm}_out"
  env EXPECT_NPROC=4 ARM="$arm" SEED=20260830 UPDATES=2 \
      AB_CONFIG=config/public.py:sd35_pickscore \
      OUTPUT_DIR="$SP/smoke_${arm}_out" \
    sbatch --parsable --job-name="$jn" \
      --account=sablab --partition=preempt_gpu --qos=low \
      --exclusive --gres=gpu:h100:4 --time=02:00:00 \
      --exclude=ai-gpu01,ai-gpu03 \
      --output=$SP/smk_${arm}_%j.out \
      "$O/slurm/ab_open3_v6.slurm"
}
echo "  rbltd0(d=0,判据 A) job=$(sub smk_rbltd0 rbltd0)"
echo "  rblt  (d≠0,判据 B) job=$(sub smk_rblt rblt)"
