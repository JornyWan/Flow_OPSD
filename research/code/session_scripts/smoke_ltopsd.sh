#!/bin/bash
# LT-OPSD code-correctness smoke(§4.830)。判据:
#  (A) lt_sig_next = 0.278、lt_h = 0.46488-0.278 ≈ 0.187(核退化推导;若 σ_s=0.009 说明 σ 覆盖没生效)
#  (B) neg_loss 非零(保留双支);并记 pos/neg 是否恒等(β=1 镜像退化,见 opsd-mirror-degeneracy-bug)
#  (C) lt_off_b 与 lt_rew_b 两项分解的量级 —— 起训前就要知道谁主导(§4.830 §三)
#  (D) ltopsd0(noop)下 lt_rew_b 必须 ≈ 0
#  (E) 三道 LT 守卫(缺 ltopsd_eps / h 不为正 / 与 RBLT 互斥)不触发;exit 0
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SP=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad
sub(){ local arm=$1 out=$2 jn=$3
  rm -rf "$SP/$out"
  env EXPECT_NPROC="3 4" ARM="$arm" SEED=20260830 UPDATES=2 \
      AB_CONFIG=config/public.py:sd35_pickscore \
      OUTPUT_DIR="$SP/$out" \
    sbatch --parsable --job-name="$jn" \
      --account=sablab --partition=preempt_gpu --qos=low \
      --gres=gpu:h100:3 --cpus-per-task=24 --mem=300G --time=02:00:00 \
      --exclude=ai-gpu01,ai-gpu03 --output=$SP/${jn}_%j.out \
      "$O/slurm/ab_open3_v6.slurm"; }
echo "  ltopsd  job=$(sub ltopsd  smoke_lt_out  smk_lt)"
echo "  ltopsd0 job=$(sub ltopsd0 smoke_lt0_out smk_lt0)"
