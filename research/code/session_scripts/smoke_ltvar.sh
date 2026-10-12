#!/bin/bash
# rbltpcfm / rbltbridge 的 code-correctness smoke。判据:
#  (A) 三臂(rblt/pcfm/bridge)的 rblt_sig_b 必须完全相同(同一轮转态)。
#  (B) rblt_dnorm_b 三臂必须完全相同 —— d 的构造没变(都是 ρ‖x‖ 内的 RewardImprove(x)),
#      变的只是"d 怎么进目标"。若 dnorm 不同,说明我动到了 d 本身,是实现错。
#  (C) opa_pos_loss 都 >0 且有限;opa_neg_loss=0;三道守卫不触发;exit 0。
#  (D) bridge 的 pos_loss 预期**显著大于** rblt —— 它的输入在直线上(离原轨迹远),
#      而 y_tar≡x₊ 是个很硬的目标。这条只作合理性观察,不作判据。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SP=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad
sub(){ local arm=$1; rm -rf "$SP/smoke_${arm}_out"
  env EXPECT_NPROC="3 4" ARM="$arm" SEED=20260830 UPDATES=2 \
      AB_CONFIG=config/public.py:sd35_pickscore OUTPUT_DIR="$SP/smoke_${arm}_out" \
    sbatch --parsable --job-name="smk3_$arm" \
      --account=sablab --partition=preempt_gpu --qos=low \
      --gres=gpu:h100:3 --cpus-per-task=24 --mem=300G --time=02:00:00 \
      --exclude=ai-gpu01,ai-gpu03 --output=$SP/smk3_${arm}_%j.out \
      "$O/slurm/ab_open3_v6.slurm"; }
for a in rbltpcfm rbltbridge; do echo "  $(printf '%-11s' $a) job=$(sub $a)"; done
