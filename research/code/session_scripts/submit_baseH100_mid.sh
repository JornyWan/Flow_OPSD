#!/bin/bash
# 补 baseH100 的 ck20/ck50 A40 评测,使 baserepH100 在 ck20/ck50 也能配对(现只有 ck100)。
# preempt_gpu + qos=low ⇒ 不抢占任何人。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$O" || exit 1
for ck in 20 50; do
  OUT="$O/outputs/matched_pi_baseH100_${ck}_default_drawbench_flow40.json"
  [ -f "$OUT" ] && { echo "  ck$ck 已有,跳过"; continue; }
  CK="$O/outputs/ab_open3_baseH100_s20260830_u100/checkpoints/checkpoint-$ck"
  [ -f "$CK/lora/adapter_config.json" ] || { echo "  ck$ck ✗ checkpoint 缺失"; continue; }
  J=$(ARM="pi_baseH100_${ck}" ADAPTER=default CKDIR="$CK" CELLS="drawbench:flow:40" \
      REWARDS=hpsv2,clipscore,aesthetic,pickscore,imagereward \
      sbatch --parsable --job-name="mev_pi_baseH100_${ck}" \
      --account=sablab --partition=preempt_gpu --qos=low \
      --gres=gpu:a40:1 --exclude=ai-gpu01,ai-gpu03 "$O/slurm/opsd_eval_matched.slurm" 2>&1)
  echo "  ck$ck -> job=$J"
done
