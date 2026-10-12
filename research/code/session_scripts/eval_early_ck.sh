#!/bin/bash
# 早 ckpt 评测:baseline / multikF4 在 H200 训练的 ck20、ck30(同硬件配对)。
# 目的:确认 MSP 的增益在**伪影出现之前**(ck20,网格=1.18)就已成立。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$O" || exit 1
for arm in baseline multikF4; do
  for c in 20 30; do
    CK="$O/outputs/ab_open3_${arm}_s20260830_u100/checkpoints/checkpoint-${c}"
    [ -f "$CK/lora/adapter_config.json" ] || { echo "  ✗ $arm ck$c 缺"; continue; }
    for cells in "drawbench:flow:4 drawbench:flow:40" "unseen1k:flow:4 unseen1k:flow:40"; do
      J=$(ARM="pi_${arm}_${c}" ADAPTER="default" CKDIR="$CK" CELLS="$cells" \
          REWARDS=hpsv2,clipscore,aesthetic,pickscore,imagereward \
          sbatch --parsable --job-name="eck_${arm}${c}" \
          --account=sablab --partition=preempt_gpu --qos=low \
          --gres=gpu:a40:1 --exclude=ai-gpu01,ai-gpu03 "$O/slurm/opsd_eval_matched.slurm" 2>&1)
      echo "  ${arm}@ck${c} [${cells%% *}...] -> job=$J"
    done
  done
done
