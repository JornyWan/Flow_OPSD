#!/bin/bash
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$O" || exit 1
for arm in baseline multikF4; do
  CK="$O/outputs/ab_open3_${arm}_s20260830_u100/checkpoints/checkpoint-80"
  [ -f "$CK/lora/adapter_config.json" ] || { echo "  ✗ $arm ck80 缺"; continue; }
  for cells in "drawbench:flow:4 drawbench:flow:40" "unseen1k:flow:4 unseen1k:flow:40"; do
    J=$(ARM="pi_${arm}_80" ADAPTER="default" CKDIR="$CK" CELLS="$cells" \
        REWARDS=hpsv2,clipscore,aesthetic,pickscore,imagereward \
        sbatch --parsable --job-name="eck_${arm}80" \
        --account=sablab --partition=preempt_gpu --qos=low \
        --gres=gpu:a40:1 --exclude=ai-gpu01,ai-gpu03 "$O/slurm/opsd_eval_matched.slurm" 2>&1)
    echo "  ${arm}@ck80 [${cells%% *}...] -> job=$J"
  done
done
