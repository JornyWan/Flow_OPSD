#!/bin/bash
# 补 lr8x@ck50 的四格(K 轴判据需要同 ckpt 的 flow4 与 flow40)
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$O" || exit 1
CK="$O/outputs/ab_open3_lr8x_s20260830_u100/checkpoints/checkpoint-50"
[ -f "$CK/lora/adapter_config.json" ] || { echo "  ✗ ckpt 缺"; exit 1; }
for cells in "drawbench:flow:4 drawbench:flow:40" "unseen1k:flow:4 unseen1k:flow:40"; do
  J=$(ARM="pi_lr8x_50" ADAPTER="default" CKDIR="$CK" CELLS="$cells" \
      REWARDS=hpsv2,clipscore,aesthetic,pickscore,imagereward \
      sbatch --parsable --job-name="eck_lr8x50" \
      --account=sablab --partition=preempt_gpu --qos=low \
      --gres=gpu:a40:1 --exclude=ai-gpu01,ai-gpu03 "$O/slurm/opsd_eval_matched.slurm" 2>&1)
  echo "  lr8x@ck50 [${cells%% *}...] -> job=$J"
done
