#!/bin/bash
# 补 multik4@ck100 的 NFE 2/6/10/20(两集),把族形状结论从 ck50 抬到 ck100。
# A40 固定(评测基准硬件);preempt_gpu + qos=low。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$O" || exit 1
CK="$O/outputs/ab_open3_multik4_s20260830_u100/checkpoints/checkpoint-100"
[ -f "$CK/lora/adapter_config.json" ] || { echo "  ✗ ckpt 缺失: $CK"; exit 1; }
for cells in "drawbench:flow:2 drawbench:flow:6" "drawbench:flow:10 drawbench:flow:20" \
             "unseen1k:flow:2 unseen1k:flow:6"   "unseen1k:flow:10 unseen1k:flow:20"; do
  J=$(ARM="pi_multik4_100" ADAPTER="default" CKDIR="$CK" CELLS="$cells" \
      REWARDS=hpsv2,clipscore,aesthetic,pickscore,imagereward \
      sbatch --parsable --job-name="mk4fill" \
      --account=sablab --partition=preempt_gpu --qos=low \
      --gres=gpu:a40:1 --exclude=ai-gpu01,ai-gpu03 "$O/slurm/opsd_eval_matched.slurm" 2>&1)
  echo "  [$cells] -> job=$J"
done
