#!/bin/bash
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$O" || exit 1
CK="$O/outputs/ab_open3_lr8x_s20260830_u100/checkpoints/checkpoint-100"
[ -f "$CK/lora/adapter_config.json" ] || { echo "  ✗ lr8x checkpoint 缺失"; exit 1; }
J=$(ARM="dflt_lr8x" ADAPTER=default CKDIR="$CK" CELLS="drawbench:flow:4 drawbench:flow:10 drawbench:flow:20" \
    REWARDS=hpsv2,clipscore,aesthetic,pickscore,imagereward \
    sbatch --parsable --job-name="lownfe_lr8x" \
    --account=sablab --partition=preempt_gpu --qos=low \
    --gres=gpu:a40:1 --exclude=ai-gpu01,ai-gpu03 "$O/slurm/opsd_eval_matched.slurm" 2>&1)
echo "  lr8x [flow:4/10/20] -> job=$J"
