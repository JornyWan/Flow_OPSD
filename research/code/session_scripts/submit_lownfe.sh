#!/bin/bash
# §4.525 低 NFE flow 探针。A40 固定;preempt_gpu + qos=low(不抢占任何人)。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$O" || exit 1
sub(){ # tag ckdir adapter cells
  local tag=$1 ck=$2 ad=$3 cells=$4
  if [ "$ad" != "none" ] && [ ! -f "$ck/lora/adapter_config.json" ]; then
    echo "  ✗ $tag: checkpoint 缺失 $ck"; return; fi
  local J
  J=$(ARM="dflt_${tag}" ADAPTER="$ad" CKDIR="$ck" CELLS="$cells" \
      REWARDS=hpsv2,clipscore,aesthetic,pickscore,imagereward \
      sbatch --parsable --job-name="lownfe_${tag}" \
      --account=sablab --partition=preempt_gpu --qos=low \
      --gres=gpu:a40:1 --exclude=ai-gpu01,ai-gpu03 "$O/slurm/opsd_eval_matched.slurm" 2>&1)
  echo "  $tag [$cells] -> job=$J"
}
C=outputs; K=checkpoints/checkpoint-100
sub basemodel ""                              none    "drawbench:flow:2 drawbench:flow:4 drawbench:flow:6"
sub basenp3   "$O/$C/ab_open3_baseline_np3_s20260830_u100/$K" default "drawbench:flow:2 drawbench:flow:4 drawbench:flow:6"
sub qcore     "$O/$C/ab_open3_qcore_s20260830_u100/$K"        default "drawbench:flow:2 drawbench:flow:4 drawbench:flow:6"
sub lr2x      "$O/$C/ab_open3_lr2x_s20260830_u100/$K"         default "drawbench:flow:4"
