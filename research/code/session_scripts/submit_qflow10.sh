#!/bin/bash
# 补 qopsd/qbt 的 flow-10 格子(§4.527 轴 C 的空缺)。非抢占,A40。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$O" || exit 1
for TAG in qopsd qbt; do
  OUT="$O/outputs/matched_dflt_${TAG}_default_drawbench_flow10.json"
  [ -f "$OUT" ] && { echo "  $TAG: 已有,跳过"; continue; }
  CK="$O/outputs/ab_open3_${TAG}_s20260830_u100/checkpoints/checkpoint-100"
  [ -f "$CK/lora/adapter_config.json" ] || { echo "  $TAG: ✗ ck 缺失"; continue; }
  J=$(ARM="dflt_${TAG}" ADAPTER=default CKDIR="$CK" CELLS="drawbench:flow:10" \
      REWARDS=hpsv2,clipscore,aesthetic,pickscore,imagereward \
      sbatch --parsable --job-name="qflow10_${TAG}" \
      --account=sablab --partition=preempt_gpu --qos=low \
      --gres=gpu:a40:1 --exclude=ai-gpu01,ai-gpu03 "$O/slurm/opsd_eval_matched.slurm" 2>&1)
  echo "  $TAG flow:10 -> job=$J"
done
