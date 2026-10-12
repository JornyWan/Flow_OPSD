#!/bin/bash
# 补 qopsd/qbt 的逐图评测(§4.523 §六 的待办)。
# 非抢占:preempt_gpu + qos=low(prio 50, preempt 为空)。A40 固定保可比性。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$O" || exit 1
for TAG in qopsd qbt; do
  STEP=100
  CK="$O/outputs/ab_open3_${TAG}_s20260830_u100/checkpoints/checkpoint-$STEP"
  OUT="$O/outputs/matched_pi_${TAG}_${STEP}_default_drawbench_flow40.json"
  if [ -f "$OUT" ]; then echo "  $TAG: 已有逐图结果,跳过"; continue; fi
  [ -f "$CK/lora/adapter_config.json" ] || { echo "  $TAG: ✗ checkpoint 缺失"; continue; }
  J=$(ARM="pi_${TAG}_${STEP}" ADAPTER=default CKDIR="$CK" CELLS="drawbench:flow:40" \
      REWARDS=hpsv2,clipscore,aesthetic,pickscore,imagereward \
      sbatch --parsable --job-name="mev_pi_${TAG}_${STEP}" \
      --account=sablab --partition=preempt_gpu --qos=low \
      --gres=gpu:a40:1 --exclude=ai-gpu01,ai-gpu03 "$O/slurm/opsd_eval_matched.slurm" 2>&1)
  echo "  $TAG -> job=$J"
done
