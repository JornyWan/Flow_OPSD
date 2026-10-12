#!/bin/bash
# 补两个已记录的口径缺口。全部 preempt_gpu + qos=low(不抢占)、A40、排除 gpu01/03。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$O" || exit 1
sub(){ # arm ckdir cells outfile-hint
  local arm=$1 ck=$2 cells=$3 hint=$4
  [ -f "$O/outputs/$hint" ] && { echo "  $arm: 产物已存在,跳过"; return; }
  [ -f "$ck/lora/adapter_config.json" ] || { echo "  $arm: ✗ checkpoint 缺失"; return; }
  local J
  J=$(ARM="$arm" ADAPTER=default CKDIR="$ck" CELLS="$cells" \
      REWARDS=hpsv2,clipscore,aesthetic,pickscore,imagereward \
      sbatch --parsable --job-name="gap_${arm}" \
      --account=sablab --partition=preempt_gpu --qos=low \
      --gres=gpu:a40:1 --exclude=ai-gpu01,ai-gpu03 "$O/slurm/opsd_eval_matched.slurm" 2>&1)
  echo "  $arm [$cells] -> job=$J"
}
K=checkpoints/checkpoint-100
# ① §4.554 表里 qopsd/qbt 的 flow-4 两行空着
sub dflt_qopsd "$O/outputs/ab_open3_qopsd_s20260830_u100/$K" "drawbench:flow:4" \
    matched_dflt_qopsd_default_drawbench_flow4.json
sub dflt_qbt   "$O/outputs/ab_open3_qbt_s20260830_u100/$K"   "drawbench:flow:4" \
    matched_dflt_qbt_default_drawbench_flow4.json
# ② §4.556 记的 qcore 只有 L40 评测,补一次 A40 使其与全表同型号
sub pi_qcore_100 "$O/outputs/ab_open3_qcore_s20260830_u100/$K" "drawbench:flow:40" \
    matched_pi_qcore_100_default_drawbench_flow40.json
