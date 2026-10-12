#!/bin/bash
# 四个判定各差一个评测,但在 qos=low 上排 66–68/71。
# 安全性(本周期实测,不沿用):gpu02/gpu04 上 0 个 QOS=low 作业;
#   yuz4023 的是 sablab-low,而 sablab-high 的 Preempt 列只含 'low' ⇒ 不会被抢。
#   显式 --exclude ai-gpu05(那里有 4 个 lus4017 的 low 作业)。
# 旧的 low 副本保留:slurm 的 [ -s $OUT ] 守卫 + mkdir 锁保证不重复计算。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$O" || exit 1
EX="ai-gpu01,ai-gpu03,ai-gpu05"
sub(){ # arm ckdir cells outname
  local arm=$1 ck=$2 cells=$3 out=$4
  [ -f "$O/outputs/$out" ] && { echo "  $arm: 产物已存在,跳过"; return; }
  [ -f "$ck/lora/adapter_config.json" ] || { echo "  $arm: ✗ checkpoint 缺失"; return; }
  local J
  J=$(ARM="$arm" ADAPTER=default CKDIR="$ck" CELLS="$cells" \
      REWARDS=hpsv2,clipscore,aesthetic,pickscore,imagereward \
      sbatch --parsable --job-name="hp_${arm}" --account=sablab --partition=sablab-gpu \
      --gres=gpu:a40:1 --exclude="$EX" "$O/slurm/opsd_eval_matched.slurm" 2>&1)
  echo "  $arm -> job=$J"
}
sub pi_baserep_20     "$O/outputs/ab_open3_baserep_s20260830_u100/checkpoints/checkpoint-20" \
    "drawbench:flow:40" matched_pi_baserep_20_default_drawbench_flow40.json
sub pi_baserepH100_20 "$O/outputs/ab_open3_baserepH100_s20260830_u100/checkpoints/checkpoint-20" \
    "drawbench:flow:40" matched_pi_baserepH100_20_default_drawbench_flow40.json
sub pi_inner8_160     "$O/outputs/ab_open3_inner8_s20260830_u100/checkpoints/checkpoint-160" \
    "drawbench:flow:40" matched_pi_inner8_160_default_drawbench_flow40.json
