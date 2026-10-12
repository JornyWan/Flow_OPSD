#!/bin/bash
# RB-LT smoke(3 卡版)。4 卡版排不上:空闲 H100 是 1+3 分散在两节点,--exclusive 要整节点
# ⇒ SLURM 估起跑在 20h 后。整节点请求必须看**逐节点**空闲,不能看跨节点加总(§4.623 §五)。
# 3 卡对 code-correctness 足够:world_size 只影响有效 batch,不影响 RB-LT 的代码路径;
# 48 组 ÷ 3 = 16,正是守卫 case 3|4|6|8 允许 3 的原因。EXPECT_NPROC="3 4" 容许拿到 4。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SP=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad
sub(){
  local arm=$1
  rm -rf "$SP/smoke_${arm}_out"
  env EXPECT_NPROC="3 4" ARM="$arm" SEED=20260830 UPDATES=2 \
      AB_CONFIG=config/public.py:sd35_pickscore \
      OUTPUT_DIR="$SP/smoke_${arm}_out" \
    sbatch --parsable --job-name="smk3_$arm" \
      --account=sablab --partition=preempt_gpu --qos=low \
      --gres=gpu:h100:3 --cpus-per-task=24 --mem=300G --time=02:00:00 \
      --exclude=ai-gpu01,ai-gpu03 \
      --output=$SP/smk3_${arm}_%j.out \
      "$O/slurm/ab_open3_v6.slurm"
}
for a in rbltd0 rblt; do echo "  $(printf '%-7s' $a) job=$(sub $a)"; done
