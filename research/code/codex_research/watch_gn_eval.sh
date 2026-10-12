#!/bin/bash
# GN 主臂每到达一个对照 rtcraw 已有评测的 step,就自动排上该点评测。
# 目标步集与 rtcraw 一致,使最终能做正式配对检验(现仅 n=1)。
set -uo pipefail
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
G="$O/CA-OPD/code/DiffusionOPD-main/logs/[RTC-GNFWD + SD3.5-M + PickScore]/checkpoints"
JOB=$(cat /midtier/minilab/scratch/chw4026/tmp/rtcgnfwd_jobid 2>/dev/null || echo 1036059)
for s in 241 331 421 511; do
  if [ -f "$O/outputs/eval/rtcgnfwd_${s}_drawbench/average_scores.json" ]; then
    echo "[skip] step $s 已评"; continue
  fi
  while [ ! -d "$G/checkpoint-$s/lora" ]; do
    squeue -j $JOB -h -o %T 2>/dev/null | grep -qE 'RUNNING|PENDING' || { echo "GN 作业已离队,停在 step $s"; exit 0; }
    sleep 240
  done
  ls -la "$G" >/dev/null 2>&1; sleep 30    # NFS 可见性宽限
  T=/midtier/minilab/scratch/chw4026/tmp/gn_s${s}.tsv
  printf 'rtcgnfwd_%s\t%s/checkpoint-%s\t4.5\t256\n' "$s" "$G" "$s" > $T
  sed -e 's/^#SBATCH --job-name=evopsdck/#SBATCH --job-name=evgns/' \
      -e 's/^#SBATCH --array=.*/#SBATCH --array=0-0/' \
      -e "s#/midtier/minilab/scratch/chw4026/tmp/opsdck_targets.tsv#$T#" \
      $O/slurm/eval_opsdck.slurm > $O/slurm/eval_gn_s${s}.slurm
  # 幂等检查原本只看"结果文件是否已存在",拦不住"另一个作业正在写同一目录"
  # (实测 1036363 与 1036360_2 并发写 rtcgnfwd_151,有损坏结果的风险)。
  # 按**脚本名**匹配会漏检:同一目标可能由不同脚本提交
  # (实测 1036399 用 eval_gn241.slurm、1036436 用 eval_gn_s241.slurm,双双评 rtcgnfwd_241)。
  # 改为扫所有在队作业的目标表,按**目标名**判重。
  if for _c in $(squeue -u chw4026 -h -o "%o" 2>/dev/null); do
       [ -f "$_c" ] && grep -ohE '/midtier[^"]*\.tsv' "$_c" 2>/dev/null
     done | sort -u | xargs -r cat 2>/dev/null | cut -f1 | grep -qx "rtcgnfwd_${s}"; then
    echo "[skip] step $s 已有同目标作业在跑"; continue
  fi
  bash -n $O/slurm/eval_gn_s${s}.slurm || { echo "step $s 脚本语法错误"; continue; }
  J=$(sbatch --parsable --partition=minilab-gpu --account=minilab --qos=high --gres=gpu:l40:2 $O/slurm/eval_gn_s${s}.slurm 2>&1)
  echo "step $s -> job $J"
done
echo "GN 目标步已排完"
# 注意:**不要编辑正在执行的本脚本**。bash 按字节偏移增量读取,
# 运行中修改会让它跳到错位的位置,实测(2026-08-29)导致
#   line 34: unexpected EOF while looking for matching `"'
# 而此前已提交的作业不受影响。正确做法:改完后跑一份冻结副本(cp 到 tmp 再执行)。
