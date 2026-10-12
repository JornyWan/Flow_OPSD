#!/bin/bash
# rtcraw43 每到达一个 rtcwarm43 已有评测的 step,就自动排上该点的评测。
# 目标步集与 rtcwarm43 完全一致,保证最终能在**完整步集**上做同种子配对
# (今晚已确认:只用早期 3 点会把效应夸大,5 轴 vs 完整集的 2 轴)。
set -uo pipefail
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
D="$O/CA-OPD/code/DiffusionOPD-main/logs/[RTC-RAWB-s43 + SD3.5-M + PickScore]/checkpoints"
JOB=1036037
for s in 331 421 511 601 691 751 781; do
  # 已评过就跳过
  if [ -f "$O/outputs/eval/rtcraw43_${s}_drawbench/average_scores.json" ]; then
    echo "[skip] step $s 已有评测"; continue
  fi
  # 等该 checkpoint 出现;训练作业没了就停
  while [ ! -d "$D/checkpoint-$s/lora" ]; do
    squeue -j $JOB -h -o %T 2>/dev/null | grep -qE 'RUNNING|PENDING' || { echo "训练作业已离队,停在 step $s"; exit 0; }
    sleep 240
  done
  ls -la "$D" >/dev/null 2>&1; sleep 30   # NFS 可见性宽限(今晚被这个坑咬过)
  T=/midtier/minilab/scratch/chw4026/tmp/rtcraw43_s${s}.tsv
  printf 'rtcraw43_%s\t%s/checkpoint-%s\t4.5\t256\n' "$s" "$D" "$s" > $T
  sed -e 's/^#SBATCH --job-name=evopsdck/#SBATCH --job-name=evr43s/' \
      -e 's/^#SBATCH --array=.*/#SBATCH --array=0-0/' \
      -e "s#/midtier/minilab/scratch/chw4026/tmp/opsdck_targets.tsv#$T#" \
      $O/slurm/eval_opsdck.slurm > $O/slurm/eval_r43_s${s}.slurm
  # 幂等检查原本只看"结果文件是否已存在",拦不住"另一个作业正在写同一目录"
  # (实测 1036363 与 1036360_2 并发写 rtcgnfwd_151,有损坏结果的风险)。
  if squeue -u chw4026 -h -o "%o" 2>/dev/null | grep -q "_s${s}\.slurm"; then
    echo "[skip] step $s 已有同目标作业在跑"; continue
  fi
  bash -n $O/slurm/eval_r43_s${s}.slurm || { echo "step $s 脚本语法错误,跳过"; continue; }
  J=$(sbatch --parsable --partition=minilab-gpu --account=minilab --qos=high --gres=gpu:l40:2 $O/slurm/eval_r43_s${s}.slurm 2>&1)
  echo "step $s -> job $J"
done
echo "全部目标步已排完"
