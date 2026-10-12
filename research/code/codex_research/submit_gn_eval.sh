#!/bin/bash
# GN 主臂跑到 step 61 时,与对照 rtcraw(同 step 已有评测)做配对评测。
# 61 是 rtcraw 已评测步集 {61,151,241,331,421,511,601,691,751,781} 的第一个点。
set -uo pipefail
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
G="$O/CA-OPD/code/DiffusionOPD-main/logs/[RTC-GNFWD + SD3.5-M + PickScore]/checkpoints"
T=/midtier/minilab/scratch/chw4026/tmp/gn_targets.tsv
: > $T
for s in 61 151 241; do
  # 只评 GN 侧;对照 rtcraw 的同 step 结果已存在,compare_arms 会自动取公共集
  [ -d "$G/checkpoint-$s/lora" ] && printf 'rtcgnfwd_%s\t%s/checkpoint-%s\t4.5\t256\n' "$s" "$G" "$s" >> $T
done
NL=$(wc -l < $T)
[ "$NL" -eq 0 ] && { echo "无可评 checkpoint"; exit 1; }
sed -e 's/^#SBATCH --job-name=evopsdck/#SBATCH --job-name=evgn/' \
    -e "s/^#SBATCH --array=.*/#SBATCH --array=0-$((NL-1))%3/" \
    -e "s#/midtier/minilab/scratch/chw4026/tmp/opsdck_targets.tsv#$T#" \
    $O/slurm/eval_opsdck.slurm > $O/slurm/eval_gn.slurm
bash -n $O/slurm/eval_gn.slurm || { echo "生成脚本语法错误"; exit 1; }
J=$(sbatch --parsable --partition=minilab-gpu --account=minilab --qos=high --gres=gpu:l40:2 $O/slurm/eval_gn.slurm)
echo "evgn=$J 目标 $NL 行: $(cut -f1 $T | tr '\n' ' ')"
echo "跑完后用: python3 $O/compare_arms.py rtcgnfwd rtcraw"
