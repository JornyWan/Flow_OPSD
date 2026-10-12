#!/bin/bash
# opsdtopk(RAFT)跑完后,与 opsd_repro 基线在 {20,40,60,80,100} 上做配对评测。
# 单变量:top-k 过滤开/关,同协议同 checkpoint 步。
set -uo pipefail
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
CB=$O/outputs/opsd_repro_sd35_pickscore/checkpoints
CT=$O/outputs/opsd_topk100_sd35_pickscore/checkpoints
T=/midtier/minilab/scratch/chw4026/tmp/topk_targets.tsv
# 竞态修复:作业离开队列时,最后一个 checkpoint 可能还没在 NFS 上可见。
# 实测 opsdtopk 1035879 跑满 100 epoch、checkpoint-100 确实存在,
# 但守候在它刚离队时立刻建表,漏掉了 step 100。这里先 ls 刷属性缓存再等 60s。
ls -la "$CT" "$CB" >/dev/null 2>&1
sleep 60
ls -la "$CT" "$CB" >/dev/null 2>&1
: > $T
for s in 20 40 60 80 100; do
  # 只有**两臂都有**该 step 才纳入,保证配对(避免公共集不一致造成选择效应)
  if [ -d "$CB/checkpoint-$s/lora" ] && [ -d "$CT/checkpoint-$s/lora" ]; then
    printf 'opsdck_%s\t%s/checkpoint-%s\t4.5\t256\n'   "$s" "$CB" "$s" >> $T
    printf 'opsdtopk_%s\t%s/checkpoint-%s\t4.5\t256\n' "$s" "$CT" "$s" >> $T
  else
    echo "[skip] step $s 两臂未同时就绪"
  fi
done
NL=$(wc -l < $T)
[ "$NL" -eq 0 ] && { echo "无可评目标"; exit 1; }
sed -e 's/^#SBATCH --job-name=evopsdck/#SBATCH --job-name=evtopk/' \
    -e "s/^#SBATCH --array=.*/#SBATCH --array=0-$((NL-1))%4/" \
    -e 's#/midtier/minilab/scratch/chw4026/tmp/opsdck_targets.tsv#/midtier/minilab/scratch/chw4026/tmp/topk_targets.tsv#' \
    $O/slurm/eval_opsdck.slurm > $O/slurm/eval_topk.slurm
bash -n $O/slurm/eval_topk.slurm || { echo "生成的脚本语法错误"; exit 1; }
J=$(sbatch --parsable --partition=minilab-gpu --account=minilab --qos=high --gres=gpu:l40:2 $O/slurm/eval_topk.slurm)
echo "evtopk=$J  目标 $NL 行: $(cut -f1 $T | tr '\n' ' ')"
