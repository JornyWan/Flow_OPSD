#!/bin/bash
# 等 A/B 臂结束 -> 按 OPSD 原生协议排评测(A40)。
# 与 watch_opsd_arm_eval.sh 的区别:那个按 ARM 名推 EVALCFG,而 A/B 臂名(ab_base/ab_a1)
# 会落到默认 sd35_pickscore,导致 train_reward 标错、对表时标错可比列(见 0041 §⑦)。
# 这里**显式**固定 EVALCFG=sd35_open3。
# 用法: watch_ab_eval.sh <jobid> <tag> <savedir> "<steps>"
set -uo pipefail
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
JID=$1; TAG=$2; SAVEDIR=$3; STEPS=$4
echo "[ab-watch:$TAG] 等 job $JID $(date)"
while [ -n "$(squeue -j "$JID" -h 2>/dev/null)" ]; do sleep 180; done
ST=$(sacct -j "$JID" -X -n --format=State 2>/dev/null | head -1 | tr -d ' ')
echo "[ab-watch:$TAG] job $JID 终态=$ST $(date)"
CK="$SAVEDIR/checkpoints"
ls "$CK" >/dev/null 2>&1                       # 刷 NFS 属性缓存
HAVE=""
for s in $STEPS; do
  [ -f "$CK/checkpoint-$s/lora/adapter_config.json" ] && HAVE="$HAVE $s"
done
if [ -z "$HAVE" ]; then
  echo "[ab-watch:$TAG] ❌ $CK 下无目标 checkpoint: $(ls "$CK" 2>/dev/null | tr '\n' ' ')"; exit 1
fi
echo "[ab-watch:$TAG] 评测 step:$HAVE"
J=$(EVALCFG=sd35_open3 ARM="$TAG" CKROOT="$CK" STEPS="$HAVE" \
    REWARDS=hpsv2,clipscore,aesthetic,pickscore,imagereward \
    sbatch --parsable --job-name="ev_$TAG" --gres=gpu:a40:1 \
    --partition=preempt_gpu --qos=low --requeue "$O/slurm/opsd_eval.slurm")
echo "[ab-watch:$TAG] 已提交评测 job=$J $(date)"
