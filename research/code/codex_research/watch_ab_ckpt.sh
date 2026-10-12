#!/bin/bash
# 等某条 A/B 臂的最终 checkpoint 落盘 -> 排评测。
# 与 watch_ab_eval.sh 的区别:那个盯**单个 job id**,而现在每条臂有 3 个副本
# (投到 sablab/minilab/scu 三个分区,先跑到的赢),盯 job id 会盯错。
# 这里盯**产物**,与哪个副本跑无关。
# 用法: watch_ab_ckpt.sh <tag> <savedir> <final_step> [extra_steps...]
set -uo pipefail
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
TAG=$1; SAVEDIR=$2; FINAL=$3; shift 3; EXTRA="${@:-}"
CK="$SAVEDIR/checkpoints"
echo "[ck:$TAG] 等 $CK/checkpoint-$FINAL 落盘 $(date)"
while true; do
  ls "$SAVEDIR" "$CK" >/dev/null 2>&1        # 刷 NFS 属性缓存
  [ -f "$CK/checkpoint-$FINAL/lora/adapter_config.json" ] && break
  sleep 180
done
sleep 60                                      # 宽限:等 ema/optimizer 等文件写完
ls "$CK" >/dev/null 2>&1
HAVE="$FINAL"
for s in $EXTRA; do
  [ -f "$CK/checkpoint-$s/lora/adapter_config.json" ] && HAVE="$s $HAVE"
done
echo "[ck:$TAG] checkpoint 就绪,评测 step:$HAVE $(date)"
J=$(EVALCFG=sd35_open3 ARM="$TAG" CKROOT="$CK" STEPS="$HAVE" \
    REWARDS=hpsv2,clipscore,aesthetic,pickscore,imagereward \
    sbatch --parsable --job-name="ev_$TAG" --gres=gpu:a40:1 \
    --partition=preempt_gpu --qos=low --requeue "$O/slurm/opsd_eval.slurm")
echo "[ck:$TAG] 已提交评测 job=$J $(date)"
