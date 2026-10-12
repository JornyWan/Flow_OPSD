#!/bin/bash
# 与 watch_ab_ckpt.sh 并行的**高优先级**评测投递。
#
# 为什么需要它:watch_ab_ckpt.sh 里写死了 `--partition=preempt_gpu --qos=low`,
# 那是全集群最低优先级档 —— a1 的评测走该路径时 SLURM 估计 17:54 才起,
# 改投 sablab-gpu(QOS=sablab-high)后 11:37 就跑了(§4.20)。
# 正在运行的 watcher 不能改(bash 会按偏移量重读脚本,改动可能导致执行错乱),
# 故新起一个并行的。两个都会投,由 opsd_eval.slurm 里的 mkdir 原子锁保证只有一个真跑。
#
# 硬件必须保持 A40:baseline/n1/a1 的评测都在 A40 上做,换卡会破坏可比性。
# 用法: watch_ab_ckpt_hipri.sh <tag> <savedir> <final_step> [extra_steps...]
set -uo pipefail
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
TAG=$1; SAVEDIR=$2; FINAL=$3; shift 3; EXTRA="${@:-}"
CK="$SAVEDIR/checkpoints"
echo "[hipri:$TAG] 等 $CK/checkpoint-$FINAL 落盘 $(date)"
while true; do
  ls "$SAVEDIR" "$CK" >/dev/null 2>&1        # 刷 NFS 属性缓存
  [ -f "$CK/checkpoint-$FINAL/lora/adapter_config.json" ] && break
  sleep 120
done
sleep 60                                      # 宽限:等 ema/optimizer 写完
ls "$CK" >/dev/null 2>&1
HAVE="$FINAL"
for s in $EXTRA; do
  [ -f "$CK/checkpoint-$s/lora/adapter_config.json" ] && HAVE="$s $HAVE"
done
echo "[hipri:$TAG] checkpoint 就绪,评测 step:$HAVE $(date)"
J=$(EVALCFG=sd35_open3 ARM="$TAG" CKROOT="$CK" STEPS="$HAVE" \
    REWARDS=hpsv2,clipscore,aesthetic,pickscore,imagereward \
    sbatch --parsable --job-name="ev_${TAG}_hi" --account=sablab --partition=sablab-gpu \
    --gres=gpu:a40:1 --exclude=ai-gpu01,ai-gpu03 "$O/slurm/opsd_eval.slurm")
echo "[hipri:$TAG] 已提交高优先级评测 job=$J $(date)"
