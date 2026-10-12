#!/bin/bash
# 与 watch_ab_ckpt_hipri.sh 并行的**第二条投递路径**,只为扩大 A40 节点池。
#
# 起因(2026-09-01):hipri watcher 写死 `--partition=sablab-gpu`,只覆盖 ai-gpu02/04 的 8 张 A40;
# 而 ai-gpu05 的 4 张在 `mosadeghlab-gpu`(我们无权),但它也在 `preempt_gpu` 里。
# 实测当时三个 A40 节点全满、最短剩余 19:45 ⇒ 评测要等约 20 小时,表出不来。
# 本脚本投 preempt_gpu(--qos=low),把可用节点从 8 张扩到 12 张。
#
# 安全性:硬件仍是 **A40**(评测可比性不变);opsd_eval.slurm 里的 mkdir 原子锁
# 保证同一 (ARM,STEPS) 只有一个副本真跑,两条路径不会写同一个 OUT json。
# preempt 被抢占只是这条路径失败,hipri 那条仍在,不影响正确性。
set -uo pipefail
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
TAG=$1; SAVEDIR=$2; FINAL=$3; shift 3; EXTRA="${@:-}"
CK="$SAVEDIR/checkpoints"
echo "[preempt:$TAG] 等 $CK/checkpoint-$FINAL 落盘 $(date)"
while true; do
  ls "$SAVEDIR" "$CK" >/dev/null 2>&1
  [ -f "$CK/checkpoint-$FINAL/lora/adapter_config.json" ] && break
  sleep 120
done
sleep 60
ls "$CK" >/dev/null 2>&1
HAVE="$FINAL"
for s in $EXTRA; do
  [ -f "$CK/checkpoint-$s/lora/adapter_config.json" ] && HAVE="$s $HAVE"
done
echo "[preempt:$TAG] checkpoint 就绪,评测 step:$HAVE $(date)"
J=$(EVALCFG=sd35_open3 ARM="$TAG" CKROOT="$CK" STEPS="$HAVE" \
    REWARDS=hpsv2,clipscore,aesthetic,pickscore,imagereward \
    sbatch --parsable --job-name="ev_${TAG}_pe" --account=sablab --partition=preempt_gpu --qos=low \
    --gres=gpu:a40:1 --requeue --exclude=ai-gpu01,ai-gpu03 "$O/slurm/opsd_eval.slurm")
echo "[preempt:$TAG] 已提交 preempt 评测 job=$J $(date)"
