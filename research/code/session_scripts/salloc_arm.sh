#!/bin/bash
# sbatch 落盘故障期间的排队式提交:
#   salloc 阻塞排队 -> 核对 AllocTRES 是否真给满 -> 不满就 scancel 重排 -> 满了才起 step。
# 为什么不用 srun 一步到位:2026-09-08 实测,--nodelist 指到只有 2 张空闲的节点时,
# Slurm 会**只授予 2 张**,而 step 仍按 3 张申请 -> "Invalid generic resource specification"。
# 用法: salloc_arm.sh ARM SEED UPDATES OUTDIR GRES NGPU PART ACCT
set -u
ARM=$1; SEED=$2; UPDATES=$3; OUTDIR=$4; GRES=$5; NGPU=$6; PART=$7; ACCT=$8
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$B" || exit 1
for try in $(seq 1 40); do
  echo "[salloc:$ARM] 第 $try 次排队 $(date)"
  OUT=$(salloc --no-shell --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --mem=300G \
        --time=24:00:00 --gres="$GRES" --partition="$PART" --account="$ACCT" \
        --exclude=ai-gpu01,ai-gpu03 --job-name="$ARM" 2>&1)
  echo "$OUT"
  J=$(echo "$OUT" | grep -oP 'job allocation \K[0-9]+' | head -1)
  [ -n "$J" ] || { echo "[salloc:$ARM] 未拿到分配,60s 后重试"; sleep 60; continue; }
  GOT=$(scontrol show job "$J" 2>/dev/null | grep -oP 'AllocTRES=[^ ]*gres/gpu[^=]*=\K[0-9]+' | head -1)
  GOT=${GOT:-0}
  if [ "$GOT" != "$NGPU" ]; then
    echo "[salloc:$ARM] ⚠️ 只授予 $GOT/$NGPU 张 -> scancel $J 重排"
    scancel "$J"; sleep 90; continue
  fi
  echo "[salloc:$ARM] ✅ job=$J 拿满 $GOT 张,起 step"
  export ARM SEED UPDATES OUTPUT_DIR="$OUTDIR" EXPECT_NPROC="$NGPU"
  exec srun --jobid="$J" --nodes=1 --ntasks=1 bash slurm/ab_open3.slurm
done
echo "[salloc:$ARM] ❌ 40 次仍未拿满"
