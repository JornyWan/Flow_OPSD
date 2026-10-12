#!/bin/bash
# srun 版逐图评测 watcher。旧的 watch_matched_ckpt.sh 用 sbatch,且**失败不重试** ——
# 2026-09-08 slurmctld 出现 "I/O error writing script/environment to file",
# batch 脚本落盘全失败而 srun 分配正常,故此版改用 srun 并加重试。
# 评测固定 A40(排除 ai-gpu01/03)以保可比性。
# 用法: watch_srun_eval.sh <TAG> <臂输出目录> <ck 编号> [更多 ck ...]
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
TAG=$1; DIR=$2; shift 2
cd "$O" || exit 1
for STEP in "$@"; do
  CK="$DIR/checkpoints/checkpoint-$STEP"
  for _ in $(seq 1 1440); do
    ls "$DIR" >/dev/null 2>&1
    [ -f "$CK/lora/adapter_config.json" ] && break
    sleep 60
  done
  [ -f "$CK/lora/adapter_config.json" ] || { echo "[sep:$TAG] ck$STEP 超时未出现"; continue; }
  sleep 60
  OUT="$O/outputs/matched_pi_${TAG}_${STEP}_default_drawbench_flow40.json"
  [ -f "$OUT" ] && { echo "[sep:$TAG] ck$STEP 已有结果,跳过"; continue; }
  # 重试直到产物真的出现(最多 20 轮)——srun 会排队等 A40,失败才返回。
  for try in $(seq 1 20); do
    ls "$O/outputs/" >/dev/null 2>&1   # ⚠️ 必须刷 NFS 目录缓存:登录节点看不到计算节点刚写的产物,
                                        #    否则产物已出仍会一直重投(2026-09-08 空跑约 80 个作业)
    [ -f "$OUT" ] && break
    echo "[sep:$TAG] ck$STEP 第 $try 次投递 $(date)"
    ARM="pi_${TAG}_${STEP}" ADAPTER=default CKDIR="$CK" CELLS="drawbench:flow:40" \
    REWARDS=hpsv2,clipscore,aesthetic,pickscore,imagereward \
    srun --job-name="mev_pi_${TAG}_${STEP}" --nodes=1 --ntasks-per-node=1 \
         --cpus-per-task=32 --mem=300G --time=12:00:00 \
         --account=sablab --partition=preempt_gpu --qos=low \
         --gres=gpu:a40:1 --exclude=ai-gpu01,ai-gpu03 \
         bash "$O/slurm/opsd_eval_matched.slurm"
    sleep 120
  done
  [ -f "$OUT" ] && echo "[sep:$TAG] ck$STEP ✅ 产物已出" || echo "[se:$TAG] ck$STEP ❌ 20 次仍无产物"
done
