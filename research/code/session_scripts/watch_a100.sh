#!/bin/bash
# 通用 A100 评测 watcher(§4.319 那个 lr8xs2 专用版的泛化)。
# 用法: watch_a100.sh <标签> <臂输出目录> <ck 编号>
# 输出 matched_pi_<标签>_<ck>_a100_...json,与共享的 A40 watcher 不撞名。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
TAG=$1; DIR=$2; STEP=$3
CK="$DIR/checkpoints/checkpoint-$STEP"
OUT="$O/outputs/matched_pi_${TAG}_${STEP}_a100_default_drawbench_flow40.json"
for _ in $(seq 1 1440); do
  ls "$DIR" >/dev/null 2>&1
  [ -f "$CK/lora/adapter_config.json" ] && break
  sleep 60
done
[ -f "$CK/lora/adapter_config.json" ] || { echo "[a100w:$TAG] ck$STEP 超时"; exit 1; }
sleep 60
[ -f "$OUT" ] && { echo "[a100w:$TAG] 已有结果"; exit 0; }
J=$(ARM="pi_${TAG}_${STEP}_a100" ADAPTER=default CKDIR="$CK" CELLS="drawbench:flow:40" \
    REWARDS=hpsv2,clipscore,aesthetic,pickscore,imagereward \
    sbatch --parsable --job-name="mev_pi_${TAG}_${STEP}_a100" \
    --account=sablab --partition=sablab-gpu \
    --gres=gpu:a100:1 --exclude=ai-gpu01,ai-gpu03 "$O/slurm/opsd_eval_matched.slurm")
echo "[a100w:$TAG] ck$STEP -> A100 评测 job=$J $(date)"
