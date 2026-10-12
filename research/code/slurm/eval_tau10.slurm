#!/bin/bash
#SBATCH --job-name=evtau10
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --array=0-10%4
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%A_%a.out
#
# 完整评测复现:对每个已有 ckpt 重跑原评测协议,和原服务器记录逐项对比。
# 每个 array task 处理目标表的一行(3 个数据集),4 个 task 并发 = 4 个 A40 节点。
# n 按各目标原始记录对齐(base/teacher/Student=512,各 idea arm=256);gs 按作者建议(GenEval-teacher=1.0,其余 4.5)。
set -uo pipefail
source /midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/env.sh
cd "$OPD_REPO"
NPROC=$(nvidia-smi -L | wc -l)
# SLURM 会降级 gres 请求(申请 4 卡实得 3)。评测按 rank 分片,rank 数不整除 max_samples 会
# 让采样器补齐出重复样本(512 -> 513),破坏与原记录的可比性 -> 拿不到 4 卡就 requeue。
if [ "$NPROC" -ne 4 ]; then echo "[guard] 需 4 卡实得 $NPROC -> requeue"; sleep 60; scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-32} / NPROC ))
export EVAL_REWARDS=pickscore,clipscore,aesthetic,hpsv2
PORT=$(( 29600 + SLURM_ARRAY_TASK_ID * 7 ))

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" /midtier/minilab/scratch/chw4026/tmp/tau10_targets.tsv)
NAME=$(echo "$LINE" | cut -f1); CKPT=$(echo "$LINE" | cut -f2)
GS=$(echo "$LINE"   | cut -f3); NS=$(echo "$LINE"   | cut -f4)
[ -z "$NAME" ] && { echo "[evalall] task $SLURM_ARRAY_TASK_ID 无对应目标"; exit 0; }

# ⚠️ checkpoint 路径含空格与方括号(来自 run_name,如 "[MOPD-CAOPD + SD3.5-M + ...]"),
# 用未加引号的变量拼参数会被 word-split -> argparse 报错退出码 2。必须用数组保留原样。
CKARG=(); [ "$CKPT" != "base" ] && CKARG=(--checkpoint_path "$CKPT")
echo "[evalall] task=$SLURM_ARRAY_TASK_ID name=$NAME gs=$GS n=$NS nproc=$NPROC host=$(hostname) $(date)"

for DS in geneval ocr drawbench; do
  OUT="$OPD_ROOT/outputs/eval/${NAME}_${DS}"
  if [ -f "$OUT/average_scores.json" ]; then echo "[evalall] $NAME/$DS 已有结果,跳过"; continue; fi
  mkdir -p "$OUT"
  echo "[evalall] ===== $NAME / $DS  $(date) ====="
  python -m torch.distributed.run --nproc_per_node=$NPROC --master_port=$PORT scripts/evaluation.py \
    "${CKARG[@]}" --model_type sd3 --dataset "$DS" --guidance_scale "$GS" \
    --max_samples "$NS" --mixed_precision fp16 --output_dir "$OUT" \
    2>&1 | grep -avE "[0-9]+%\|" || echo "[evalall] $NAME/$DS rc=$?"
  echo "[evalall] --- $NAME/$DS ---"; cat "$OUT/average_scores.json" 2>/dev/null | tr -d '\n '; echo
done
echo "[evalall] TASK_DONE $NAME $(date)"
