#!/bin/bash
#SBATCH --job-name=opd_eval
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=8:00:00
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
#
# 评测。用法:
#   CKPT=<...>/checkpoint-721  DATASETS=geneval,ocr,drawbench  OUTNAME=mopd721  sbatch opd_eval.slurm
#   CKPT=base GS=1.0 OUTNAME=base_gs1.0 sbatch opd_eval.slurm      # 不加 LoRA 的裸 SD3.5
# 作者建议的 guidance: student=4.5 (所有数据集), GenEval-teacher=1.0, OCR/Aes-teacher=4.5
set -uo pipefail
CKPT=${CKPT:?需要 CKPT=<checkpoint-N 目录> 或 "base"}
DATASETS=${DATASETS:-drawbench}
GS=${GS:-4.5}
OUTNAME=${OUTNAME:-eval}
MAXSAMPLES=${MAXSAMPLES:-0}

source /midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/env.sh
cd "$OPD_REPO"
NPROC=$(nvidia-smi -L | wc -l)
# SLURM 会降级 gres 请求(申请 4 卡实得 3)。评测按 rank 分片,rank 数不整除 max_samples 会
# 让采样器补齐出重复样本(512 -> 513),破坏与原记录的可比性 -> 拿不到 4 卡就 requeue。
if [ "$NPROC" -ne 4 ]; then echo "[guard] 需 4 卡实得 $NPROC -> requeue"; sleep 60; scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
OUTROOT=$OPD_ROOT/outputs/eval
mkdir -p "$OUTROOT"
[ -n "${EVAL_REWARDS:-}" ] && export EVAL_REWARDS

# ⚠️ checkpoint 路径含空格/方括号(来自 run_name) -> 必须用数组,否则被 word-split
CKPT_ARG=()
[ "$CKPT" != "base" ] && CKPT_ARG=(--checkpoint_path "$CKPT")

for DS in $(echo "$DATASETS" | tr ',' ' '); do
  OUT="$OUTROOT/${OUTNAME}_${DS}"; mkdir -p "$OUT"
  echo "[eval] name=$OUTNAME ds=$DS gs=$GS nproc=$NPROC $(date)"
  python -m torch.distributed.run --nproc_per_node=$NPROC --master_port=29503 scripts/evaluation.py \
    "${CKPT_ARG[@]}" --model_type sd3 --dataset "$DS" --guidance_scale "$GS" \
    --max_samples "$MAXSAMPLES" --mixed_precision fp16 --save_images --output_dir "$OUT" \
    || echo "[eval] $DS rc=$?"
  echo "[eval] === $DS 结果 ==="; cat "$OUT/average_scores.json" 2>/dev/null
done
echo "[eval] DONE $(date) -> $OUTROOT/${OUTNAME}_*"
