#!/bin/bash
#SBATCH --job-name=opd_evalrepro
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=10:00:00
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
#
# 迁移验证:用本集群新建的环境重跑原服务器已有评测,看数字能否对上。
# 对齐原协议: n=512, student gs=4.5, 40 步 ODE, EVAL_REWARDS 四项(去掉需要 sglang 服务的 unifiedreward)。
set -uo pipefail
source /midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/env.sh
cd "$OPD_REPO"
NPROC=$(nvidia-smi -L | wc -l)
# SLURM 会降级 gres 请求(申请 4 卡实得 3)。评测按 rank 分片,rank 数不整除 max_samples 会
# 让采样器补齐出重复样本(512 -> 513),破坏与原记录的可比性 -> 拿不到 4 卡就 requeue。
if [ "$NPROC" -ne 4 ]; then echo "[guard] 需 4 卡实得 $NPROC -> requeue"; sleep 60; scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
export OMP_NUM_THREADS=$(( 48 / NPROC ))
export EVAL_REWARDS=pickscore,clipscore,aesthetic,hpsv2
OUTROOT=$OPD_ROOT/outputs/eval
CK=$OPD_ROOT/CA-OPD/checkpoints

# name|checkpoint_path(内含 lora/ 子目录)
TARGETS="
Student|$OPD_MODELS_DIR/DiffusionOPD/Student
mopd931|$CK/mopd/logs/[MOPD + SD3.5-M + PickScore+OCR+GenEval]/checkpoints/checkpoint-931
"
echo "[evalrepro] nproc=$NPROC host=$(hostname) job=$SLURM_JOB_ID $(date)"
while IFS='|' read -r NAME CKPT; do
  [ -z "${NAME// /}" ] && continue
  if [ ! -d "$CKPT/lora" ]; then echo "[evalrepro] ❌ $NAME: 找不到 $CKPT/lora"; continue; fi
  for DS in geneval ocr drawbench; do
    OUT="$OUTROOT/${NAME}_${DS}"; mkdir -p "$OUT"
    echo "[evalrepro] ===== $NAME / $DS  $(date) ====="
    python -m torch.distributed.run --nproc_per_node=$NPROC --master_port=29520 scripts/evaluation.py \
      --checkpoint_path "$CKPT" --model_type sd3 --dataset "$DS" --guidance_scale 4.5 \
      --max_samples 512 --mixed_precision fp16 --output_dir "$OUT" \
      2>&1 | grep -avE "^\s*(Timestep|Eval|Generating|Sampling)?:? *[0-9]+%\|" || echo "[evalrepro] $DS rc=$?"
    echo "[evalrepro] --- $NAME/$DS 结果 ---"; cat "$OUT/average_scores.json" 2>/dev/null; echo
  done
done <<< "$TARGETS"
echo "EVALREPRO_FINISHED $(date)"
