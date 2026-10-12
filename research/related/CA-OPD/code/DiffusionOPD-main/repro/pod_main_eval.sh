#!/bin/bash
# DiffusionOPD EVAL pod bootstrap. Reuses venv_mopd (all reward envs). Evaluates one or many
# LoRA checkpoints (or base SD3.5-M) with scripts/evaluation.py, uploads average_scores.json +
# evaluation_results.jsonl + sample images to S3.
# Modes (env):
#   single : EVAL_CKPT_S3 = s3 checkpoint-N dir with lora/, OR literal "base" (no LoRA)   OUTNAME=prefix
#   curve  : CKPT_STEPS = space-sep step numbers + CKPT_BASE = s3 .../checkpoints dir; loops,
#            OUTNAME_${step} per point (for metric-vs-step convergence figure)
# Common (env): DATASETS (comma-sep of geneval,ocr,pickscore,drawbench; default drawbench),
#   GS (default 4.5), GSLIST (space-sep gs; loops), EVAL_REWARDS (drawbench/pickscore set),
#   MAXSAMPLES (>0 = first-N subset), NPROC (default 8).
# Guidance per author GitHub issue: student=4.5 all datasets (incl GenEval); GenEval-teacher=1.0
# (DiffusionNFT, CFG-incompat); OCR/Aes-teacher=4.5 (Flow-GRPO); base w/o-CFG=1.0 & +CFG=4.5.
set -euxo pipefail
export AWS_REGION=us-west-2
S3=s3://dit-scale-up/cwan/opd_reproduction
BUNDLE=$S3/bundle
OPD=/mnt/localssd/opd
REPO=/home/colligo/opd_related/DiffusionOPD-main
NPROC=${NPROC:-8}
DATASETS=${DATASETS:-${DATASET:-drawbench}}
GS=${GS:-4.5}
OUTNAME=${OUTNAME:-eval}
VENV=$OPD/venv_mopd/bin

echo "[pod_main_eval] $(date -u) datasets=$DATASETS nproc=$NPROC gs=$GS gslist=${GSLIST:-} steps=${CKPT_STEPS:-} ckpt=${EVAL_CKPT_S3:-}"
mkdir -p "$OPD" /home/colligo /home/colligo/opd_related "$OPD/models"
if [ ! -x "$VENV/python" ]; then
  for f in uvpython venv_mopd hf_cache sd35; do aws s3 cp "$BUNDLE/$f.tar" "/mnt/localssd/$f.tar"; done
  tar xf /mnt/localssd/uvpython.tar  -C /home/colligo
  tar xf /mnt/localssd/venv_mopd.tar -C "$OPD"
  tar xf /mnt/localssd/hf_cache.tar  -C "$OPD"
  tar xf /mnt/localssd/sd35.tar      -C "$OPD/models"
fi
for f in code mmdetection reward_ckpts; do aws s3 cp "$BUNDLE/$f.tar" "/mnt/localssd/$f.tar"; done
rm -rf "$REPO";              tar xf /mnt/localssd/code.tar        -C /home/colligo/opd_related
rm -rf "$OPD/mmdetection";   tar xf /mnt/localssd/mmdetection.tar -C "$OPD"
rm -rf "$REPO/reward_ckpts"; tar xf /mnt/localssd/reward_ckpts.tar -C "$REPO"

export HF_HOME=$OPD/hf_cache TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
cd "$REPO"
HF_HUB_OFFLINE=0 "$VENV/python" -c "import open_clip; open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')" 2>/dev/null || true
export PYTHONPATH="$REPO:${PYTHONPATH:-}"   # evaluation.py imports flow_grpo (run by path)

# run_eval <ckpt_src ("base" | s3 checkpoint dir)> <outname>
run_eval() {
  local src="$1" name="$2" CKPT_ARG="" RGS DS tag OUT
  if [ "$src" = "base" ]; then
    CKPT_ARG=""; echo "[pod_main_eval] BASE mode (no LoRA)"
  else
    local CKDIR=/mnt/localssd/eval_ckpt; rm -rf "$CKDIR"; mkdir -p "$CKDIR/lora"
    aws s3 cp "$src/lora/" "$CKDIR/lora/" --recursive --only-show-errors
    CKPT_ARG="--checkpoint_path $CKDIR"
  fi
  local MULTIGS=0; [ -n "${GSLIST:-}" ] && MULTIGS=1
  for RGS in ${GSLIST:-$GS}; do
    for DS in $(echo "$DATASETS" | tr "," " "); do
      tag="$name"; [ "$MULTIGS" = "1" ] && tag="${name}_gs${RGS}"
      OUT=/mnt/localssd/${tag}_${DS}; mkdir -p "$OUT"
      echo "[pod_main_eval] === EVAL name=$name dataset=$DS gs=$RGS $(date -u) ==="
      "$VENV/python" -m torch.distributed.run --nproc_per_node=$NPROC --master_port=29503 scripts/evaluation.py \
        $CKPT_ARG --model_type sd3 --dataset "$DS" --guidance_scale "$RGS" \
        --max_samples "${MAXSAMPLES:-0}" \
        --mixed_precision fp16 --save_images --output_dir "$OUT" || echo "[pod_main_eval] $DS gs=$RGS rc=$?"
      echo "[pod_main_eval] === RESULTS $DS gs=$RGS ==="; cat "$OUT/average_scores.json" 2>/dev/null || true
      aws s3 sync "$OUT" "$S3/eval/${tag}_${DS}/" --exclude "*.png" --exclude "*.jpg" --only-show-errors || true
      aws s3 sync "$OUT/images" "$S3/eval/${tag}_${DS}/images/" --exclude "*" --include "0000[0-9].jpg" --include "0001[0-9].jpg" --only-show-errors 2>/dev/null || true
    done
  done
}

if [ -n "${RUNS:-}" ]; then
  # RUNS = newline-sep "name|gs|s3src" entries (per-entry gs; s3src must have no spaces). For teacher rows.
  while IFS='|' read -r rname rgs rsrc; do
    [ -z "${rname// /}" ] && continue
    echo "[pod_main_eval] ##### RUN name=$rname gs=$rgs src=$rsrc #####"
    GS="$rgs" GSLIST="" run_eval "$rsrc" "$rname"
  done <<< "$RUNS"
elif [ -n "${CKPT_STEPS:-}" ]; then
  : "${CKPT_BASE:?curve mode needs CKPT_BASE (s3 .../checkpoints dir)}"
  for step in $CKPT_STEPS; do
    echo "[pod_main_eval] ##### CURVE checkpoint-$step #####"
    run_eval "$CKPT_BASE/checkpoint-$step" "${OUTNAME}_${step}"
  done
else
  : "${EVAL_CKPT_S3:?set EVAL_CKPT_S3 (s3 checkpoint dir or \"base\") or use CKPT_STEPS+CKPT_BASE}"
  run_eval "$EVAL_CKPT_S3" "$OUTNAME"
fi
echo "[pod_main_eval] ALL DONE $(date -u) -> $S3/eval/${OUTNAME}*"
