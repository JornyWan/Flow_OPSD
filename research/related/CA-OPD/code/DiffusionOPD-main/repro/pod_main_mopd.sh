#!/bin/bash
# DiffusionOPD MOPD (3-teacher) pod bootstrap. Uses venv_mopd (adds ocr/geneval reward envs:
# paddleocr + mmcv-full1.7.2[non-editable] + mmdet2.28.2[editable @ /mnt/localssd/opd/mmdetection]).
# Ships reward_ckpts to <repo>/reward_ckpts (reward_ckpt_path.py). Everything open-source; no colligo internals.
set -euxo pipefail
export AWS_REGION=us-west-2
S3=s3://dit-scale-up/cwan/opd_reproduction
BUNDLE=$S3/bundle
OPD=/mnt/localssd/opd
REPO=/home/colligo/opd_related/DiffusionOPD-main
RUN_CFG=${RUN_CFG:-mopd}
RUN_NAME=${RUN_NAME:-$RUN_CFG}
NPROC=${NPROC:-8}

echo "[pod_main_mopd] $(date -u) host=$(hostname) cfg=$RUN_CFG"
mkdir -p "$OPD" /home/colligo /home/colligo/opd_related "$OPD/models"

# Big artifacts — cache-gated on the mopd venv.
if [ ! -x "$OPD/venv_mopd/bin/python" ]; then
  for f in uvpython venv_mopd hf_cache sd35; do
    echo "[pod_main_mopd] downloading $f.tar $(date -u)"; aws s3 cp "$BUNDLE/$f.tar" "/mnt/localssd/$f.tar"
  done
  tar xf /mnt/localssd/uvpython.tar  -C /home/colligo
  tar xf /mnt/localssd/venv_mopd.tar -C "$OPD"
  tar xf /mnt/localssd/hf_cache.tar  -C "$OPD"
  tar xf /mnt/localssd/sd35.tar      -C "$OPD/models"
fi
# Small / editable / config artifacts — always refresh so fixes propagate.
for f in teachers code mmdetection reward_ckpts; do
  echo "[pod_main_mopd] fetch $f.tar $(date -u)"; aws s3 cp "$BUNDLE/$f.tar" "/mnt/localssd/$f.tar"
done
rm -rf "$OPD/models/DiffusionOPD"; tar xf /mnt/localssd/teachers.tar -C "$OPD"
rm -rf "$REPO";                    tar xf /mnt/localssd/code.tar     -C /home/colligo/opd_related
rm -rf "$OPD/mmdetection";         tar xf /mnt/localssd/mmdetection.tar -C "$OPD"    # editable target
rm -rf "$REPO/reward_ckpts";       tar xf /mnt/localssd/reward_ckpts.tar -C "$REPO"  # -> <repo>/reward_ckpts

export HF_HOME=$OPD/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-offline}
VENV=$OPD/venv_mopd/bin

# Sanity: CUDA + full reward env (torch before mmcv; from $REPO so no mmcv-dir shadowing).
cd "$REPO"
"$VENV/python" -c "import torch; print('[pod_main_mopd] torch',torch.__version__,'cuda',torch.cuda.is_available(),'ngpu',torch.cuda.device_count()); x=torch.randn(64,64,device='cuda'); print('[pod_main_mopd] cuda ok',float((x@x).sum()))"
"$VENV/python" -c "import torch, mmcv, mmcv._ext, mmdet; from mmdet.apis import init_detector; import paddleocr; print('[pod_main_mopd] reward env OK mmcv',mmcv.__version__,'mmdet',mmdet.__version__)"

# Pre-warm open_clip ViT-L-14 ONCE (geneval attribute classifier) so the 8 training ranks hit the
# cache instead of racing to download it. Needs internet (open_clip pulls from the OpenAI CDN).
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "$VENV/python" -c "import open_clip; open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai'); print('[pod_main_mopd] open_clip ViT-L-14 pre-warmed')" || echo "[pod_main_mopd] WARN open_clip prewarm failed"

mkdir -p "$REPO/logs"
( while true; do sleep 300; aws s3 sync "$REPO/logs" "$S3/runs/$RUN_NAME/logs" --only-show-errors || true; done ) &
trap 'aws s3 sync "$REPO/logs" "$S3/runs/$RUN_NAME/logs" --only-show-errors || true' EXIT

"$VENV/python" -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_machines 1 --num_processes "$NPROC" --main_process_port 29500 \
  -m scripts.train_sd3_opd --config "config/opd.py:$RUN_CFG"
