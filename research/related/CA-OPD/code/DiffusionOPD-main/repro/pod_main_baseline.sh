#!/bin/bash
# DiffusionOPD Multi-Task RL BASELINE pod bootstrap: Multi-Task GRPO-Guard / Multi-Task NFT / Cascade NFT.
# Reuses venv_mopd (same reward envs as mopd: mmcv1.7.2+mmdet2.28 geneval, paddleocr, open_clip, hpsv2, pickscore).
# NO teacher LoRAs (these baselines are direct multi-reward RL, not distillation).
# Params (env): LAUNCHER=accelerate|torchrun ; RUN_CFG=<fn in config/baselines.py> ; RUN_NAME=<s3 run dir> ;
#   NPROC (default 8) ; CASCADE_PREV_LORA (cascade stages 2/3; read by config/baselines.py via os.environ).
set -euxo pipefail
export AWS_REGION=us-west-2
S3=s3://dit-scale-up/cwan/opd_reproduction
BUNDLE=$S3/bundle
OPD=/mnt/localssd/opd
REPO=/home/colligo/opd_related/DiffusionOPD-main
LAUNCHER=${LAUNCHER:?set LAUNCHER=accelerate|torchrun}
RUN_CFG=${RUN_CFG:?set RUN_CFG=<config function in config/baselines.py>}
RUN_NAME=${RUN_NAME:-$RUN_CFG}
NPROC=${NPROC:-8}

echo "[bl] $(date -u) host=$(hostname) launcher=$LAUNCHER cfg=$RUN_CFG run=$RUN_NAME prevlora=${CASCADE_PREV_LORA:-none}"
mkdir -p "$OPD" /home/colligo /home/colligo/opd_related "$OPD/models"
if [ ! -x "$OPD/venv_mopd/bin/python" ]; then
  for f in uvpython venv_mopd hf_cache sd35; do echo "[bl] dl $f.tar"; aws s3 cp "$BUNDLE/$f.tar" "/mnt/localssd/$f.tar"; done
  tar xf /mnt/localssd/uvpython.tar  -C /home/colligo
  tar xf /mnt/localssd/venv_mopd.tar -C "$OPD"
  tar xf /mnt/localssd/hf_cache.tar  -C "$OPD"
  tar xf /mnt/localssd/sd35.tar      -C "$OPD/models"
fi
for f in code mmdetection reward_ckpts; do echo "[bl] fetch $f.tar"; aws s3 cp "$BUNDLE/$f.tar" "/mnt/localssd/$f.tar"; done
rm -rf "$REPO";              tar xf /mnt/localssd/code.tar        -C /home/colligo/opd_related
rm -rf "$OPD/mmdetection";   tar xf /mnt/localssd/mmdetection.tar -C "$OPD"       # editable target
rm -rf "$REPO/reward_ckpts"; tar xf /mnt/localssd/reward_ckpts.tar -C "$REPO"     # -> <repo>/reward_ckpts

export HF_HOME=$OPD/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-offline}
VENV=$OPD/venv_mopd/bin
cd "$REPO"
"$VENV/python" -c "import torch; print('[bl] torch',torch.__version__,'cuda',torch.cuda.is_available(),'ngpu',torch.cuda.device_count()); x=torch.randn(64,64,device='cuda'); print('[bl] cuda ok',float((x@x).sum()))"
"$VENV/python" -c "import torch, mmcv, mmcv._ext, mmdet, paddleocr; print('[bl] reward env OK mmcv',mmcv.__version__,'mmdet',mmdet.__version__)"
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "$VENV/python" -c "import open_clip; open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai'); print('[bl] open_clip pre-warmed')" || echo "[bl] WARN open_clip prewarm failed"

mkdir -p "$REPO/logs"
( while true; do sleep 300; aws s3 sync "$REPO/logs" "$S3/runs/$RUN_NAME/logs" --only-show-errors || true; done ) &
trap 'aws s3 sync "$REPO/logs" "$S3/runs/$RUN_NAME/logs" --only-show-errors || true' EXIT

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
if [ "$LAUNCHER" = "accelerate" ]; then
  "$VENV/python" -m accelerate.commands.launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_machines 1 --num_processes "$NPROC" --main_process_port 29500 \
    -m scripts.baselines.train_sd3_GRPO_Guard_multireward --config "config/baselines.py:$RUN_CFG"
else
  "$VENV/python" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port=29500 \
    scripts/baselines/train_nft_sd3_multireward.py --config "config/baselines.py:$RUN_CFG"
fi
