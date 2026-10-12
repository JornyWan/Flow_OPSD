#!/bin/bash
# DiffusionOPD reproduction — Pluto pod bootstrap.
# Fetched from S3 by the job's MAIN script, then run. Downloads the self-contained
# bundle (uv venv + uv python + HF cache + teacher LoRAs + code + SD3.5), stages it at
# the SAME absolute paths it was built at (so venv shebangs/symlinks resolve), sanity-
# checks CUDA, then launches accelerate. NOTE: nothing here uses colligo internal models
# or data — everything is the open-source DiffusionOPD stack.
set -euxo pipefail

export AWS_REGION=us-west-2
S3=s3://dit-scale-up/cwan/opd_reproduction
BUNDLE=$S3/bundle
OPD=/mnt/localssd/opd
RUN_CFG=${RUN_CFG:-sopd_pickscore}          # config/opd.py:<RUN_CFG>
RUN_NAME=${RUN_NAME:-$RUN_CFG}
NPROC=${NPROC:-8}

echo "[pod_main] $(date -u) host=$(hostname) cfg=$RUN_CFG nproc=$NPROC"
mkdir -p "$OPD" /home/colligo /home/colligo/opd_related "$OPD/models"

# 1. Fetch + extract bundle at original absolute paths (skip if already staged on this pod).
if [ ! -x "$OPD/venv/bin/python" ]; then
  for f in uvpython venv hf_cache teachers code sd35; do
    echo "[pod_main] downloading $f.tar $(date -u)"
    aws s3 cp "$BUNDLE/$f.tar" "/mnt/localssd/$f.tar"
  done
  tar xf /mnt/localssd/uvpython.tar -C /home/colligo                 # -> /home/colligo/.local/share/uv/python/...
  tar xf /mnt/localssd/venv.tar     -C "$OPD"                        # -> /mnt/localssd/opd/venv
  tar xf /mnt/localssd/hf_cache.tar -C "$OPD"                        # -> /mnt/localssd/opd/hf_cache
  tar xf /mnt/localssd/teachers.tar -C "$OPD"                        # -> /mnt/localssd/opd/models/DiffusionOPD/...
  tar xf /mnt/localssd/code.tar     -C /home/colligo/opd_related     # -> .../DiffusionOPD-main
  tar xf /mnt/localssd/sd35.tar     -C "$OPD/models"                 # -> /mnt/localssd/opd/models/stable-diffusion-3.5-medium
fi

# 2. Environment (use the shipped HF cache offline; no HF token needed on the pod).
export HF_HOME=$OPD/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-offline}
VENV=$OPD/venv/bin

# 3. CUDA sanity check (our cu126 wheels on the cluster's 12.4 driver, via minor-version compat).
"$VENV/python" -c "import torch; print('[pod_main] torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count()); x=torch.randn(64,64,device='cuda'); print('[pod_main] cuda matmul ok:', float((x@x).sum()))"

# 4. Periodic checkpoint/log sync to S3 (localssd is wiped on job end).
mkdir -p /home/colligo/opd_related/DiffusionOPD-main/logs
( while true; do sleep 300; aws s3 sync "/home/colligo/opd_related/DiffusionOPD-main/logs" "$S3/runs/$RUN_NAME/logs" --only-show-errors || true; done ) &
SYNC_PID=$!
trap 'aws s3 sync /home/colligo/opd_related/DiffusionOPD-main/logs "$S3/runs/$RUN_NAME/logs" --only-show-errors || true; kill $SYNC_PID 2>/dev/null || true' EXIT

# 5. Launch training (accelerate, single node, DDP fp16 — matches scripts/single_node/sopd.sh).
cd /home/colligo/opd_related/DiffusionOPD-main
"$VENV/python" -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_machines 1 --num_processes "$NPROC" --main_process_port 29500 \
  -m scripts.train_sd3_opd --config "config/opd.py:$RUN_CFG"
