#!/bin/bash
#SBATCH --job-name=wbprobe
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad/wbprobe_%j.out
source /midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/env.sh
echo "host=$(hostname) MODE=$WANDB_MODE ENTITY=$WANDB_ENTITY PROJECT=$WANDB_PROJECT"
python - <<'PY'
import wandb, os
r = wandb.init(project=os.environ.get("WANDB_PROJECT","diffusionopsd"),
               name="[CONNECT-PROBE]", tags=["probe"])
print("MODE_ACTUAL =", r.settings.mode)
print("ENTITY_ACTUAL =", r.entity)
print("URL =", r.url)
wandb.log({"probe": 1.0})
wandb.finish()
print("PROBE_OK")
PY
