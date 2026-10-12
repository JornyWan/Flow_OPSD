#!/bin/bash
#SBATCH --job-name=p6cloc
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=ai-gpu14
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_p6c_%j.out
set -uo pipefail
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
R=$B/CA-OPD/code/DiffusionOPSD-rpi
PY=$B/../conda_envs/envs/diffusionopd/bin/python
P=$B/CA-OPD/reference_papers/DiffusionOPSD-main/data/pickapic/test.txt
export HF_HUB_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPATH=$B/CA-OPD/code/shim_rpi:$R/src:$R
echo "===== P6 smoke (2 prompts) ====="
$PY $B/probes/p6_locality.py --repo $R --lora none --prompts $P \
  --out $B/outputs/p6c_smoke.json --n-prompts 2 --chunk 2 --kind pickscore 2>&1 | tail -25
echo "smoke rc=${PIPESTATUS[0]}"
if [ -s $B/outputs/p6c_smoke.json ]; then
  echo "===== P6 full (48 prompts) ====="
  $PY $B/probes/p6_locality.py --repo $R --lora none --prompts $P \
    --out $B/outputs/p6c_locality.json --n-prompts 48 --chunk 2 --kind pickscore 2>&1 | tail -90
  echo "full rc=${PIPESTATUS[0]}"
else
  echo "smoke 未产出 JSON,不跑 full"
fi
