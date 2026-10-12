#!/bin/bash
#SBATCH --job-name=p10sfx
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=6:00:00
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_p10_%j.out
set -uo pipefail
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
R=$B/CA-OPD/code/DiffusionOPSD-rpi
PY=$B/../conda_envs/envs/diffusionopd/bin/python
P=$R/data/pickapic/test.txt
L=$B/outputs/ab_open3_baseline_s20260830_u300/checkpoints/checkpoint-300/lora
export HF_HUB_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPATH=$B/CA-OPD/code/shim_rpi:$R/src:$R
echo "===== P10 smoke (1 prompt) ====="
$PY $B/probes/p10_suffix_direction.py --repo $R --lora $L --prompts $P \
  --out $B/outputs/p10_smoke.json --n-prompts 1 2>&1 | grep -vE "it/s|%\|" | tail -25
echo "smoke rc=${PIPESTATUS[0]}"
if [ -s $B/outputs/p10_smoke.json ]; then
  echo "===== P10 full (32 prompts) ====="
  $PY $B/probes/p10_suffix_direction.py --repo $R --lora $L --prompts $P \
    --out $B/outputs/p10_suffix.json --n-prompts 32 2>&1 | grep -E "\[p10\]|wrote|Error|Traceback|error" | tail -110
  echo "full rc=${PIPESTATUS[0]}"
fi
