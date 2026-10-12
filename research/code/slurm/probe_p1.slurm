#!/bin/bash
#SBATCH --job-name=p1_probe
#SBATCH --partition=sablab-gpu
#SBATCH --account=sablab
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/logs/p1_probe_%j.out
set -uo pipefail
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
R=$B/CA-OPD/code/DiffusionOPSD-rpi
PY=$B/../conda_envs/envs/diffusionopd/bin/python
export HF_HUB_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPATH=$B/CA-OPD/code/shim_rpi:$R/src:$R
L=$B/outputs/ab_open3_baseAA100_s20260830_u100/checkpoints/checkpoint-100/lora
PR=$B/CA-OPD/reference_papers/DiffusionOPSD-main/data/pickapic/test.txt
echo "[p1] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0) job=$SLURM_JOB_ID $(date)"
$PY $B/probes/p1_reward_reachability.py --repo $R --lora $L --prompts $PR \
  --out $B/outputs/probe_p1_reachability.json --n-prompts 48 --chunk 8 --seeds 6
echo "[p1] exit=$? $(date)"
