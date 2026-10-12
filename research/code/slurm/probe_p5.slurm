#!/bin/bash
#SBATCH --job-name=p5_probe
#SBATCH --partition=sablab-gpu
#SBATCH --account=sablab
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/logs/p5_probe_%j.out
set -uo pipefail
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
R=$B/CA-OPD/code/DiffusionOPSD-rpi
PY=$B/../conda_envs/envs/diffusionopd/bin/python
export HF_HUB_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$B/CA-OPD/code/shim_rpi:$R/src:$R
L=$B/outputs/ab_open3_baseAA100_s20260830_u100/checkpoints/checkpoint-100/lora
PR=$B/CA-OPD/reference_papers/DiffusionOPSD-main/data/pickapic/test.txt
echo "[p5] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0) job=$SLURM_JOB_ID $(date)"
$PY $B/probes/p5_bestofn.py --repo $R --lora $L --prompts $PR \
  --out $B/outputs/probe_p5_bestofn.json --n-prompts 64 --chunk 4 --max-n 64 --grad-mb 2
echo "[p5] exit=$? $(date)"
