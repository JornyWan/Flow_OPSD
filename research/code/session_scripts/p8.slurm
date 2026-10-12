#!/bin/bash
#SBATCH --job-name=p8gap
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_p8_%j.out
set -uo pipefail
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
R=$B/CA-OPD/code/DiffusionOPSD-rpi
PY=$B/../conda_envs/envs/diffusionopd/bin/python
P=$B/CA-OPD/reference_papers/DiffusionOPSD-main/data/pickapic/test.txt
export HF_HUB_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPATH=$B/CA-OPD/code/shim_rpi:$R/src:$R
for tag in base msp200; do
  if [ $tag = base ]; then L=none; else L=$B/outputs/ab_open3_multikF4_s20260830_u300/checkpoints/checkpoint-200/lora; fi
  echo "===== P8 $tag (lora=$L) ====="
  $PY $B/probes/p8_msp_gap.py --repo $R --lora $L --prompts $P --solver dpm2 \
    --out $B/outputs/p8_gap_${tag}.json --n-prompts 48 --chunk 4 2>&1 | grep -E "\[p8\] 48/|wrote|Error|Traceback" | tail -6
  echo "$tag rc=${PIPESTATUS[0]}"
done
