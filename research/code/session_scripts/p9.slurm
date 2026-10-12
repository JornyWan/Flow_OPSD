#!/bin/bash
#SBATCH --job-name=p9prof
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_p9_%j.out
set -uo pipefail
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
R=$B/CA-OPD/code/DiffusionOPSD-rpi
PY=$B/../conda_envs/envs/diffusionopd/bin/python
P=$B/CA-OPD/reference_papers/DiffusionOPSD-main/data/pickapic/test.txt
export HF_HUB_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPATH=$B/CA-OPD/code/shim_rpi:$R/src:$R
for tag in canon100 canon200 msp100 msp200; do
  case $tag in msp*) A=multikF4;; canon*) A=baseline;; esac; L=$B/outputs/ab_open3_${A}_s20260830_u300/checkpoints/checkpoint-${tag//[a-z]/}/lora
  echo "===== P8 $tag (lora=$L) ====="
  $PY $B/probes/p9_gap_profile.py --repo $R --lora $L --prompts $P --solver dpm2 \
    --out $B/outputs/p9_prof_${tag}.json --n-prompts 48 --chunk 4 2>&1 | grep -E "wrote|Error|Traceback" | tail -3
  echo "$tag rc=${PIPESTATUS[0]}"
done
