#!/bin/bash
#SBATCH --job-name=diag2
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
# Diagnostic 2: 推理时 alpha 探针(纯测量,单卡即可,不占 H200)
set -uo pipefail
source /midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/env.sh
cd "$OPD_REPO"
echo "[diag2] host=$(hostname) $(date)"
python scripts/diag2_alpha_probe.py \
  --student_lora "" \
  --teacher_lora "$AES_TEACHER_PATH" \
  --prompts_file dataset/pickscore/test.txt \
  --n_prompts ${N_PROMPTS:-24} \
  --alphas 0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0 \
  --probe_steps 2,5,8 \
  --out "$OPD_ROOT/outputs/diag2_alpha_base_vs_aesteacher.json" 2>&1 | grep -avE "[0-9]+%\|"
echo "DIAG2_DONE $(date)"
