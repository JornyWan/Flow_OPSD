#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=24 --mem=300G
#SBATCH --gres=gpu:a100:3
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --job-name=smk_split
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
# §4.888 code-correctness smoke:multikF4sa(λ=0.5)+ OPT_DIAG=1,3 次更新。
#   过关:sa_gn > 0 且有限、sa_urms ∈ (0, 1.5)、sa_skip 多数为 0、loss 有限、无 Traceback。
OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
MK_SPLIT=0.5 OPT_DIAG=1 ARM=multikF4sa SEED=20260905 UPDATES=3 EXPECT_NPROC="3" OUTPUT_DIR=/midtier/minilab/scratch/chw4026/tmp/smk_split \
  bash $OPD_ROOT/slurm/ab_open3_v6.slurm
