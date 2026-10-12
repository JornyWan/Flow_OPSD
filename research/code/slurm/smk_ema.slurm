#!/bin/bash
#SBATCH --account=minilab
#SBATCH --partition=minilab-gpu
#SBATCH --qos=high
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=24 --mem=300G
#SBATCH --gres=gpu:h100:3
#SBATCH --time=01:30:00
#SBATCH --job-name=smk_ema
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
# §4.860 code-correctness smoke:同 seed、各 2 次更新;canonema 用 EMA_P=1(每步重启)对照 baseline(不重启)。
OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
for arm in baseline canonema multikF4ema; do
  EMA_P=1 ARM=$arm SEED=20260830 UPDATES=2 EXPECT_NPROC=3 OUTPUT_DIR=/midtier/minilab/scratch/chw4026/tmp/smk_ema_$arm \
    bash $OPD_ROOT/slurm/ab_open3_v6.slurm
done
