#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=preempt_gpu
#SBATCH --qos=low
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=24 --mem=300G
#SBATCH --gres=gpu:h100:3
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --time=01:30:00
#SBATCH --job-name=smk_tpd
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
# §4.868 code-correctness smoke:TPD(M=8),各 2 次更新,独立 OUTPUT_DIR。
OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
for arm in tpdF4; do
  ARM=$arm SEED=20260830 UPDATES=2 EXPECT_NPROC=3 OUTPUT_DIR=/midtier/minilab/scratch/chw4026/tmp/smk_$arm \
    bash $OPD_ROOT/slurm/ab_open3_v6.slurm
done
