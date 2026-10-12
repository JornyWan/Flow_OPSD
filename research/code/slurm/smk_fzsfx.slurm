#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=preempt_gpu
#SBATCH --qos=low
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=24 --mem=300G
#SBATCH --gres=gpu:h100:3
#SBATCH --time=01:30:00
#SBATCH --job-name=smk_fzsfx
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
# §4.855 (a) 的 code-correctness smoke:两臂各 2 次更新,独立 OUTPUT_DIR(不进 outputs/)。
OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
for arm in qlook1fzsfx; do
  ARM=$arm SEED=20260830 UPDATES=2 EXPECT_NPROC=3 OUTPUT_DIR=/midtier/minilab/scratch/chw4026/tmp/smk_$arm \
    bash $OPD_ROOT/slurm/ab_open3_v6.slurm
done
