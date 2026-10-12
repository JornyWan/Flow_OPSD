#!/bin/bash
#SBATCH --account=minilab
#SBATCH --partition=minilab-gpu
#SBATCH --qos=high
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=24 --mem=300G
#SBATCH --gres=gpu:h100:3
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --job-name=smk_unrun
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
# §4.884 code-correctness smoke:vadv / hybridtr / hybridw 各 2 次更新(H100 np3,与 baseH100 / mkF4hr2 同硬件)。
#   vadv:vadv_wp_std > 0 且 vadv_wp_zero、vadv_wp_one 都 < 1(权重不是常数);hybrid 两臂:mk_n_extra = 2、loss 有限、exit 0。
OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
for arm in vadv hybridtr36 hybridw36; do
  OPT_DIAG=1 ARM=$arm SEED=20260830 UPDATES=2 EXPECT_NPROC=3 OUTPUT_DIR=/midtier/minilab/scratch/chw4026/tmp/smk_unrun_$arm \
    bash $OPD_ROOT/slurm/ab_open3_v6.slurm
done
