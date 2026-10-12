#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=24 --mem=300G
#SBATCH --gres=gpu:a100:3
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --job-name=smk_stepcap
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
# §4.875 修复臂 B 的 code-correctness smoke:正式 batch(slurm 会 unset PUBLIC_NUM_GROUPS)、14 次更新、c=0.9 ⇒ 第 10 次之后必然有截断。
#   核:sc_scale<1 的步 sc_dn·sc_scale = 0.9·sc_ref;训练照常往下走(loss 有限)。
OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
STEP_CAP=0.9 OPT_DIAG=1 ARM=multikF4cap SEED=20260830 UPDATES=14 EXPECT_NPROC=3 \
  OUTPUT_DIR=/midtier/minilab/scratch/chw4026/tmp/smk_stepcap bash $OPD_ROOT/slurm/ab_open3_v6.slurm
