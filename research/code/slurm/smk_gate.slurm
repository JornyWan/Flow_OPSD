#!/bin/bash
#SBATCH --account=minilab
#SBATCH --partition=minilab-gpu
#SBATCH --qos=high
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=24 --mem=300G
#SBATCH --gres=gpu:h100:4
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --job-name=smk_gate
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
# §4.883 code-correctness smoke。
#   (1) multikF4 + OPT_DIAG=2,2 次更新:mk_D_k / mk_C_k / mk_G_k 出现且有限;od_gate_w ≡ 1(门控关)。
#   (2) multikF4gate τ=0.05 + OPT_DIAG=1,12 次更新:step ≥10 时 od_gate_w = 0.05·od_gn_q/od_gn_x < 1,训练照常(loss 有限)。
OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
OPT_DIAG=2 ARM=multikF4 SEED=20260830 UPDATES=2 EXPECT_NPROC="3 4" OUTPUT_DIR=/midtier/minilab/scratch/chw4026/tmp/smk_gate_dc \
  bash $OPD_ROOT/slurm/ab_open3_v6.slurm
MK_GATE=0.05 OPT_DIAG=1 ARM=multikF4gate SEED=20260830 UPDATES=12 EXPECT_NPROC="3 4" OUTPUT_DIR=/midtier/minilab/scratch/chw4026/tmp/smk_gate_g \
  bash $OPD_ROOT/slurm/ab_open3_v6.slurm
