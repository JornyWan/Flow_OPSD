#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=24 --mem=300G
#SBATCH --gres=gpu:a100:3
#SBATCH --time=03:00:00
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --job-name=smk_odiag
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
# §4.875 code-correctness smoke:同 seed、各 2 次更新。
#   mkF4 diag=2 vs diag=0:update 0 的 opa_policy_loss 逐位相同(诊断不改训练);update 1 只差 GPU 非确定性。
#   update 0 的 od_urms ≈ 1(Adam 第一步 m̂/√v̂ = sign(g));baseline 的 od_gn_x = 0;diag=2 的 od_gn_xb、od_gn_xg 都 > 0;
#   multikF4pos 与 multikF4 在 update 0 同样本同模型 ⇒ od_gn_q 相同、od_gn_x 不同(ADVPOS 进了梯度)。
OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
for spec in multikF4:2 multikF4:0 multikF4pos:1 baseline:1; do
  arm=${spec%%:*}; od=${spec##*:}
  OPT_DIAG=$od ARM=$arm SEED=20260830 UPDATES=2 EXPECT_NPROC=3 OUTPUT_DIR=/midtier/minilab/scratch/chw4026/tmp/smk_odiag_${arm}_d$od \
    bash $OPD_ROOT/slurm/ab_open3_v6.slurm
done
