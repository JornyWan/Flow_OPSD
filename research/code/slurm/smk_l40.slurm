#!/bin/bash
#SBATCH --account=minilab
#SBATCH --partition=minilab-gpu
#SBATCH --qos=high
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --mem=300G
#SBATCH --gres=gpu:l40:4
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --nodelist=ai-gpu13
#SBATCH --job-name=smk_l40
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
# 48GB 卡(L40)能否跑 open3 训练:显存与速度。transformer 梯度检查点(数学等价)。每格 2 次更新。
#   (1) baseline + GRADCKPT  (2) multikF4gate + GRADCKPT  (3)(4) 若 OOM:再把 reward 反传 micro-batch opa_mb 6→2
OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
( while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> /midtier/minilab/scratch/chw4026/tmp/smk_l40_mem.log; sleep 20; done ) &
MP=$!
export SD3_TRANSFORMER_GRADCKPT=1 MIN_GPU_MIB=40000
for spec in "baseline:" "multikF4gate:" ; do arm=${spec%%:*}
  echo "=== $arm $(date +%s)"
  MK_GATE=0.75 OPT_DIAG=1 ARM=$arm SEED=20260830 UPDATES=2 EXPECT_NPROC="3 4" OUTPUT_DIR=/midtier/minilab/scratch/chw4026/tmp/smk_l40_$arm \
    bash $OPD_ROOT/slurm/ab_open3_v6.slurm
  echo "=== end $arm $(date +%s)"
done
for arm in opacfg multikF4cfg; do
  echo "=== $arm mb2 $(date +%s)"
  OPA_CFG="--config.opsd.opa_mb=2" OPT_DIAG=1 ARM=$arm SEED=20260830 UPDATES=2 EXPECT_NPROC="3 4" OUTPUT_DIR=/midtier/minilab/scratch/chw4026/tmp/smk_l40_${arm}_mb2 \
    bash $OPD_ROOT/slurm/ab_open3_v6.slurm
  echo "=== end $arm $(date +%s)"
done
kill $MP
