#!/bin/bash
# canonresume200:canonical 从**自己的** ck200 续训到 300。
# 作用:让"可部署口径"变成可判的 —— anneal200 − canonresume200 两臂**都**有一次 EMA 重启
#      (§4.789 实测该伪效应值 +6.4~+9.1 MDE)⇒ 差异只剩"前 200 步的目标是 MSP 还是 canonical"。
# 代码路径与 anneal200 完全相同(ARM=baseline + OPA_RESUME_DIR),已 smoke 验证(§4.777)⇒ 不另跑 smoke。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SEED=20260830
SRC=$O/outputs/ab_open3_baseline_s20260830_u300/checkpoints/checkpoint-200
[ -f "$SRC/trainer_state.json" ] || { echo "❌ 源 ckpt 不对"; exit 1; }
env EXPECT_NPROC=4 ARM=baseline SEED="$SEED" UPDATES=300 \
    AB_CONFIG=config/public.py:sd35_open3 \
    OPA_RESUME_DIR="$SRC" \
    OUTPUT_DIR="$O/outputs/ab_open3_canonresume200_s${SEED}_u300" \
  sbatch --parsable --job-name=canonresume200 \
    --account=minilab --partition=minilab-gpu --qos=high \
    --exclusive --gres=gpu:h200:4 \
    "$O/slurm/ab_open3_v6.slurm"
