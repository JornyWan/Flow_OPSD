#!/bin/bash
# code-correctness smoke:跨 run 续训(OPA_RESUME_DIR)。
# 查三件事:(a) 从 ck200 起跑而不是从 0;(b) ARM=baseline 下 MSP 确实关掉(MULTIK=none);
# (c) 用 MSP 训出的 checkpoint 喂给 canonical 臂不炸(LoRA/optimizer 形状一致)。
# ⚠️ 独立 OUTPUT_DIR 放 scratchpad,不落 outputs/(否则污染表格生成器与 watcher)。
# 1 卡即可:resume 定位读 trainer_state.json,与 world_size 无关。有效 batch 变小,
# 所以**不比数值**,只验证代码路径。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SP=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad
SRC=$O/outputs/ab_open3_multikF4_s20260830_u300/checkpoints/checkpoint-200
[ -f "$SRC/trainer_state.json" ] || { echo "❌ 源 ckpt 不对: $SRC"; exit 1; }
OUT=$SP/smoke_anneal_out
rm -rf "$OUT"
J=$(env EXPECT_NPROC=1 ARM=baseline SEED=20260830 UPDATES=202 \
      AB_CONFIG=config/public.py:sd35_open3 \
      OPA_RESUME_DIR="$SRC" OUTPUT_DIR="$OUT" \
    sbatch --parsable --job-name=smk_anneal \
      --account=sablab --partition=sablab-gpu --qos=sablab-high \
      --gres=gpu:a100:1 --time=02:00:00 \
      --output=$SP/smk_anneal_%j.out \
      "$O/slurm/ab_open3_v6.slurm" 2>&1)
echo "smoke job=$J  OUT=$OUT  log=$SP/smk_anneal_<job>.out"
