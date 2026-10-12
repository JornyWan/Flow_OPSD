#!/bin/bash
# §4.773 退火臂:smoke(4×H200,2 步)+ 两条真臂(afterok 依赖)。
# 硬件必须 H200 —— 参考臂 canon300/msp300 都在 ai-gpu17=H200×4(§配对三条轴)。
# --exclusive:§4.727 实测集群会把 gres:4 只授予 3 张,EXPECT_NPROC 守卫拦下后空转 requeue。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SP=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad
SEED=20260830
SRC=$O/outputs/ab_open3_multikF4_s20260830_u300/checkpoints/checkpoint-200
[ -f "$SRC/trainer_state.json" ] || { echo "❌ 源 ckpt 不对: $SRC"; exit 1; }

H2(){ # jobname ARM UPDATES OUTDIR [extra sbatch args...]
  local jn=$1 arm=$2 upd=$3 out=$4; shift 4
  env EXPECT_NPROC=4 ARM="$arm" SEED="$SEED" UPDATES="$upd" \
      AB_CONFIG=config/public.py:sd35_open3 \
      OPA_RESUME_DIR="$SRC" OUTPUT_DIR="$out" \
    sbatch --parsable --job-name="$jn" \
      --account=minilab --partition=minilab-gpu --qos=high \
      --exclusive --gres=gpu:h200:4 \
      "$@" "$O/slurm/ab_open3_v6.slurm" 2>&1
}

rm -rf "$SP/smoke_anneal_out"
SMK=$(H2 smk_anneal baseline 202 "$SP/smoke_anneal_out" --time=02:00:00 --output=$SP/smk_anneal_%j.out)
echo "  smoke            job=$SMK"
A=$(H2 anneal200    baseline 300 "$O/outputs/ab_open3_anneal200_s${SEED}_u300"    --dependency=afterok:$SMK)
echo "  anneal200   (MSP关) job=$A  dep=afterok:$SMK"
B=$(H2 resumectl200 multikF4 300 "$O/outputs/ab_open3_resumectl200_s${SEED}_u300" --dependency=afterok:$SMK)
echo "  resumectl200(MSP开) job=$B  dep=afterok:$SMK"
echo "$SMK $A $B" > "$SP/anneal_jobs.txt"
