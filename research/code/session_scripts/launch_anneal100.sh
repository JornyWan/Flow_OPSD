#!/bin/bash
# §4.836 第 1 步:只改交接点 200 → 100。照抄 launch_anneal.sh,只换 SRC 与名字。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SP=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad
SEED=20260830
SRC=$O/outputs/ab_open3_multikF4_s20260830_u300/checkpoints/checkpoint-100
[ -f "$SRC/trainer_state.json" ] || { echo "❌ 源 ckpt 不对: $SRC"; exit 1; }
H2(){ local jn=$1 arm=$2 upd=$3 out=$4; shift 4
  env EXPECT_NPROC=4 ARM="$arm" SEED="$SEED" UPDATES="$upd" \
      AB_CONFIG=config/public.py:sd35_open3 \
      OPA_RESUME_DIR="$SRC" OUTPUT_DIR="$out" \
    sbatch --parsable --job-name="$jn" \
      --account=minilab --partition=minilab-gpu --qos=high \
      --exclusive --gres=gpu:h200:4 \
      "$@" "$O/slurm/ab_open3_v6.slurm" 2>&1; }
rm -rf "$SP/smoke_anneal100_out"
SMK=$(H2 smk_ann100 baseline 102 "$SP/smoke_anneal100_out" --time=02:00:00 --output=$SP/smk_ann100_%j.out)
echo "  smoke     job=$SMK"
A=$(H2 anneal100 baseline 300 "$O/outputs/ab_open3_anneal100_s${SEED}_u300" --dependency=afterok:$SMK)
echo "  anneal100 job=$A  dep=afterok:$SMK"
nohup bash "$O/codex_research/watch_matched_ckpt.sh" anneal100 \
  "$O/outputs/ab_open3_anneal100_s${SEED}_u300" 200 250 300 \
  > "$O/outputs/watch_anneal100.log" 2>&1 &
echo "  watcher pid=$!"
