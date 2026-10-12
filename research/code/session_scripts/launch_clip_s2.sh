#!/bin/bash
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
OUT="$O/outputs/ab_clipscore_baseline_s20260905_u100"
J=$(env EXPECT_NPROC=4 ARM=baseline SEED=20260905 UPDATES=100 \
      AB_CONFIG=config/public.py:sd35_clipscore OUTPUT_DIR="$OUT" \
    sbatch --parsable --job-name=rs_clip_s2 \
      --account=minilab --partition=minilab-gpu --qos=high \
      --exclusive --gres=gpu:h200:4 "$O/slurm/ab_open3_v6.slurm")
nohup bash "$O/codex_research/watch_matched_ckpt.sh" rsclipscorecanons2 "$OUT" 50 100 > "$O/outputs/watch_rsclips2.log" 2>&1 &
echo "  (b2) clipscore seed2 job=$J watcher=$!"
