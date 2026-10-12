#!/bin/bash
# rbltrand:§4.817 §四 的归因对照。smoke smk3_rand2 四条判据全过(‖d‖ 16.911 对 16.911,+0.0%)。
# 硬件照 rbltg/bridge:minilab-gpu h200 整节点 4 张(§4.590 配对三轴:训练硬件必须同)。
# CK_MULT=1(EXTRA_CFG 不动 inner_epochs/gacc,smoke 日志已核 ckmult=1)。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SEED=20260830
J=$(env EXPECT_NPROC=4 ARM=rbltrand SEED="$SEED" UPDATES=100 \
      AB_CONFIG=config/public.py:sd35_pickscore \
      OUTPUT_DIR="$O/outputs/ab_pickscore_rbltrand_s${SEED}_u100" \
    sbatch --parsable --job-name=rbl_rand \
      --account=minilab --partition=minilab-gpu --qos=high \
      --exclusive --gres=gpu:h200:4 \
      "$O/slurm/ab_open3_v6.slurm")
echo "  rbltrand job=$J"
nohup bash "$O/codex_research/watch_matched_ckpt.sh" rbltrand \
  "$O/outputs/ab_pickscore_rbltrand_s${SEED}_u100" 50 100 \
  > "$O/outputs/watch_rbltrand.log" 2>&1 &
echo "  watcher pid=$!"
