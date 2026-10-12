#!/bin/bash
# 与官方 OPSD 主表口径对齐的实验批次。
# 官方协议(arXiv-2608.24646v1 app:main_result_details):
#   联合三 reward(open3)= **300 updates**、β=0.1;单 reward = **100 updates**、β=1.0
#   其余(LoRA r32/α64、512、Pick-a-Pic、CFG-free guidance 1.0、10 步 DPM++2M、
#         48 组×24 图、lr 3e-4、ρ=0.10、M_tgt=2、checkpoint EMA、DrawBench 40 步评测)我们已逐项一致。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$O" || exit 1
SEED=20260830
sub(){ # jobname ARM UPDATES AB_CONFIG OUTDIR extra_env
  local jn=$1 arm=$2 upd=$3 cfg=$4 out=$5; shift 5
  local J
  J=$(env "$@" EXPECT_NPROC=4 ARM="$arm" SEED="$SEED" UPDATES="$upd" AB_CONFIG="$cfg" OUTPUT_DIR="$O/outputs/$out" \
      sbatch --parsable --job-name="$jn" \
      --account=minilab --partition=minilab-gpu --qos=high --gres=gpu:h200:4 \
      "$O/slurm/ab_open3_v6.slurm" 2>&1)
  echo "  $(printf '%-16s' "$jn") ARM=$arm upd=$upd $(basename $cfg) -> job=$J"
}
echo "── 优先级 1:open3 @ 300 updates(官方满预算口径)──"
sub o3_300_canon  baseline  300 config/public.py:sd35_open3  ab_open3_baseline_s${SEED}_u300
sub o3_300_msp    multikF4  300 config/public.py:sd35_open3  ab_open3_multikF4_s${SEED}_u300
echo "── 优先级 2:单 reward @ 100 updates(官方 reward-specific 口径,β=1.0 自带)──"
for r in hpsv2 pickscore clipscore; do
  sub rs_${r}_canon baseline 100 config/public.py:sd35_${r} ab_${r}_baseline_s${SEED}_u100
  sub rs_${r}_msp   multikF4 100 config/public.py:sd35_${r} ab_${r}_multikF4_s${SEED}_u100
done
echo "── 优先级 3:β 归因臂(open3 @100,只改 β)──"
sub beta10 opacfg 100 config/public.py:sd35_open3 ab_open3_beta10_s${SEED}_u100 OPA_CFG=--config.beta=1.0
sub beta03 opacfg 100 config/public.py:sd35_open3 ab_open3_beta03_s${SEED}_u100 OPA_CFG=--config.beta=0.3
