#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=preempt_gpu
#SBATCH --qos=low
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=120G
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
# 定性图 v2:prompt 按"canonical@ck100/NFE=40 逐 prompt 均分最高"选出(规则写死在 make_qual2 里)。
# 三个臂 × 两个预算,同 seed=42、同 sampler,只换 checkpoint。
set -uo pipefail
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
R=$O/CA-OPD/code/DiffusionOPSD-rpi
PY=$O/../conda_envs/envs/diffusionopd/bin/python
SHIM=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad/opsd_shim
export PYTHONPATH="$SHIM:$R"
export PUBLIC_POLICY_WORLD_SIZE=4 PUBLIC_N_GPUS=4 PUBLIC_LAUNCH_WORLD_SIZE=4
export ZIMAGE_HEAVY_BRIDGE=0 ZIMAGE_HEAVY_DIFF_BRIDGE=0 WANDB_MODE=offline
export REWARD_CKPT_PATH=$O/CA-OPD/code/DiffusionOPD-main/reward_ckpts
PF=$O/paper_figs/qual3/prompts_gain.txt
NFE=${NFE:-4}
cd "$R" || exit 1
RC=0
for pair in "base:"; do
  tag=${pair%%:*}; ck=${pair#*:}
  out="$O/paper_figs/qual3/${tag}_nfe${NFE}"
  [ -d "$out" ] && { echo "[q3] $out 已存在,跳过"; continue; }
  echo "[q3] ===== $tag NFE=$NFE $(date) ====="
  "$PY" scripts/cross_eval.py --ckpt "$ck" \
    --config sd35_open3 --config_file config/public.py \
    --prompts "$PF" --prompt_set_name drawbench \
    --protocol_name "qual3_flow${NFE}" --num_steps "$NFE" --sampler flow \
    --seed 42 --pickscore_scale raw --images_dir "$out" \
    --out "$O/outputs/qual3_${tag}_nfe${NFE}.json" --rewards hpsv2,clipscore,pickscore 2>&1 | grep -avE "[0-9]+%\|"
  rc=${PIPESTATUS[0]}; n=$(ls "$out"/*.png 2>/dev/null | wc -l)
  echo "[q3] $tag rc=$rc -> $n 张"
  { [ "$rc" -ne 0 ] || [ "$n" -eq 0 ]; } && { echo "[q3] ❌ $tag 失败"; RC=1; }
done
echo "[q3] exit=$RC $(date)"; exit $RC
