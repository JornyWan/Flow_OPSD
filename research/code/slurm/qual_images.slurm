#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=preempt_gpu
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
# 定性图:同 prompt / 同 seed(42)/ 同 NFE,只换 checkpoint。
# 只取前 N 条 prompt(cross_eval 的 --n_prompts 是确定性取前 N),几分钟即可。
set -uo pipefail
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
R=$O/CA-OPD/code/DiffusionOPSD-rpi
PY=$O/../conda_envs/envs/diffusionopd/bin/python
# ⚠️ 照抄 opsd_eval_matched.slurm 的环境设置 —— 少了这几行会 ModuleNotFoundError: diffusionopsd
SHIM=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad/opsd_shim
export PYTHONPATH="$SHIM:$R"
export PUBLIC_POLICY_WORLD_SIZE=4 PUBLIC_N_GPUS=4 PUBLIC_LAUNCH_WORLD_SIZE=4
export ZIMAGE_HEAVY_BRIDGE=0 ZIMAGE_HEAVY_DIFF_BRIDGE=0 WANDB_MODE=offline
export REWARD_CKPT_PATH=$O/CA-OPD/code/DiffusionOPD-main/reward_ckpts
NFE=${NFE:-4}
NP=${NP:-40}
cd "$R" || exit 1
for pair in "canonical:$O/outputs/ab_open3_baseline_s20260830_u100/checkpoints/checkpoint-100/lora" \
            "ours:$O/outputs/ab_open3_multikF4_s20260830_u100/checkpoints/checkpoint-100/lora"; do
  tag=${pair%%:*}; ck=${pair#*:}
  out="$O/paper_figs/qual/${tag}_nfe${NFE}"
  [ -d "$out" ] && { echo "[q] $out 已存在,跳过"; continue; }
  echo "[q] ===== $tag  NFE=$NFE  $(date) ====="
  "$PY" scripts/cross_eval.py --ckpt "$ck" \
    --config sd35_open3 --config_file config/public.py \
    --prompts "$R/data/pickapic/test.txt" --prompt_set_name drawbench \
    --protocol_name "qual_flow${NFE}" --num_steps "$NFE" --sampler flow \
    --seed 42 --pickscore_scale raw --n_prompts "$NP" \
    --images_dir "$out" \
    --out "$O/outputs/qual_${tag}_nfe${NFE}.json" --rewards hpsv2,clipscore,pickscore 2>&1 | grep -avE "[0-9]+%\|"
  rc=${PIPESTATUS[0]}; n=$(ls "$out"/*.png 2>/dev/null | wc -l)
  echo "[q] $tag rc=$rc -> $n 张"
  [ "$rc" -ne 0 ] || [ "$n" -eq 0 ] && { echo "[q] ❌ $tag 失败(rc=$rc, png=$n)"; RC=1; }
done
echo "[q] exit=${RC:-0} $(date)"
exit ${RC:-0}
