#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=preempt_gpu
#SBATCH --qos=low
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=96G
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=p13ps
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%A_%a.out
# §4.869 PS1:策略空间搜索探针。环境与 p11_abcd_eval.slurm 相同(pristine 仓库 + shim)。数组下标 = 样本分片。
set -uo pipefail
OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
R=$OPD_ROOT/CA-OPD/reference_papers/DiffusionOPSD-main
SHIM=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad/opsd_shim
PY=$OPD_ROOT/../conda_envs/envs/diffusionopd/bin/python
CK300=$OPD_ROOT/outputs/ab_open3_baseline_s20260830_u300/checkpoints/checkpoint-300
TAG=${TAG:?}; NSHARD=${NSHARD:-1}; SH=${SLURM_ARRAY_TASK_ID:-0}
[ -f "$HOME/.secrets/hf_jorny.env" ] && . "$HOME/.secrets/hf_jorny.env" 2>/dev/null
cd "$R" || exit 1
export PYTHONPATH="$SHIM:$R"
export PUBLIC_POLICY_WORLD_SIZE=4 PUBLIC_N_GPUS=4 PUBLIC_LAUNCH_WORLD_SIZE=4
export ZIMAGE_HEAVY_BRIDGE=0 ZIMAGE_HEAVY_DIFF_BRIDGE=0 WANDB_MODE=offline
export REWARD_CKPT_PATH=$OPD_ROOT/CA-OPD/code/DiffusionOPD-main/reward_ckpts
mkdir -p "$OPD_ROOT/outputs/p13"
OUT=$OPD_ROOT/outputs/p13/${TAG}_sh${SH}of${NSHARD}.json
[ -s "$OUT" ] && { echo "[p13] $OUT 已存在,跳过"; exit 0; }
echo "[p13] tag=$TAG shard=$SH/$NSHARD host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0) $(date)"
"$PY" "$OPD_ROOT/probes/p13_psearch.py" --repo "$R" --pre "$CK300/lora" --dirs "$OPD_ROOT/outputs/p11/dirs_main.pt" \
  --prompts "${PROMPTS:?}" --tag "$TAG" --shard "$SH" --nshard "$NSHARD" --out "$OUT" ${EXTRA:-} \
  2>&1 | grep --line-buffered -avE "[0-9]+%\|"
exit ${PIPESTATUS[0]}
