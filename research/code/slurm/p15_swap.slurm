#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=preempt_gpu
#SBATCH --qos=low
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=96G
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=ai-gpu01,ai-gpu03,ai-gpu04
#SBATCH --time=02:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=p15swap
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%A_%a.out
# §4.880 前后缀交换探针。环境与 p13 相同(pristine 仓库 + shim);A40(与 matched_pi 评测同硬件;排除 ai-gpu04 的 VAE 非确定性)。
# 数组下标 = outputs/p15/spec.txt 的行号(0 起):tag C_lora M_lora refC refM set b0 b1
set -uo pipefail
OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
R=$OPD_ROOT/CA-OPD/reference_papers/DiffusionOPSD-main
SHIM=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad/opsd_shim
PY=$OPD_ROOT/../conda_envs/envs/diffusionopd/bin/python
[ -f "$HOME/.secrets/hf_jorny.env" ] && . "$HOME/.secrets/hf_jorny.env" 2>/dev/null
L=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $OPD_ROOT/outputs/p15/spec.txt); set -- $L
TAG=$1; C=$2; M=$3; RC=$4; RM=$5; SET=$6; B0=$7; B1=$8
case $SET in unseen1k) PF=$OPD_ROOT/CA-OPD/code/DiffusionOPSD-rpi/data/pickapic/unseen_train1k.txt ;; drawbench) PF=$R/data/pickapic/test.txt ;; esac
OUT=$OPD_ROOT/outputs/p15/${TAG}_${SET}_b${B0}-${B1}.json
[ -s "$OUT" ] && { echo "[p15] $OUT 已存在,跳过"; exit 0; }
cd "$R" || exit 1
export PYTHONPATH="$SHIM:$R"
export PUBLIC_POLICY_WORLD_SIZE=4 PUBLIC_N_GPUS=4 PUBLIC_LAUNCH_WORLD_SIZE=4
export ZIMAGE_HEAVY_BRIDGE=0 ZIMAGE_HEAVY_DIFF_BRIDGE=0 WANDB_MODE=offline
export REWARD_CKPT_PATH=$OPD_ROOT/CA-OPD/code/DiffusionOPD-main/reward_ckpts
echo "[p15] $TAG $SET b[$B0,$B1) host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0) $(date)"
"$PY" "$OPD_ROOT/probes/p15_swap.py" --repo "$R" --C "$C" --M "$M" --ref-C "$RC" --ref-M "$RM" \
  --prompts "$PF" --set "$SET" --b0 "$B0" --b1 "$B1" --out "$OUT" 2>&1 | grep --line-buffered -avE "[0-9]+%\|"
exit ${PIPESTATUS[0]}
