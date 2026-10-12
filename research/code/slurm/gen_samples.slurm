#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
#
# 存图用。此前所有评测都没传 --images_dir,1000 张图打完分就丢了 ——
# 二十多个 checkpoint 评过,一张自己生成的图都没留下。这里补上定性对照。
# seed 固定 42 ⇒ 各 checkpoint 共享同一批初始 latent,可以逐图并排比。
set -uo pipefail
OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
R=$OPD_ROOT/CA-OPD/reference_papers/DiffusionOPSD-main
SHIM=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad/opsd_shim
PY=$OPD_ROOT/../conda_envs/envs/diffusionopd/bin/python
NP=${NP:-12}
OUTROOT=$OPD_ROOT/outputs/samples
[ -f "$HOME/.secrets/hf_jorny.env" ] && . "$HOME/.secrets/hf_jorny.env" 2>/dev/null
cd "$R" || exit 1
export PYTHONPATH="$SHIM:$R"
export PUBLIC_POLICY_WORLD_SIZE=4 PUBLIC_N_GPUS=4 PUBLIC_LAUNCH_WORLD_SIZE=4
export ZIMAGE_HEAVY_BRIDGE=0 ZIMAGE_HEAVY_DIFF_BRIDGE=0 WANDB_MODE=offline
export REWARD_CKPT_PATH=$OPD_ROOT/CA-OPD/code/DiffusionOPD-main/reward_ckpts

# tag : lora 路径(空 = 未训练底模) : sampler : steps
SPECS=(
  "base::flow:40"
  "baseline:$OPD_ROOT/outputs/ab_open3_baseline_np3_s20260830_u100/checkpoints/checkpoint-100/lora:flow:40"
  "n1:$OPD_ROOT/outputs/ab_open3_n1_s20260830_u100/checkpoints/checkpoint-100/lora:flow:40"
  "qopsd:$OPD_ROOT/outputs/ab_open3_qopsd_s20260830_u100/checkpoints/checkpoint-100/lora:flow:40"
  "baseline_dpm10:$OPD_ROOT/outputs/ab_open3_baseline_np3_s20260830_u100/checkpoints/checkpoint-100/lora/old:dpm2:10"
  "qopsd_dpm10:$OPD_ROOT/outputs/ab_open3_qopsd_s20260830_u100/checkpoints/checkpoint-100/lora/old:dpm2:10"
)
_RC=0
for spec in "${SPECS[@]}"; do
  TAG=${spec%%:*}; rest=${spec#*:}
  CK=${rest%%:*}; rest=${rest#*:}
  SAMP=${rest%%:*}; NST=${rest##*:}
  D="$OUTROOT/$TAG"
  if [ -f "$D/prompts.jsonl" ] && [ "$(ls "$D"/*.png 2>/dev/null | wc -l)" -ge "$NP" ]; then
    echo "[gen] $TAG 已有 $NP 张,跳过"; continue
  fi
  [ -n "$CK" ] && [ ! -f "$CK/adapter_config.json" ] && { echo "[gen] ❌ $TAG: $CK 无 adapter_config.json"; _RC=1; continue; }
  echo "[gen] ===== $TAG  ${SAMP}-${NST}  $(date) ====="
  "$PY" scripts/cross_eval.py ${CK:+--ckpt "$CK"} \
    --config sd35_open3 --config_file config/public.py \
    --prompts "$R/data/pickapic/test.txt" --prompt_set_name drawbench \
    --protocol_name "samples_${SAMP}${NST}" \
    --n_prompts "$NP" --num_steps "$NST" --sampler "$SAMP" \
    --seed 42 --pickscore_scale raw --images_dir "$D" \
    --out "$OUTROOT/scores_${TAG}.json" 2>&1 | grep -avE "[0-9]+%\|"
  rc=$?; [ $rc -ne 0 ] && _RC=$rc
  n=$(ls "$D"/*.png 2>/dev/null | wc -l)
  echo "[gen] --- $TAG rc=$rc images=$n ---"
  [ "$n" -lt "$NP" ] && { echo "[gen] ❌ $TAG 只出了 $n 张,判为失败"; _RC=1; }
done
echo "[gen] exit=$_RC $(date)"
exit $_RC
