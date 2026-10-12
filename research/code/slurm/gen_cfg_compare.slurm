#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
#
# 直观对照:官方协议(512 + CFG off)看着"不正常",是因为它本来就把 SD3.5-M 跑在
# 一个被削弱的工作点上。这里生成三组同 prompt 同 seed 的图说明:
#   base@gs1.0(官方协议) / base@gs4.5(常规用法) / 训练后@gs1.0
# 结论应是:训练把 CFG 缺席补上了还超出(六项指标已量化,见 EXPERIMENT_LOG §4.157)。
set -uo pipefail
OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
R=$OPD_ROOT/CA-OPD/reference_papers/DiffusionOPSD-main
SHIM=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad/opsd_shim
PY=$OPD_ROOT/../conda_envs/envs/diffusionopd/bin/python
[ -f "$HOME/.secrets/hf_jorny.env" ] && . "$HOME/.secrets/hf_jorny.env" 2>/dev/null
cd "$R" || exit 1
export PYTHONPATH="$SHIM:$R"
export PUBLIC_POLICY_WORLD_SIZE=4 PUBLIC_N_GPUS=4 PUBLIC_LAUNCH_WORLD_SIZE=4
export ZIMAGE_HEAVY_BRIDGE=0 ZIMAGE_HEAVY_DIFF_BRIDGE=0 WANDB_MODE=offline
export REWARD_CKPT_PATH=$OPD_ROOT/CA-OPD/code/DiffusionOPD-main/reward_ckpts
BL=$OPD_ROOT/outputs/ab_open3_baseline_np3_s20260830_u100/checkpoints/checkpoint-100/lora
# tag : lora(空=底模) : guidance_scale
SPECS=("base_cfg1::1.0" "base_cfg45::4.5" "trained_cfg1:$BL:1.0" "trained_cfg45:$BL:4.5")
_RC=0
for spec in "${SPECS[@]}"; do
  TAG=${spec%%:*}; rest=${spec#*:}; CK=${rest%%:*}; GS=${rest##*:}
  D=$OPD_ROOT/outputs/samples_cfg/$TAG
  [ "$(ls "$D"/*.png 2>/dev/null | wc -l)" -ge 12 ] && { echo "[cfg] $TAG 已有,跳过"; continue; }
  echo "[cfg] ===== $TAG  gs=$GS  $(date) ====="
  "$PY" scripts/cross_eval.py ${CK:+--ckpt "$CK"} \
    --config sd35_open3 --config_file config/public.py \
    --prompts "$R/data/pickapic/test.txt" --prompt_set_name drawbench \
    --protocol_name "cfgcmp_gs${GS}" --n_prompts 12 \
    --num_steps 40 --sampler flow --guidance_scale "$GS" \
    --seed 42 --pickscore_scale raw --images_dir "$D" \
    --out "$OPD_ROOT/outputs/samples_cfg/scores_${TAG}.json" 2>&1 | grep -avE "[0-9]+%\|"
  rc=$?; n=$(ls "$D"/*.png 2>/dev/null | wc -l)
  echo "[cfg] --- $TAG rc=$rc images=$n ---"
  { [ $rc -ne 0 ] || [ "$n" -lt 12 ]; } && _RC=1
done
echo "[cfg] exit=$_RC $(date)"
exit $_RC
