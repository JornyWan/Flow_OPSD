#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
#
# 训练后的模型再开 CFG 会怎样?12 张上的初步结果是四轴全跌(目标合计 −0.056),
# 但 n=12 只有约 1.8σ。这里跑全量 1000 prompt 并存逐图分数,与已有的
# opsd_eval_ab_basenp3_ck100(同 checkpoint、gs=1.0)做逐图配对。
# 论文只报了**未训练**模型的 ±CFG 两行,没有训练后的。
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
_RC=0
# gs=1.0 也重跑一遍:已有的那份没存逐图分数,配对检验需要
for spec in "trained_gs1:$BL:1.0" "trained_gs45:$BL:4.5" "base_gs45::4.5"; do
  TAG=${spec%%:*}; rest=${spec#*:}; CK=${rest%%:*}; GS=${rest##*:}
  OUT="$OPD_ROOT/outputs/cfgfull_${TAG}.json"
  [ -s "$OUT" ] && { echo "[cf] $TAG 已存在,跳过"; continue; }
  echo "[cf] ===== $TAG gs=$GS $(date) ====="
  "$PY" scripts/cross_eval.py ${CK:+--ckpt "$CK"} \
    --config sd35_open3 --config_file config/public.py \
    --prompts "$R/data/pickapic/test.txt" --prompt_set_name drawbench \
    --protocol_name "cfgfull_gs${GS}" --num_steps 40 --sampler flow \
    --guidance_scale "$GS" --seed 42 --pickscore_scale raw --dump_per_image \
    --rewards hpsv2,clipscore,aesthetic,pickscore,imagereward \
    --out "$OUT" 2>&1 | grep -avE "[0-9]+%\|"
  rc=$?
  n=$("$PY" -c "
import json
try:
    pi=json.load(open('$OUT')).get('per_image') or []
    print(sum(1 for r in pi if all(isinstance(r.get(k),(int,float)) for k in ('pickscore','clipscore','hpsv2'))))
except Exception: print(0)")
  echo "[cf] --- $TAG rc=$rc per_image=$n ---"
  { [ $rc -ne 0 ] || [ "$n" -lt 900 ]; } && _RC=1
done
echo "[cf] exit=$_RC $(date)"
exit $_RC
