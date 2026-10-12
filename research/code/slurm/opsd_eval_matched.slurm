#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
#
# 0069 §9 的 protocol-transfer 2x2。
# 现有正式结果(DrawBench + flow-40 + default adapter)相对训练协议**同时**改了四件事:
#   prompt 域、sampler/步数、每 prompt 图数、以及 behavior(old) vs checkpoint(default) 权重。
# Q 方法的梯度里含**特定 suffix solver 的 Jacobian**,所以 sampler 迁移必须单独测,
# 不能把整个 gap 都算到 prompt 上。本脚本把这些因子拆开逐个跑。
#
# ⚠️ 硬件固定 A40:已有的 DrawBench/flow-40 结果都在 A40 上做,换卡会破坏与 cell D 的可比性。
set -uo pipefail
OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
R=$OPD_ROOT/CA-OPD/reference_papers/DiffusionOPSD-main
SHIM=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad/opsd_shim
PY=$OPD_ROOT/../conda_envs/envs/diffusionopd/bin/python
ARM=${ARM:?必须给 ARM(如 basenp3/qopsd/qbt)}
ADAPTER=${ADAPTER:-old}       # old = 训练时的 behavior adapter;default = checkpoint 主 adapter;none = 未训练底模
# ADAPTER=none(评未训练底模)不需要 checkpoint,故守卫要按 ADAPTER 分情况。
if [ "$ADAPTER" != "none" ]; then
  CKDIR=${CKDIR:?必须给 checkpoint 目录(到 checkpoints/checkpoint-N);ADAPTER=none 时可省}
else
  CKDIR=${CKDIR:-}
fi
CELLS=${CELLS:?形如 "unseen1k:dpm2:10 drawbench:dpm2:10 unseen1k:flow:40"}
[ -f "$HOME/.secrets/hf_jorny.env" ] && . "$HOME/.secrets/hf_jorny.env" 2>/dev/null
cd "$R" || exit 1
export PYTHONPATH="$SHIM:$R"
# 见 opsd_eval.slurm:public.py 的 48x24 分组检查无条件执行,这里填 4 仅为过检查,不影响评测。
export PUBLIC_POLICY_WORLD_SIZE=4 PUBLIC_N_GPUS=4 PUBLIC_LAUNCH_WORLD_SIZE=4
export ZIMAGE_HEAVY_BRIDGE=0 ZIMAGE_HEAVY_DIFF_BRIDGE=0 WANDB_MODE=offline
export REWARD_CKPT_PATH=$OPD_ROOT/CA-OPD/code/DiffusionOPD-main/reward_ckpts
REWARDS=${REWARDS:-hpsv2,clipscore,aesthetic,pickscore,imagereward}

case "$ADAPTER" in
  old)     CK="$CKDIR/lora/old" ;;
  default) CK="$CKDIR/lora" ;;
  # none = 未训练底模(cross_eval 的 --ckpt 留空即评底模)。加速叙事的基础参照:
  # reward 训练本身带来了多少步数摊销,qcore 的压平是在此之上的增量还是全部。
  none)    CK="" ;;
  *) echo "[m] 未知 ADAPTER=$ADAPTER"; exit 1 ;;
esac
if [ -n "$CK" ] && [ ! -f "$CK/adapter_config.json" ]; then
  echo "[m] ❌ $CK 无 adapter_config.json"; exit 1
fi

_LK="/midtier/minilab/scratch/chw4026/tmp/mevlock_${ARM}_${ADAPTER}_$(echo $CELLS | tr ' :' '--')"
mkdir "$_LK" 2>/dev/null || { echo "[m] 同参数评测已在跑,本副本退出"; exit 0; }
trap 'rmdir "$_LK" 2>/dev/null' EXIT

echo "[m] arm=$ARM adapter=$ADAPTER ck=$CK cells='$CELLS' host=$(hostname) $(date)"
_RC=0
for cell in $CELLS; do
  PT=${cell%%:*}; rest=${cell#*:}; SAMP=${rest%%:*}; NST=${rest##*:}
  case "$PT" in
    unseen1k)  PF="$OPD_ROOT/CA-OPD/code/DiffusionOPSD-rpi/data/pickapic/unseen_train1k.txt" ;;
    drawbench) PF="$R/data/pickapic/test.txt" ;;
    *) echo "[m] 未知 prompt 集 $PT"; _RC=1; continue ;;
  esac
  [ -f "$PF" ] || { echo "[m] ❌ prompt 文件不存在 $PF"; _RC=1; continue; }
  OUT="$OPD_ROOT/outputs/matched_${ARM}_${ADAPTER}_${PT}_${SAMP}${NST}.json"
  [ -s "$OUT" ] && { echo "[m] $OUT 已存在,跳过"; continue; }
  echo "[m] ===== $ARM/$ADAPTER  $PT + $SAMP-$NST  $(date) ====="
  "$PY" scripts/cross_eval.py ${CK:+--ckpt "$CK"} \
    --config sd35_open3 --config_file config/public.py \
    --prompts "$PF" --prompt_set_name "$PT" \
    --protocol_name "matched_${PT}_${SAMP}${NST}_${ADAPTER}" \
    --num_steps "$NST" --sampler "$SAMP" \
    --seed 42 --pickscore_scale raw --dump_per_image \
    --out "$OUT" --rewards "$REWARDS" 2>&1 | grep -avE "[0-9]+%\|"
  rc=$?; [ $rc -ne 0 ] && _RC=$rc
  # 不能只看 rc:核实 per_image 真的有 1000 条
  n=$("$PY" -c "
import json,sys
try:
    d=json.load(open('$OUT'))
    # per_image 是 **list of dicts**(每条 {index, prompt, <reward>: float}),
    # 不是 dict of lists。写错过一次:守卫恒返回 0,把 4 份好数据全判成失败(1039225-27)。
    pi=d.get('per_image') or []
    print(sum(1 for r in pi if all(isinstance(r.get(k),(int,float))
                                   for k in ('pickscore','clipscore','hpsv2'))))
except Exception: print(0)")
  echo "[m] --- $PT/$SAMP$NST rc=$rc per_image=$n ---"
  [ "$n" -lt 900 ] && { echo "[m] ❌ per_image 只有 $n 条,判为失败"; _RC=1; }
done
echo "[m] exit=$_RC $(date)"
exit $_RC
