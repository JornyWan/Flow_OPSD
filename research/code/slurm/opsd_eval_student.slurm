#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
#
# Phase 1 收尾: 用官方 cross_eval.py 在 held-out DrawBench(999 条)上评测复现出的 checkpoint,
# 与论文主表逐列对照。训练本身 eval_freq=0,不产出 held-out 分数,故必须单独跑这一步。
set -uo pipefail
OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
R=$OPD_ROOT/CA-OPD/reference_papers/DiffusionOPSD-main
SHIM=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad/opsd_shim
PY=$OPD_ROOT/../conda_envs/envs/diffusionopd/bin/python
CKROOT=${CKROOT:-/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/models/DiffusionOPD}
STEPS=${STEPS:-100}
# cross_eval 默认 scorer 集不含 imagereward(源码 DEFAULT_Y_REWARDS)。留空 = 官方默认。
# TAG 进入输出文件名,避免换 scorer 集后撞上旧结果被"已存在→跳过"而静默无产出。
REWARDS=${REWARDS:-}
TAG=${TAG:-}
[ -f "$HOME/.secrets/hf_jorny.env" ] && . "$HOME/.secrets/hf_jorny.env" 2>/dev/null
cd "$R" || exit 1
export PYTHONPATH="$SHIM:$R"
# ⚠️ 必须设成 3/4/6/8 之一,不能用实际卡数 1:
# `config/public.py:get_config` **无条件**执行 48x24 分组检查,1 卡直接抛
# `ValueError: Cannot satisfy 48x24 grouping with 1 policy GPUs`(实测 1034797 因此 FAILED)。
# 该分组只决定**训练**的 batch 布局;已核对 cross_eval.py 只读
# dataset / pretrained.model / resolution / reward_fn / sample.* / train.lora_path,
# **完全不用** train_batch_size / num_batches_per_epoch / gradient_accumulation_steps,
# 故此处填 4 仅为通过配置检查,**不影响评测结果**。生成 batch 由 --batch_size 控制。
export PUBLIC_POLICY_WORLD_SIZE=4 PUBLIC_N_GPUS=4 PUBLIC_LAUNCH_WORLD_SIZE=4
export ZIMAGE_HEAVY_BRIDGE=0 ZIMAGE_HEAVY_DIFF_BRIDGE=0 WANDB_MODE=offline
# ⚠️ 必须显式给 REWARD_CKPT_PATH:
# 官方 `src/reward_ckpt_path.py` 的回退路径是 `dirname(__file__)/../reward_ckpts`,
# 而我们用**符号链接垫片**让 diffusionopsd 可 import,导致 __file__ 落在 scratchpad 下,
# 相对路径解析到垫片上级而非仓库 -> 实测 hpsv2 报
# "open_clip_pytorch_model.bin is not a known tag or valid file path"(1034825)。
# 官方仓库本身没有 reward_ckpts/,权重在我们自己的仓库里(open_clip 3.7G + HPS_v2.1 1.9G)。
export REWARD_CKPT_PATH=$OPD_ROOT/CA-OPD/code/DiffusionOPD-main/reward_ckpts
ls "$CKROOT" >/dev/null 2>&1   # 刷 NFS 属性缓存
echo "[eval] ckroot=$CKROOT steps=$STEPS host=$(hostname) $(date)"
nvidia-smi -L
_RC=0
for s in $STEPS; do
  # cross_eval.py:358 要求 --ckpt 目录下直接有 adapter_config.json。
  # 官方 checkpoint-N/ 里是 ema.pt / optimizer.pt / lora/ ,adapter 在 **lora/ 子目录**。
  # 实测: checkpoint-40/adapter_config.json 不存在, checkpoint-40/lora/adapter_config.json 存在。
  CK="$CKROOT/Student/lora"
  ls "$CK" >/dev/null 2>&1
  if [ ! -f "$CK/adapter_config.json" ]; then
    # 保险: 目标 step 不存在时回退到**最高可用** checkpoint,避免整轮评测空手而归。
    # (官方保存条件是 global_step % save_freq == 0,编号可能与 num_epochs 差一位。)
    _ALT=$(ls -d "$CKROOT"/checkpoint-* 2>/dev/null | sed 's|.*/checkpoint-||' | sort -n | tail -1)
    if [ -n "$_ALT" ] && [ -f "$CKROOT/checkpoint-$_ALT/lora/adapter_config.json" ]; then
      echo "[eval] ⚠️ checkpoint-$s 不存在,回退到最高可用 checkpoint-$_ALT"
      s=$_ALT; CK="$CKROOT/checkpoint-$_ALT/lora"
    else
      echo "[eval] ❌ $CK 无 adapter_config.json 且无可回退项,跳过"; _RC=1; continue
    fi
  fi
  # pickscore_scale: 论文主表(24.94)用 raw;训练日志用 /26 归一。两种都算,避免尺度误读。
  # 只跑 raw:两种口径仅差 pickscore 的 /26 归一化(training = raw / 26),
  # 而每种口径都要重新生成 1000 张图(实测 2322s)。生成一次、事后换算,省掉一次 39 分钟的重复生成。
  for SC in raw; do
    OUT="$OPD_ROOT/outputs/opsd_eval_STUDENT${TAG:+_${TAG}}_${SC}.json"
    [ -s "$OUT" ] && { echo "[eval] $OUT 已存在,跳过"; continue; }
    echo "[eval] ===== checkpoint-$s  pickscore_scale=$SC  $(date) ====="
    "$PY" scripts/cross_eval.py --ckpt "$CK" \
      --config sd35_pickscore --config_file config/public.py \
      --prompts "$R/data/pickapic/test.txt" --prompt_set_name drawbench_test \
      --protocol_name repro_drawbench_flow40 --num_steps 40 --sampler flow \
      --seed 42 --pickscore_scale "$SC" --out "$OUT" ${REWARDS:+--rewards "$REWARDS"} 2>&1 | grep -avE "[0-9]+%\|"
    rc=$?; [ $rc -ne 0 ] && _RC=$rc
    echo "[eval] --- checkpoint-$s/$SC rc=$rc ---"; cat "$OUT" 2>/dev/null | head -c 600; echo
  done
done
echo "[eval] exit=$_RC $(date)"
exit $_RC
