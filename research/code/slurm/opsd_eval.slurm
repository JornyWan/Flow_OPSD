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
CKROOT=${CKROOT:-$OPD_ROOT/outputs/opsd_repro_sd35_pickscore/checkpoints}
STEPS=${STEPS:-100}
ARM=${ARM:-}          # 臂标识,进入输出文件名;留空 = pickscore 臂的历史命名
# cross_eval 默认 scorer 集 DEFAULT_Y_REWARDS=(hpsv2,clipscore,aesthetic,pickscore),
# **不含 imagereward**(源码注释:"ImageReward ... can be requested explicitly")。
# 留空 = 保持官方默认;显式给值可补齐论文第 5 列。同一次生成即可多算 scorer,边际成本≈0。
REWARDS=${REWARDS:-}
# cross_eval 用 `--config` 的 reward_fn 推出 `train_reward` 元数据,而 compare_opsd_repro.py
# 正是靠它判"哪一列可比"。写死 sd35_pickscore 会让 clipscore/hpsv2 臂被标成 pickscore ->
# 对表时标错可比列。已核对四个 config 的**评测相关**字段(resolution/gs/noise/eval_steps/
# model/dataset)完全相同,故换 config 只改标签,评测逐位不变。
EVALCFG=${EVALCFG:-sd35_pickscore}
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
# ⚠️ 显式给 REWARD_CKPT_PATH(保险,非必需):
# 官方 `src/reward_ckpt_path.py` 的回退路径是 `dirname(__file__)/../reward_ckpts`。
# 【2026-08-30 更正】原注释说"相对路径解析到垫片上级而非仓库" —— **不成立**。
# os.path.join 不做词法归一,交给 OS 的字面路径是 `<shim>/diffusionopsd/../reward_ckpts`,
# 而 OS 是**先跟随符号链接再解析 `..`**,故它落在 DiffusionOPSD-main/ 内,不是垫片上级。
# 1034825 的真因只是:**官方仓库根本没有 reward_ckpts/ 目录**。
# 现已软链 DiffusionOPSD-main/reward_ckpts -> DiffusionOPD-main/reward_ckpts,回退路径本身已可用。
export REWARD_CKPT_PATH=$OPD_ROOT/CA-OPD/code/DiffusionOPD-main/reward_ckpts
# ── 多分区投递的"先跑到的赢"互斥 ────────────────────────────────────────
# 同一次评测会同时投到 sablab-gpu(A40, 优先级高)与 preempt_gpu(A40, qos=low),
# 必须保证只有一个副本真跑,否则两份会写同一个 OUT json、互相覆盖。
# 与 ab_open3.slurm 用同一套 mkdir 原子锁模式。
_LK="/midtier/minilab/scratch/chw4026/tmp/evlock_${ARM:-none}_$(echo $STEPS | tr ' ' '-')"
if mkdir "$_LK" 2>/dev/null; then
  echo "$SLURM_JOB_ID" > "$_LK/jobid"; echo "[lock] 本副本($SLURM_JOB_ID)取得评测锁 $_LK"
  trap 'rmdir "$_LK" 2>/dev/null' EXIT
else
  _OWN=$(cat "$_LK/jobid" 2>/dev/null)
  if squeue -j "$_OWN" -h -o %T 2>/dev/null | grep -q RUNNING; then
    echo "[lock] 评测已由 job $_OWN 在跑,本副本($SLURM_JOB_ID)退出"; exit 0
  fi
  echo "[lock] 锁存在但持有者 $_OWN 已不在运行,接管"
  echo "$SLURM_JOB_ID" > "$_LK/jobid"; trap 'rmdir "$_LK" 2>/dev/null' EXIT
fi

ls "$CKROOT" >/dev/null 2>&1   # 刷 NFS 属性缓存
echo "[eval] ckroot=$CKROOT steps=$STEPS arm=${ARM:-<空>} evalcfg=$EVALCFG rewards=${REWARDS:-<官方默认>} host=$(hostname) $(date)"
# evalcfg 决定 JSON 里的 train_reward 元数据,而 compare_opsd_repro.py 靠它判"哪一列可比"。
# 启动即打印,避免要等 40 分钟看产物才能确认口径(见 0041 §⑦)。
nvidia-smi -L
_RC=0
for s in $STEPS; do
  # cross_eval.py:358 要求 --ckpt 目录下直接有 adapter_config.json。
  # 官方 checkpoint-N/ 里是 ema.pt / optimizer.pt / lora/ ,adapter 在 **lora/ 子目录**。
  # 实测: checkpoint-40/adapter_config.json 不存在, checkpoint-40/lora/adapter_config.json 存在。
  # STEPS 里给字面量 base = 评测未加 LoRA 的底模(cross_eval.py 的 --ckpt "" 语义:
  # "Empty evaluates the base checkpoint directly")。Open3 对比里每条臂都从 base 出发,
  # 这一行是该表的锚点。约定与 eval_opsdck.slurm 的 `[ "$CKPT" != "base" ]` 一致。
  if [ "$s" = "base" ]; then
    CK=""
    echo "[eval] ===== base(无 LoRA)  $(date) ====="
  else
  CK="$CKROOT/checkpoint-$s/lora"
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
  fi
  # pickscore_scale: 论文主表(24.94)用 raw;训练日志用 /26 归一。两种都算,避免尺度误读。
  # 只跑 raw:两种口径仅差 pickscore 的 /26 归一化(training = raw / 26),
  # 而每种口径都要重新生成 1000 张图(实测 2322s)。生成一次、事后换算,省掉一次 39 分钟的重复生成。
  for SC in raw; do
    # ⚠️ 不同臂的 checkpoint 编号会重合(各臂都有 checkpoint-100)。不带臂标识会让
    # 后来的臂命中 `[ -s "$OUT" ]` 而被**静默跳过**,作业照样 exit=0 却没产出新数据。
    # ARM 为空时沿用旧名(pickscore 臂的既有产物与 compare_opsd_repro.py 的默认 glob 不受影响)。
    OUT="$OPD_ROOT/outputs/opsd_eval_${ARM:+${ARM}_}ck${s}_${SC}.json"
    [ -s "$OUT" ] && { echo "[eval] $OUT 已存在,跳过"; continue; }
    echo "[eval] ===== checkpoint-$s  pickscore_scale=$SC  $(date) ====="
    "$PY" scripts/cross_eval.py --ckpt "$CK" \
      --config "$EVALCFG" --config_file config/public.py \
      --prompts "$R/data/pickapic/test.txt" --prompt_set_name drawbench_test \
      --protocol_name repro_drawbench_flow40 --num_steps 40 --sampler flow \
      --seed 42 --pickscore_scale "$SC" --out "$OUT" ${REWARDS:+--rewards "$REWARDS"} 2>&1 | grep -avE "[0-9]+%\|"
    rc=$?; [ $rc -ne 0 ] && _RC=$rc
    echo "[eval] --- checkpoint-$s/$SC rc=$rc ---"; cat "$OUT" 2>/dev/null | head -c 600; echo
  done
done
echo "[eval] exit=$_RC $(date)"
exit $_RC
