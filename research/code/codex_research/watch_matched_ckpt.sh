#!/bin/bash
# 等某个 checkpoint 出现,然后提交**逐图**评测(opsd_eval_matched.slurm)。
# 背景:watch_ab_ckpt_hipri.sh 投的是 opsd_eval.slurm,只出聚合均值;
# 而所有判定都用逐图配对口径(MDE=0.0033),故需要这个并行的 watcher。
# 用法: watch_matched_ckpt.sh <ARM 标签> <臂输出目录> <ck 编号> [更多 ck ...]
# EVAL_PART/EVAL_QOS 可覆盖评测分区(默认 sablab-gpu,保持原行为)。
# EVAL_CELLS 可覆盖评测格(默认只 drawbench)。unseen1k 图数相同但 1000 个唯一 prompt,
# 统计功效约 2.2x(drawbench 只有 200 唯一 prompt × 5 重复,5 张相关)⇒ 值得一起跑。
# ⚠️ A40 评测每个占 1 张卡;投 sablab 会吃掉"同时 6 卡"预算 ⇒ 常规用 EVAL_PART=preempt_gpu EVAL_QOS=low。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
TAG=$1; DIR=$2; shift 2
for STEP in "$@"; do
  CK="$DIR/checkpoints/checkpoint-$STEP"
  # 每个 ck 最多等 WAIT_MIN 分钟(默认 1440=24h);排队很久的臂要调大,否则静默超时
  for _ in $(seq 1 "${WAIT_MIN:-1440}"); do
    # ⚠️ 必须刷到 checkpoints/ 这一层:被检查的是 checkpoints/checkpoint-N/lora/...,
    # 只 ls 臂目录不会让该层的负缓存失效(§4.722:pickscore watcher 因此卡了 27 分钟)。
    ls "$DIR" "$DIR/checkpoints" >/dev/null 2>&1   # 刷 NFS 缓存
    [ -f "$CK/lora/adapter_config.json" ] && break
    sleep 60
  done
  [ -f "$CK/lora/adapter_config.json" ] || { echo "[mw:$TAG] ck$STEP 超时未出现"; continue; }
  sleep 60                                          # 宽限:等 ema/optimizer 写完
  # §4.779:评测前查发散守卫。只**告警**不拦 —— 发散臂的评测正是证明发散的证据,
  # 而 A40 不紧张;但必须喊出来,否则会像 aesthetic MSP 那样跑满 100 步才被发现。
  _AD=$(basename "$DIR")
  if python3 "$O/codex_research/check_diverged.py" 2>/dev/null \
       | grep -qE "^A[B ] *[0-9]+.*${_AD}( |$)"; then
    echo "[mw:$TAG] ⚠️⚠️ 发散守卫触发(臂 $_AD)—— 先看逐点序列再解读任何判定"
  fi
  OUT="$O/outputs/matched_pi_${TAG}_${STEP}_default_drawbench_flow40.json"
  if [ -f "$OUT" ]; then echo "[mw:$TAG] ck$STEP 已有逐图结果,跳过"; continue; fi
  J=$(ARM="pi_${TAG}_${STEP}" ADAPTER=default CKDIR="$CK" CELLS="${EVAL_CELLS:-drawbench:flow:40 drawbench:flow:4 unseen1k:flow:40 unseen1k:flow:4}" \
      REWARDS=hpsv2,clipscore,aesthetic,pickscore,imagereward \
      sbatch --parsable --job-name="mev_pi_${TAG}_${STEP}" \
      --account=sablab --partition="${EVAL_PART:-sablab-gpu}" ${EVAL_QOS:+--qos=$EVAL_QOS} \
      --gres=gpu:a40:1 --exclude=ai-gpu01,ai-gpu03 "$O/slurm/opsd_eval_matched.slurm")
  echo "[mw:$TAG] ck$STEP 就绪 -> 逐图评测 job=$J $(date)"
done
