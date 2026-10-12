#!/bin/bash
# 轮询等待各 checkpoint 出现 -> **一出现就派该点的评测**,不等整个训练结束。
# 动机: open3 要跑约 19h;若等结束再串行评 3 个点(每点 41 分钟),要再等 2h 才有数。
# 用法: watch_ckpt_eval.sh <jobid> <arm> <ckroot> "<steps>"
set -uo pipefail
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
JID=$1; ARM=$2; CKROOT=$3; STEPS=$4
echo "[ck:$ARM] 监视 $CKROOT 的 step:$STEPS (训练 job=$JID) $(date)"
for s in $STEPS; do
  CK="$CKROOT/checkpoint-$s/lora/adapter_config.json"
  while true; do
    ls "$CKROOT" >/dev/null 2>&1                       # 刷 NFS 属性缓存(必需)
    ls "$CKROOT/checkpoint-$s" >/dev/null 2>&1
    [ -f "$CK" ] && break
    # 训练作业已消失且该点仍未出现 -> 不再等
    if [ -z "$(squeue -j "$JID" -h 2>/dev/null)" ]; then
      ls "$CKROOT" >/dev/null 2>&1; sleep 30           # 收尾写盘的宽限
      [ -f "$CK" ] && break
      echo "[ck:$ARM] 训练已结束但 checkpoint-$s 未出现 -> 放弃该点 $(date)"; break 2
    fi
    sleep 120
  done
  [ -f "$CK" ] || continue
  echo "[ck:$ARM] checkpoint-$s 就绪 -> 派评测 $(date)"
# cross_eval 由 `--config` 的 reward_fn 推 `train_reward` 元数据,compare_opsd_repro.py 靠它
# 判"哪一列可比"。按臂选对应 config,否则 clipscore/hpsv2 臂会被标成 pickscore(见 0041)。
case "$ARM" in
  pickscore|clipscore|hpsv2|aesthetic|imagereward|open3) EVALCFG="sd35_$ARM" ;;
  *) EVALCFG="sd35_pickscore" ;;   # base/Student 等无训练 reward,标签无意义,不影响评分
esac
  J=$(EVALCFG="$EVALCFG" ARM="$ARM" CKROOT="$CKROOT" STEPS="$s" \
      REWARDS=hpsv2,clipscore,aesthetic,pickscore,imagereward \
      sbatch --parsable --job-name="ev_${ARM}_$s" --gres=gpu:a40:1 "$O/slurm/opsd_eval.slurm")
  echo "[ck:$ARM] step=$s eval job=$J"
done
echo "[ck:$ARM] 全部 step 处理完毕 $(date)"
