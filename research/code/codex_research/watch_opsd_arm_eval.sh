#!/bin/bash
# 等某个 OPSD 训练臂结束 -> 用官方 cross_eval.py 在 held-out DrawBench(999) 上排评测(A40)。
# 与 auto_eval_on_finish.sh 的区别: 那个走 DiffusionOPD 协议(gs=4.5,n=256,三数据集),
# 这个走 **OPSD 原生协议**(gs=1.0,flow 40 步,seed 42),才是与论文主表对标的口径。
# 用法: watch_opsd_arm_eval.sh <jobid> <arm> <ckroot> "<steps>"
set -uo pipefail
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
JID=$1; ARM=$2; CKROOT=$3; STEPS=$4
echo "[watch:$ARM] 等作业 $JID 结束 $(date)"
while [ -n "$(squeue -j "$JID" -h 2>/dev/null)" ]; do sleep 180; done
echo "[watch:$ARM] $JID 已结束 $(date)"
ST=$(sacct -j "$JID" -X -n --format=State 2>/dev/null | head -1 | tr -d ' ')
echo "[watch:$ARM] 终态=$ST"
ls "$CKROOT" >/dev/null 2>&1      # 刷 NFS 属性缓存(曾因缓存漏派 checkpoint-781)
HAVE=""
for s in $STEPS; do
  [ -f "$CKROOT/checkpoint-$s/lora/adapter_config.json" ] && HAVE="$HAVE $s"
done
if [ -z "$HAVE" ]; then
  echo "[watch:$ARM] ❌ $CKROOT 下没有任何目标 checkpoint(要求 checkpoint-N/lora/adapter_config.json)"
  ls "$CKROOT" 2>/dev/null | tr '\n' ' '; echo; exit 1
fi
echo "[watch:$ARM] 将评测 step:$HAVE"
# cross_eval 由 `--config` 的 reward_fn 推 `train_reward` 元数据,compare_opsd_repro.py 靠它
# 判"哪一列可比"。按臂选对应 config,否则 clipscore/hpsv2 臂会被标成 pickscore(见 0041)。
case "$ARM" in
  pickscore|clipscore|hpsv2|aesthetic|imagereward|open3) EVALCFG="sd35_$ARM" ;;
  *) EVALCFG="sd35_pickscore" ;;   # base/Student 等无训练 reward,标签无意义,不影响评分
esac
J=$(EVALCFG="$EVALCFG" ARM="$ARM" CKROOT="$CKROOT" STEPS="$HAVE" \
    REWARDS=hpsv2,clipscore,aesthetic,pickscore,imagereward \
    sbatch --parsable --job-name="ev_opsd_$ARM" --gres=gpu:a40:1 "$O/slurm/opsd_eval.slurm")
echo "[watch:$ARM] 已提交评测 job=$J $(date)"
