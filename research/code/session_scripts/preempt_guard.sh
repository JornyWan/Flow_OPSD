#!/bin/bash
# §4.896:单 reward split 的 preempt 副本守卫。被抢占(PREEMPTED/NODE_FAIL)且没到 ck100 ⇒ 不续训(续训 = EMA 重启,污染 MSP 系),
# 把半截 run 与其评测挪开、重投一个从零开始的 preempt 副本(最多 2 次)。其它失败只报告不重投。
# 用法: preempt_guard.sh <jobid> <reward> <tag>
R=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
J=$1; RW=$2; TAG=$3; OUT=$R/outputs/ab_${RW}_multikF4sa_s20260830_u100; N=0
while true; do
  while [ -n "$(squeue -j $J -h 2>/dev/null)" ]; do sleep 120; done
  sleep 30; ls $OUT $OUT/checkpoints >/dev/null 2>&1
  ST=$(sacct -j $J -X -n -o State | head -1 | tr -d ' ')
  if [ -f $OUT/checkpoints/checkpoint-100/lora/adapter_config.json ]; then echo "[pg:$TAG] job $J $ST,ck100 已在 ⇒ 完成 $(date)"; exit 0; fi
  case "$ST" in PREEMPTED*|NODE_FAIL*) ;; *) echo "[pg:$TAG] job $J 结束于 $ST 且无 ck100 —— 不自动重投,需人工看 $(date)"; exit 1;; esac
  N=$((N+1)); [ $N -gt 2 ] && { echo "[pg:$TAG] 已重投 2 次仍被抢占,停止 $(date)"; exit 1; }
  echo "[pg:$TAG] job $J $ST(无 ck100)⇒ 丢弃半截 run 并从零重投(第 $N 次)$(date)"
  for e in $(squeue -u chw4026 -h -o "%i %j" | awk -v t="mev_pi_${TAG}_" 'index($2,t)==1{print $1}'); do scancel $e; echo "  scancel 评测 $e"; done
  bash $R/codex_research/list_watchers.sh kill $TAG | tail -1
  D=$R/outputs/_discarded_${TAG}_$J; mkdir -p $D
  mv $R/outputs/matched_pi_${TAG}_* $D/ 2>/dev/null; mv $OUT $D/ 2>/dev/null; echo "  已挪到 $D"
  J=$(ARM=multikF4sa MK_SPLIT=0.5 SEED=20260830 UPDATES=100 OPT_DIAG=1 EXPECT_NPROC=4 AB_CONFIG=config/public.py:sd35_$RW OUTPUT_DIR=$OUT \
      sbatch --parsable --exclusive --account=sablab --partition=preempt_gpu --qos=low --gres=gpu:h100:4 --nodelist=ai-gpu16 --cpus-per-task=32 --mem=300G \
      --time=12:00:00 --no-requeue --job-name=pe_${RW}_sa --output=$R/outputs/slurm_%x_%j.out --export=ALL $R/slurm/ab_open3_v6.slurm)
  echo "  重投 job=$J"
  EVAL_PART=preempt_gpu EVAL_QOS=low WAIT_MIN=4320 nohup bash $R/codex_research/watch_matched_ckpt.sh $TAG $OUT 50 100 > $R/outputs/watch_${TAG}.log 2>&1 &
done
