#!/bin/bash
# 把一个新 arm 插到 poe15 之后、horizon2 之前,并挂 auto_eval 监视。
# 用法: insert_arm.sh <RUN_CFG> <JOBNAME> <RUNDIR> <TAG>
set -uo pipefail
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
CFG=$1; NAME=$2; RUNDIR=$3; TAG=$4
POE15=1032590; HORIZON2=1033234
cd "$O"
JID=$(RUN_CFG=$CFG ./sbatch_retry.sh --tries 20 --job-name="$NAME" --gres=gpu:h200:4 \
        --dependency=afterany:$POE15 slurm/opd_arm.slurm 2>&1 | tail -1)
case "$JID" in ''|*[!0-9]*) echo "提交失败: $JID"; exit 1;; esac
echo "[insert] 新 arm $NAME = $JID (依赖 afterany:$POE15)"
scontrol update JobId=$HORIZON2 Dependency=afterany:$JID || { echo "改依赖失败"; exit 1; }
# 用 scontrol 逐个核对(不看 squeue Reason,有缓存延迟)
echo "[insert] 核对:"
for j in $JID $HORIZON2 1033235 1033236 1032591; do
  scontrol show job $j 2>/dev/null | tr ' ' '\n' | grep -E "^JobId=|^JobName=|^Dependency=" | tr '\n' ' '; echo
done
nohup "$O/auto_eval_on_finish.sh" "$JID" "$RUNDIR" "$TAG" 61 151 241 331 421 511 601 691 781 800 \
  > "$O/outputs/autoeval_${TAG}.log" 2>&1 &
echo "[insert] auto_eval 监视已挂 (pid $!)"
