#!/bin/bash
# 把两个新 arm 依次插到 poe15 之后、horizon2 之前:
#   poe15 -> ARM1 -> ARM2 -> horizon2 -> ste -> retain -> poe125
# 用法: insert_two_arms.sh <CFG1> <NAME1> <RUNDIR1> <TAG1> <CFG2> <NAME2> <RUNDIR2> <TAG2>
set -uo pipefail
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
POE15=1032590; HORIZON2=1033234
cd "$O"
sub() {  # $1=cfg $2=name $3=dep
  RUN_CFG=$1 ./sbatch_retry.sh --tries 20 --job-name="$2" --gres=gpu:h200:4 \
    --dependency=afterany:$3 slurm/opd_arm.slurm 2>&1 | tail -1
}
J1=$(sub "$1" "$2" $POE15); case "$J1" in ''|*[!0-9]*) echo "ARM1 提交失败: $J1"; exit 1;; esac
echo "[insert] ARM1 $2 = $J1 (afterany:$POE15)"
J2=$(sub "$5" "$6" $J1);    case "$J2" in ''|*[!0-9]*) echo "ARM2 提交失败: $J2"; exit 1;; esac
echo "[insert] ARM2 $6 = $J2 (afterany:$J1)"
scontrol update JobId=$HORIZON2 Dependency=afterany:$J2 || { echo "改 horizon2 依赖失败"; exit 1; }
echo "[insert] 用 scontrol 逐个核对(不看 squeue Reason):"
for j in $J1 $J2 $HORIZON2 1033235 1033236 1032591; do
  scontrol show job $j 2>/dev/null | tr ' ' '\n' | grep -E "^JobId=|^JobName=|^Dependency=" | tr '\n' ' '; echo
done
for pair in "$J1|$3|$4" "$J2|$7|$8"; do
  IFS='|' read -r jid rd tg <<< "$pair"
  nohup "$O/auto_eval_on_finish.sh" "$jid" "$rd" "$tg" 60 61 150 151 240 241 330 331 420 421 510 511 600 601 690 691 750 751 780 781 \
    > "$O/outputs/autoeval_${tg}.log" 2>&1 &
  echo "[insert] auto_eval 监视已挂: $tg (job $jid, pid $!)"
done
