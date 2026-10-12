#!/bin/bash
# 把 sbatch 版 watcher 换成 srun 版(sbatch 落盘故障期间 sbatch 版失败且不重试)。
# 写成脚本:§4.319 教训 —— 内联 grep 会匹配到调用它的 shell 自身。
set -u
S=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SELF=$$
echo "== 杀 sbatch 版 watcher =="
k=0
while read -r pid args; do
  [ "$pid" = "$SELF" ] && continue
  case "$args" in
    *watch_matched_ckpt.sh*|*scratchpad/watch_a100.sh*)
      kill "$pid" 2>/dev/null && { k=$((k+1)); echo "  killed $pid $(echo "$args"|awk '{print $3}')"; } ;;
  esac
done < <(ps -u chw4026 -o pid=,args=)
echo "  共杀 $k 个"
sleep 2
echo "== 挂 srun 版(A40 + A100)=="
for a in qstrain9 qstrain7 rho2x qstrain5; do
  D="$O/outputs/ab_open3_${a}_s20260830_u100"
  nohup bash "$S/watch_srun_eval.sh"      "$a" "$D" 100 > "$S/se_${a}_a40.log"  2>&1 &
  echo "  $a A40  pid=$!"
  nohup bash "$S/watch_srun_eval_a100.sh" "$a" "$D" 100 > "$S/se_${a}_a100.log" 2>&1 &
  echo "  $a A100 pid=$!"
done
sleep 2
echo "== 最终计数 =="
n_old=0; n_new=0
while read -r pid args; do
  [ "$pid" = "$SELF" ] && continue
  case "$args" in
    *watch_matched_ckpt.sh*|*scratchpad/watch_a100.sh*) n_old=$((n_old+1)) ;;
    *watch_srun_eval*.sh*) n_new=$((n_new+1)) ;;
  esac
done < <(ps -u chw4026 -o pid=,args=)
echo "  sbatch 版残留=$n_old   srun 版=$n_new (期望 0 / 9)"
