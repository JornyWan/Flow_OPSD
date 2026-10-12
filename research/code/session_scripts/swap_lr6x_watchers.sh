#!/bin/bash
# lr6x 的 3 个 watcher 只剩 6.7h,而 sablab-gpu 的 A100 被占到 57h+ ⇒ 必然先超时消失。
# 换成 120h 版本。写成脚本执行,避免 ps 模式命中调用者自身。
set -u
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
S=$(dirname "$0"); SELF=$$
D="$B/outputs/ab_open3_lr6x_s20260830_u100"
echo "== 停旧的 lr6x watcher =="
k=0
while read -r pid args; do
  [ "$pid" = "$SELF" ] || [ "$pid" = "$PPID" ] && continue
  case "$args" in *watch_srun_eval_preempt.sh\ lr6x\ *) kill "$pid" 2>/dev/null && k=$((k+1));; esac
done < <(ps -u chw4026 -o pid=,args=)
echo "  已停 $k 个"
echo "== 挂 120h 版 =="
for ck in 20 50 100; do
  nohup bash "$S/watch_srun_eval_long.sh" lr6x "$D" "$ck" > "$S/se_lr6x_ck${ck}_long.log" 2>&1 &
  echo "  ck$ck pid=$!"
done
