#!/bin/bash
# 把 11 个 watcher 迁到修好 NFS 刷新的 long2 版。逐个记录原参数后重起。
set -u
S=$(dirname "$0"); SELF=$$
mapfile -t LINES < <(ps -u chw4026 -o pid=,args= | grep 'watch_srun_eval_long\.sh' | grep -v grep)
echo "== 现存 long 版 watcher: ${#LINES[@]} 个 =="
for l in "${LINES[@]}"; do
  pid=$(echo "$l" | awk '{print $1}')
  tag=$(echo "$l" | awk '{print $4}'); dir=$(echo "$l" | awk '{print $5}'); ck=$(echo "$l" | awk '{print $6}')
  [ "$pid" = "$SELF" ] && continue
  [ -z "$tag" ] || [ -z "$dir" ] || [ -z "$ck" ] && { echo "  ⚠️ 解析失败,跳过: $l"; continue; }
  kill "$pid" 2>/dev/null
  nohup bash "$S/watch_srun_eval_long2.sh" "$tag" "$dir" "$ck" > "$S/se2_${tag}_ck${ck}.log" 2>&1 &
  echo "  $tag ck$ck: $pid -> $!"
done
