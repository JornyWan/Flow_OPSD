#!/bin/bash
# 用 ps 快照 + 精确匹配,避免 pgrep -f 匹配到本脚本自身(历史上因此杀过自己的 shell 3 次)
set -u; SELF=$$; k=0
while read -r pid args; do
  [ "$pid" = "$SELF" ] && continue
  case "$args" in
    *watch_matched_ckpt.sh\ lr12x\ *) kill "$pid" 2>/dev/null && { echo "  killed pid=$pid  $args"; k=$((k+1)); } ;;
  esac
done < <(ps -u chw4026 -o pid=,args=)
echo "  共杀 $k 个"
