#!/bin/bash
# 写成脚本:pkill -f 会匹配到调用者自己的命令行(已栽四次)
set -u
SELF=$$; TARGET_CK=$1; TARGET_ARM=$2; k=0
while read -r pid args; do
  [ "$pid" = "$SELF" ] && continue
  case "$args" in
    *watch_srun_eval*) ;;
    *) continue ;;
  esac
  set -- $args
  eval "last=\${$#}"
  case "$args" in *"$TARGET_ARM"*) ;; *) continue ;; esac
  [ "$last" = "$TARGET_CK" ] || continue
  kill "$pid" 2>/dev/null && k=$((k+1))
done < <(ps -u chw4026 -o pid=,args=)
echo "  已停 $k 个 watcher"
