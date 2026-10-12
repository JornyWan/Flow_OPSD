#!/bin/bash
set -u
SELF=$$
while read -r pid args; do
  [ "$pid" = "$SELF" ] && continue
  case "$args" in
    /bin/bash*watch_matched_ckpt.sh\ lr16x*) kill "$pid" 2>/dev/null && echo "  已杀 lr16x watcher (pid=$pid) —— 该臂已因发散终止,ck100 永不会出现";;
  esac
done < <(ps -u chw4026 -o pid=,args=)
echo "  剩余 watcher:"
ps -u chw4026 -o args= | grep '[w]atch_matched_ckpt.sh' | awk '{printf "    %-10s ck=%s\n",$3,$5}'
