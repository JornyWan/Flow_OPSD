#!/bin/bash
set -u
SELF=$$
while read -r pid args; do
  [ "$pid" = "$SELF" ] && continue
  case "$args" in
    *watch_matched_ckpt.sh\ lr16x\ *) kill "$pid" 2>/dev/null && echo "  已杀 lr16x watcher (pid=$pid)";;
  esac
done < <(ps -u chw4026 -o pid=,args=)
sleep 1
echo "  剩余 watcher(应为 lr4xs2 / lrhalf / lr8xs2 / lr12x 四个):"
ps -u chw4026 -o args= | grep '[w]atch_matched_ckpt.sh' | awk '{printf "    %-10s dir=%s ck=%s\n",$3,$4,$5}'
