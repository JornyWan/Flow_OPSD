#!/bin/bash
# 安全协议:绝不 pkill -f;取 PID 后核 /proc/<pid>/cmdline 前缀,再比对 $$,最后 kill <PID>
set -u
ME=$$
for pid in $(pgrep -x bash 2>/dev/null); do
  [ "$pid" = "$ME" ] && continue
  [ -r "/proc/$pid/cmdline" ] || continue
  cl=$(tr '\0' ' ' < "/proc/$pid/cmdline")
  case "$cl" in
    *"codex_research/watch_matched_ckpt.sh rbltpcfm "*) ;;
    *) continue ;;
  esac
  echo "命中 pid=$pid : $cl"
  kill "$pid" && echo "  已 kill $pid"
done
echo "剩余 rbltpcfm watcher: $(pgrep -x bash | while read p; do tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -c 'watch_matched_ckpt.sh rbltpcfm '; done | paste -sd+ | bc)"
