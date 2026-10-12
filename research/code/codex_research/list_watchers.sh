#!/bin/bash
# watcher 清单 / 定点杀。唯一正确实现 —— 不要再现写(已踩过三个坑):
#   坑1 两种调用形式:老 watcher 相对路径启动、新的绝对路径 ⇒ 锚在 **basename** 上,不锚绝对路径。
#   坑2 自匹配(已 7 次):创建脚本的 shell 的 cmdline 里含脚本全文 ⇒ 子串包含会命中父 shell。
#        必须要求 **第 2 个 token 以 watch_matched_ckpt.sh 结尾**(真的是那个位置结构)。
#   坑3 comm 被截断:`/bin/bash script.sh` 起的进程 comm = "watch_matched_c" 而不是 "bash",
#        所以 `pgrep -x bash` 会漏掉**全部**老 watcher(实测 16 个只找到 2 个)。
#        ⇒ 直接扫 /proc,不按 comm 过滤。
# 用法: list_watchers.sh            列出
#       list_watchers.sh kill ARM   只杀该 arm 的 watcher
set -u
MODE=${1:-list}; TARGET=${2:-}
for d in /proc/[0-9]*; do
  pid=${d#/proc/}
  [ "$pid" = "$$" ] && continue
  [ -r "$d/cmdline" ] || continue
  cl=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
  s=$(printf '%s\n' "$cl" | awk '{print $2}')
  case "$s" in *watch_matched_ckpt.sh) ;; *) continue ;; esac
  arm=$(printf '%s\n' "$cl" | awk '{print $3}')
  if [ "$MODE" = "kill" ]; then
    [ "$arm" = "$TARGET" ] || continue
    echo "  kill pid=$pid arm=$arm"; kill "$pid"
  else
    echo "  pid=$pid arm=$arm"
  fi
done
