#!/bin/bash
# 整理 watcher:杀掉聚合版(逐图版的输出已含 scores 均值),补齐逐图版。
# 写成脚本是为了避免 pgrep -f 匹配到调用它的命令行本身(上一次因此把自己的 shell 杀了)。
set -u
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
W="$B/codex_research/watch_matched_ckpt.sh"
SELF=$$

echo "== 杀聚合 watcher =="
k=0
while read -r pid args; do
  [ "$pid" = "$SELF" ] && continue
  case "$args" in
    /bin/bash*watch_ab_ckpt_hipri.sh*) kill "$pid" 2>/dev/null && k=$((k+1)) ;;
  esac
done < <(ps -u chw4026 -o pid=,args=)
echo "  已杀 $k 个"

echo "== 现有逐图 watcher 的标签 =="
have=""
while read -r pid args; do
  [ "$pid" = "$SELF" ] && continue
  case "$args" in
    /bin/bash*watch_matched_ckpt.sh*) have="$have $(echo "$args" | awk '{print $3}')" ;;
  esac
done < <(ps -u chw4026 -o pid=,args=)
echo "  [$have ]"

arm(){ # tag dir cks...
  local tag=$1 dir=$2; shift 2
  case " $have " in *" $tag "*) echo "  已有 $tag,跳过"; return;; esac
  nohup bash "$W" "$tag" "$dir" "$@" >/dev/null 2>&1 &
  echo "  挂上 $tag -> ck $*"
}
echo "== 补齐 =="
arm inner4  "$B/outputs/ab_open3_inner4_s20260830_u100"    400
arm gacc4   "$B/outputs/ab_open3_gacc4_s20260830_u100"     200 400
arm lr2x200 "$B/outputs/ab_open3_lr2x_s20260830_u200"      150 200
arm bases2  "$B/outputs/ab_open3_baseline_s20260905_u100"  50 100
arm lr2xs2  "$B/outputs/ab_open3_lr2x_s20260905_u100"      100
arm lr4x    "$B/outputs/ab_open3_lr4x_s20260830_u100"      100
arm gacc2   "$B/outputs/ab_open3_gacc2_s20260830_u100"     200
arm inner2b "$B/outputs/ab_open3_inner2_s20260830_u200"    400
arm baseAA  "$B/outputs/ab_open3_baseAA100_s20260830_u100" 100

sleep 2
echo "== 最终 =="
n1=0; n2=0
while read -r pid args; do
  [ "$pid" = "$SELF" ] && continue
  case "$args" in
    /bin/bash*watch_matched_ckpt.sh*) n1=$((n1+1)) ;;
    /bin/bash*watch_ab_ckpt_hipri.sh*) n2=$((n2+1)) ;;
  esac
done < <(ps -u chw4026 -o pid=,args=)
echo "  逐图 watcher: $n1   聚合 watcher: $n2"
