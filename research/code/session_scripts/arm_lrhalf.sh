#!/bin/bash
set -u
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
W="$B/codex_research/watch_matched_ckpt.sh"; SELF=$$; have=""
while read -r pid args; do
  [ "$pid" = "$SELF" ] && continue
  case "$args" in /bin/bash*watch_matched_ckpt.sh*) have="$have $(echo "$args"|awk '{print $3}')";; esac
done < <(ps -u chw4026 -o pid=,args=)
case " $have " in
  *" lrhalf "*) echo "  已有 lrhalf watcher";;
  *) nohup bash "$W" lrhalf "$B/outputs/ab_open3_lrhalf_s20260830_u100" 100 >/dev/null 2>&1 &
     echo "  已挂 lrhalf -> ck100 (pid=$!)";;
esac
ps -u chw4026 -o args= | grep watch_matched_ckpt.sh | grep -v grep | awk '{printf "    %-10s ck=%s\n",$3,$5}'
