#!/bin/bash
set -u
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
W="$B/codex_research/watch_matched_ckpt.sh"; SELF=$$; have=""
while read -r pid args; do
  [ "$pid" = "$SELF" ] && continue
  case "$args" in *watch_matched_ckpt.sh\ *) have="$have $(echo "$args"|awk '{print $3}')";; esac
done < <(ps -u chw4026 -o pid=,args=)
case " $have " in
  *" lr12xeps "*) echo "  已有 lr12xeps watcher";;
  *) nohup bash "$W" lr12xeps "$B/outputs/ab_open3_lr12xeps_s20260830_u100" 100 >/dev/null 2>&1 &
     echo "  已挂 lr12xeps -> ck100 (pid=$!)";;
esac
ps -u chw4026 -o args= | grep '[w]atch_matched_ckpt.sh' | awk '{printf "    %-10s ck=%s\n",$3,$5}'
