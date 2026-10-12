#!/bin/bash
set -u
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
W="$B/codex_research/watch_matched_ckpt.sh"
SELF=$$
have=""
while read -r pid args; do
  [ "$pid" = "$SELF" ] && continue
  case "$args" in /bin/bash*watch_matched_ckpt.sh*) have="$have $(echo "$args" | awk '{print $3}')";; esac
done < <(ps -u chw4026 -o pid=,args=)
case " $have " in
  *" lr4xs2 "*) echo "  已有 lr4xs2 watcher,跳过";;
  *) nohup bash "$W" lr4xs2 "$B/outputs/ab_open3_lr4x_s20260905_u100" 100 >/dev/null 2>&1 &
     echo "  已挂 lr4xs2 -> ck100 (pid=$!)";;
esac
echo "  现存逐图 watcher:"
ps -u chw4026 -o args= | grep watch_matched_ckpt.sh | grep -v grep | awk '{printf "    %-10s ck=%s\n",$3,$5}'
