#!/bin/bash
set -u
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
W="$B/codex_research/watch_matched_ckpt.sh"; SELF=$$; have=""
while read -r pid args; do
  [ "$pid" = "$SELF" ] && continue
  case "$args" in /bin/bash*watch_matched_ckpt.sh*) have="$have $(echo "$args"|awk '{print $3}')";; esac
done < <(ps -u chw4026 -o pid=,args=)
for spec in "lr8xs2:$B/outputs/ab_open3_lr8x_s20260905_u100" "lr12x:$B/outputs/ab_open3_lr12x_s20260830_u100"; do
  t=${spec%%:*}; d=${spec#*:}
  case " $have " in
    *" $t "*) echo "  已有 $t watcher";;
    *) nohup bash "$W" "$t" "$d" 100 >/dev/null 2>&1 & echo "  已挂 $t -> ck100 (pid=$!)";;
  esac
done
ps -u chw4026 -o args= | grep '[w]atch_matched_ckpt.sh' | awk '{printf "    %-10s ck=%s\n",$3,$5}'
