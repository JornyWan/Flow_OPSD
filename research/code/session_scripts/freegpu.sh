#!/bin/bash
# 按 GPU 型号统计各分区空闲(排除 RESERVED/DRAIN/DOWN)。§4.623 §五 的教训:加总型号会误导。
# 无参数时默认扫四个分区 —— 空参会让整个循环不执行、输出全空,
# 而"空输出"会被读成"没有空闲 GPU"从而无限期压制排队(静默假阴性)。
if [ "$#" -eq 0 ]; then set -- minilab-gpu scu-gpu sablab-gpu preempt_gpu; fi
for p in "$@"; do
  declare -A F=()
  for n in $(sinfo -p "$p" -h -o "%n" | sort -u); do
    s=$(scontrol show node "$n" 2>/dev/null) || continue
    st=$(echo "$s" | grep -oP 'State=\K\S+')
    case "$st" in *RESERVED*|*DRAIN*|*DOWN*|*NOT_RESPOND*) continue;; esac
    while read -r ty c; do
      a=$(echo "$s" | grep -oP 'AllocTRES=\S+' | grep -oP "gres/gpu:$ty=\K[0-9]+" | head -1)
      F[$ty]=$(( ${F[$ty]:-0} + c - ${a:-0} ))
    done < <(echo "$s" | grep -oP 'CfgTRES=\S+' | grep -oP 'gres/gpu:\K\w+=[0-9]+' | tr '=' ' ')
  done
  printf "  %-12s" "$p"
  for ty in "${!F[@]}"; do printf " %s=%s" "$ty" "${F[$ty]}"; done
  echo
  unset F
done
