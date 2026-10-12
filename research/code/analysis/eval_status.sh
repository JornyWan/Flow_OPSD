#!/bin/bash
# 统计评测完成度。
# 关键: 必须先 ls 父目录强制刷新 NFS 属性缓存,否则计算节点刚写完的文件在登录节点上
# 用 [ -f ] 测会返回"不存在"(实测 caopd721/drawbench 已写入 15 分钟仍被判为缺失)。
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
ls "$O/outputs/eval" >/dev/null 2>&1
for t in "$@"; do
  n=0; miss=""
  for ds in drawbench geneval ocr; do
    d="$O/outputs/eval/${t}_${ds}"
    ls "$d" >/dev/null 2>&1          # 逐个目录也刷一次
    if [ -s "$d/average_scores.json" ]; then n=$((n+1)); else miss="$miss$ds "; fi
  done
  printf "%-14s %s/3" "$t" "$n"; [ -n "$miss" ] && printf "  缺: %s" "$miss"; echo
done
