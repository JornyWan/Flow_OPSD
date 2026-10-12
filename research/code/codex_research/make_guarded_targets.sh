#!/bin/bash
# 为 0010 两臂在**同一组 epoch** 上建评测目标表。
# 两臂评相同 epoch 是硬性要求:公共 step 集不一致会造成选择效应(τ 上栽过一次)。
set -uo pipefail
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
D=$O/codex_research/runs/0010_guarded_pair
T=/midtier/minilab/scratch/chw4026/tmp/guarded_targets.tsv
EPOCHS="${1:?用法: make_guarded_targets.sh '10 20 30 ...'}"
: > "$T"
for e in $EPOCHS; do
  cd=$D/gdirect/artifacts/job_1035940/checkpoints/checkpoint-$e
  c3=$D/gguard3/artifacts/job_1035951/checkpoints/checkpoint-$e
  cg=$D/gguard/artifacts/job_1035941/checkpoints/checkpoint-$e
  # 只有**三臂都存在**该 epoch 才纳入,保证配对与剂量-反应可比
  if [ -d "$cd/lora" ] && [ -d "$c3/lora" ] && [ -d "$cg/lora" ]; then
    printf 'gdirect_%s\t%s\t4.5\t256\n' "$e" "$cd" >> "$T"
    printf 'gguard3_%s\t%s\t4.5\t256\n' "$e" "$c3" >> "$T"
    printf 'gguard_%s\t%s\t4.5\t256\n'  "$e" "$cg" >> "$T"
  else
    echo "[skip] epoch $e 三臂未同时就绪 (gdirect=$([ -d "$cd/lora" ] && echo y || echo n) gguard3=$([ -d "$c3/lora" ] && echo y || echo n) gguard=$([ -d "$cg/lora" ] && echo y || echo n))"
  fi
done
echo "目标表 $T 共 $(wc -l < "$T") 行:"; cut -f1 "$T" | tr '\n' ' '; echo
