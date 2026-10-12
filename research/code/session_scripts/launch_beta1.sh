#!/bin/bash
# β 修复臂:open3 的官方 β=0.1 会把模型被要求的位移放大 10×(驻点 d = g/β)。
# 官方 release 的单 reward preset 用 β=1.0 且无网格伪影 ⇒ 检验 β 是不是伪影的旋钮。
# 用通用 opacfg 臂 + 独立 OUTPUT_DIR,不改任何 slurm。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$O" || exit 1
for b in 1.0 0.3; do
  tag="beta${b/./}"
  OUT="$O/outputs/ab_open3_${tag}_s20260830_u100"
  J=$(ARM=opacfg OPA_CFG="--config.beta=${b}" OUTPUT_DIR="$OUT" \
      sbatch --parsable --job-name="${tag}_h4" \
      --account=minilab --partition=minilab-gpu --qos=high --gres=gpu:h200:4 "$O/slurm/ab_open3_v6.slurm" 2>&1)
  echo "  β=${b}  ARM=opacfg  OUT=$(basename $OUT)  -> job=$J"
done
