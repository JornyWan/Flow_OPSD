#!/bin/bash
# ck200 已落盘 11 分钟但 watcher 未触发(疑 NFS 属性缓存)。手工补投,保证 §4.459 判据数据产出。
# 参数与 watcher 完全一致:preempt_gpu + qos=low(不抢占)、A40、排除 gpu01/03。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$O" || exit 1
CK="$O/outputs/ab_open3_inner2H100_s20260830_u200/checkpoints/checkpoint-200"
OUT="$O/outputs/matched_pi_inner2H100_200_default_drawbench_flow40.json"
[ -f "$OUT" ] && { echo "  产物已存在,跳过"; exit 0; }
[ -f "$CK/lora/adapter_config.json" ] || { echo "  ✗ checkpoint 不完整"; exit 1; }
J=$(ARM="pi_inner2H100_200" ADAPTER=default CKDIR="$CK" CELLS="drawbench:flow:40" \
    REWARDS=hpsv2,clipscore,aesthetic,pickscore,imagereward \
    sbatch --parsable --job-name="mev_pi_inner2H100_200" \
    --account=sablab --partition=preempt_gpu --qos=low \
    --gres=gpu:a40:1 --exclude=ai-gpu01,ai-gpu03 "$O/slurm/opsd_eval_matched.slurm" 2>&1)
echo "  已提交 job=$J"
