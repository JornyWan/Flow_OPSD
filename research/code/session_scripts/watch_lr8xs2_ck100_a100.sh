#!/bin/bash
# lr8x_s2 的 ck100 专用 watcher,投 **A100**。
# 理由:§4.287/§4.302 的判据是在 A100 口径(对照 bases2_100_a100)上登记的;
# §4.318 实测同一 Δ 跨评测硬件可差 0.00243,占 §4.302 窄带宽度的 34%,足以翻转边界判定。
# 共享的 watch_matched_ckpt.sh 固定投 A40,不改它(它同时服务其它臂),这里单开一个。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
DIR="$O/outputs/ab_open3_lr8x_s20260905_u100"
CK="$DIR/checkpoints/checkpoint-100"
OUT="$O/outputs/matched_pi_lr8xs2_100_a100_default_drawbench_flow40.json"
for _ in $(seq 1 1440); do
  ls "$DIR" >/dev/null 2>&1                       # 刷 NFS 缓存
  [ -f "$CK/lora/adapter_config.json" ] && break
  sleep 60
done
[ -f "$CK/lora/adapter_config.json" ] || { echo "[a100w] ck100 超时未出现"; exit 1; }
sleep 60                                          # 宽限:等 ema/optimizer 写完
[ -f "$OUT" ] && { echo "[a100w] 已有结果,跳过"; exit 0; }
J=$(ARM="pi_lr8xs2_100_a100" ADAPTER=default CKDIR="$CK" CELLS="drawbench:flow:40" \
    REWARDS=hpsv2,clipscore,aesthetic,pickscore,imagereward \
    sbatch --parsable --job-name="mev_pi_lr8xs2_100_a100" \
    --account=sablab --partition=sablab-gpu \
    --gres=gpu:a100:1 --exclude=ai-gpu01,ai-gpu03 "$O/slurm/opsd_eval_matched.slurm")
echo "[a100w] ck100 就绪 -> A100 逐图评测 job=$J $(date)"
