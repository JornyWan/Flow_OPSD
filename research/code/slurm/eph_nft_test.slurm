#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --job-name=ephnft
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=5:00:00
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
#
# 只训练,不做探针: 问题是"本仓库的 NFT 能否在短程内提升 reward"。
# 答案直接来自训练日志的 reward_avg 曲线。
# 对照: GRPO 版 25 epoch 的 reward 从 1.5809 降到 1.5337(~8 SEM)。
# 背景: 归档 mt_nft 曲线显示 NFT 从 base 出发在 ~630 步达峰(geneval 0.21->0.87),
#       所以若 60 步仍不涨,说明"短程"这个前提本身不成立。
set -uo pipefail
source /midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/env.sh
cd "$OPD_REPO"
NPROC=$(nvidia-smi -L | wc -l)
_RS=${SLURM_RESTART_COUNT:-0}
case "$NPROC" in 1|2|4|8) ;; *) [ "$_RS" -ge 3 ] && { echo "[guard] 已重启 $_RS 次,放弃"; exit 1; }
  echo "[guard] nproc=$NPROC -> requeue"; sleep 900; scontrol requeue "$SLURM_JOB_ID"; exit 0;; esac
export OPD_NPROC=$NPROC OPD_EVAL_FREQ=0 OPD_RESUME=none
export EPH_ON=1 EPH_EPOCHS=${EPH_EPOCHS:-60} BOOT_TAG=nfttest
export CASCADE_PREV_LORA="/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/CA-OPD/code/DiffusionOPD-main/logs/[MOPD-CAOPD-TAU0.2 + SD3.5-M + PickScore+OCR+GenEval]/checkpoints/checkpoint-601/lora"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-32} / NPROC ))
export MASTER_PORT=$(( 29000 + SLURM_JOB_ID % 20000 ))
[ -d "$CASCADE_PREV_LORA" ] || { echo "[ephnft] 找不到 student: $CASCADE_PREV_LORA"; exit 1; }
echo "[ephnft] nproc=$NPROC epochs=$EPH_EPOCHS host=$(hostname) $(date)"
python -m accelerate.commands.launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_machines 1 --num_processes "$NPROC" --main_process_port "$MASTER_PORT" \
  -m scripts.baselines.train_nft_sd3_multireward --config config/baselines.py:eph_aes_nft
echo "[ephnft] exit=$? $(date)"
