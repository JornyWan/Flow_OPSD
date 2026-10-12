#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --job-name=gapprobeA
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1:30:00
#SBATCH --array=4-4
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%A_%a.out
#
# Bootstrap 前提诊断:在不同训练进度的 student checkpoint 上测 teacher-student 相对差距。
# 纯测量(lr=0),不改变被测 checkpoint。
set -uo pipefail
source /midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/env.sh
cd "$OPD_REPO"
STEPS=(1 121 331 601 991)
S=${STEPS[$SLURM_ARRAY_TASK_ID]}
NPROC=$(nvidia-smi -L | wc -l)
# SLURM 有时会降配(要 4 张给 3 张),此时 nproc 不整除 8,配置缩放会失效 -> 必须 requeue。
# 但要限次数:实测遇到过节点持续只给 3 张,导致无限 requeue 空转 17 次。
_RS=${SLURM_RESTART_COUNT:-0}
case "$NPROC" in
  1|2|4|8) ;;
  *) if [ "$_RS" -ge 3 ]; then echo "[guard] nproc=$NPROC 且已重启 $_RS 次 -> 放弃,不再 requeue"; exit 1; fi
     echo "[guard] nproc=$NPROC 不整除 8 -> requeue (第 $_RS 次)"; sleep 60; scontrol requeue "$SLURM_JOB_ID"; exit 0;; esac
export OPD_NPROC=$NPROC
export OPD_NUM_EPOCHS=4          # 只为收集指标
export OPD_EVAL_FREQ=0           # 完全关掉在线 eval
export OPD_RESUME=none
export PROBE_TAG="s${S}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-16} / NPROC ))
export MASTER_PORT=$(( 29000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 37) % 20000 ))
export OPD_STUDENT_INIT="/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/CA-OPD/code/DiffusionOPD-main/logs/[MOPD-CAOPD-TAU0.2 + SD3.5-M + PickScore+OCR+GenEval]/checkpoints/checkpoint-${S}/lora"
[ -d "$OPD_STUDENT_INIT" ] || { echo "[probe] 找不到 $OPD_STUDENT_INIT"; exit 1; }
echo "[probe] step=$S nproc=$NPROC host=$(hostname) init=$OPD_STUDENT_INIT $(date)"
python -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_machines 1 --num_processes "$NPROC" --main_process_port "$MASTER_PORT" \
  -m scripts.train_sd3_opd --config config/opd.py:mopd_gapprobe
echo "[probe] step=$S exit=$? $(date)"
