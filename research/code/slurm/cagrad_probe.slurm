#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --job-name=cgprobe
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:4
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=1:00:00
#SBATCH --array=0-1
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%A_%a.out
#
# 测 CAGrad 的 min-norm 权重是否退化。PROBE_BS=1 把峰值显存压到 A40 装得下。
# 排除 ai-gpu01/03: 这两台各有一个泄漏的 GPU 进程(2.01/1.48 GiB),会把吃紧的作业挤爆。
set -uo pipefail
source /midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/env.sh
cd "$OPD_REPO"
NPROC=$(nvidia-smi -L | wc -l)
_RS=${SLURM_RESTART_COUNT:-0}
case "$NPROC" in 1|2|4|8) ;; *) [ "$_RS" -ge 3 ] && { echo "[guard] 已重启 $_RS 次,放弃"; exit 1; }
  echo "[guard] nproc=$NPROC -> requeue"; sleep 60; scontrol requeue "$SLURM_JOB_ID"; exit 0;; esac
export OPD_NPROC=$NPROC OPD_NUM_EPOCHS=4 OPD_EVAL_FREQ=0 OPD_RESUME=none PROBE_BS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-32} / NPROC ))
export MASTER_PORT=$(( 29000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID*53) % 20000 ))
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  export PROBE_TAG=base; unset OPD_STUDENT_INIT 2>/dev/null || true; export OPD_STUDENT_INIT=""
else
  export PROBE_TAG=s271 OPD_STUDENT_INIT="/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/CA-OPD/checkpoints/mopd_cagrad/logs/[MOPD-CAGRAD + SD3.5-M + PickScore+OCR+GenEval]/checkpoints/checkpoint-271/lora"
  [ -d "$OPD_STUDENT_INIT" ] || { echo "[probe] 找不到 $OPD_STUDENT_INIT"; exit 1; }
fi
echo "[cgprobe] tag=$PROBE_TAG nproc=$NPROC host=$(hostname) $(date)"
python -m accelerate.commands.launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_machines 1 --num_processes "$NPROC" --main_process_port "$MASTER_PORT" \
  -m scripts.train_sd3_opd --config config/opd.py:mopd_cagrad_probe
echo "[cgprobe] tag=$PROBE_TAG exit=$? $(date)"
