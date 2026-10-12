#!/bin/bash
#SBATCH --job-name=opd_resumetest
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=2:00:00
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
# 两阶段验证 resume: 先跑 2 epoch 存档 -> 再以 OPD_RESUME=auto 续到 4 epoch,检查是否从 epoch 2 接上。
set -uo pipefail
source /midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/env.sh
cd "$OPD_REPO"
NPROC=$(nvidia-smi -L | wc -l)
export OMP_NUM_THREADS=$(( 48 / NPROC ))
CKDIR="logs/[RESUMETEST + SD3.5-M + PickScore]/checkpoints"
rm -rf "logs/[RESUMETEST + SD3.5-M + PickScore]"

run() { python -m accelerate.commands.launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
        --num_machines 1 --num_processes $NPROC --main_process_port $1 \
        -m scripts.train_sd3_opd --config config/opd.py:sopd_resumetest; }

echo "############ 阶段1: 从头跑 2 epoch (RESUMETEST_EPOCHS=2, 无 resume) ############"
OPD_RESUME=none RESUMETEST_EPOCHS=2 run 29511
echo "[phase1] exit=$?"
echo "--- 阶段1 产出 ---"; ls -d "$CKDIR"/checkpoint-* 2>/dev/null
ls -la "$CKDIR"/*/train_state.pt 2>/dev/null | awk '{print "   train_state:", $5, $9}'

echo "############ 阶段2: OPD_RESUME=auto 续到 4 epoch ############"
OPD_RESUME=auto RESUMETEST_EPOCHS=4 run 29512
echo "[phase2] exit=$?"
echo "--- 阶段2 产出 ---"; ls -d "$CKDIR"/checkpoint-* 2>/dev/null
echo "RESUMETEST_FINISHED"
