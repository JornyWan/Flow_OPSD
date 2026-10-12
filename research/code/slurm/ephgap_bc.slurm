#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --job-name=ephgapBC
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=5:00:00
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
#
# 决定性实验(决定 bootopd 那 26 小时值不值):
# 同一个已收敛的 student(tau02@601)上,对拍两个 teacher 的相对信号强度
#   A) 固定的官方 AesTeacher
#   B) 从**该 student** 出发跑 25 epoch GRPO 得到的 ephemeral 专家
# B > A 才说明 BootstrapOPD 能提供更新鲜的信号;B <= A 则前提虽成立但机制无效。
set -uo pipefail
source /midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/env.sh
cd "$OPD_REPO"
NPROC=$(nvidia-smi -L | wc -l)
case "$NPROC" in 1|2|4|8) ;; *) echo "[guard] nproc=$NPROC -> requeue"; sleep 60; scontrol requeue "$SLURM_JOB_ID"; exit 0;; esac
export OPD_NPROC=$NPROC OPD_EVAL_FREQ=0 OPD_RESUME=none
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-32} / NPROC ))
BP=$(( 29000 + SLURM_JOB_ID % 20000 ))
STU="/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/CA-OPD/code/DiffusionOPD-main/logs/[MOPD-CAOPD-TAU0.2 + SD3.5-M + PickScore+OCR+GenEval]/checkpoints/checkpoint-601/lora"
launch(){ python -m accelerate.commands.launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_machines 1 --num_processes "$NPROC" --main_process_port "$1" -m "$2" --config "$3"; }

echo "[ephgap] host=$(hostname) nproc=$NPROC student=$STU $(date)"

# 阶段 A 已在作业 1034012 完成: 固定 AesTeacher 在该 student 上的 gap = 1.42-1.45%
# 这里只跑 B(训练 25 epoch ephemeral 专家)与 C(测它的 gap),两者直接对拍。

# --- B) 训练 ephemeral 专家(25 epoch, 从同一 student 出发) ---
export EPH_ON=1 EPH_EPOCHS=${EPH_EPOCHS:-25} BOOT_TAG=gapx CASCADE_PREV_LORA="$STU"
unset OPD_SMOKE 2>/dev/null || true
echo "[ephgap] B1) 训练 ephemeral 专家 25 epoch"
launch $((BP+2)) scripts.baselines.train_sd3_GRPO_Guard_multireward "config/baselines.py:eph_aes"
echo "[ephgap] B1 exit=$?"
EXP=""; for c in "logs/Bootstrap-gapx_aes/checkpoints/checkpoint-"*; do
  [ -d "$c/lora" ] && n="${c##*-}" && case "$n" in ''|*[!0-9]*) ;; *) EXP="$c/lora";; esac; done
[ -n "$EXP" ] || { echo "[ephgap] 专家没产出 checkpoint,中止"; exit 1; }
echo "[ephgap] 专家 = $EXP"

# --- C) 该专家的 gap(同一 student, 同一设置) ---
export AES_TEACHER_PATH="$EXP" PROBE_TAG=eph601 OPD_NUM_EPOCHS=4
launch $((BP+3)) scripts.train_sd3_opd "config/opd.py:sopd_gapprobe"
echo "[ephgap] C exit=$? $(date)"
