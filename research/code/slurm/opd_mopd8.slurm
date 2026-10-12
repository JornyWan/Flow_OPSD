#!/bin/bash
#SBATCH --job-name=opd_mopd8
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
#
# mopd 正式训练: 2 节点 × 4 卡 = 8 进程,复刻原 8×H200 的 72 samples/epoch。
# --gres 由提交时指定 (a100:4 或 h200:4),两边同时排队,先起来的抢锁并取消另一个。
# ⚠️ 锁的取消逻辑按 job-name=opd_mopd8 匹配,只适用于"同一 RUN_CFG 两种 GPU 竞速";跑多个不同 arm 请用 opd_arm.slurm。
# 提交: sbatch --gres=gpu:h200:4 opd_mopd8.slurm  /  sbatch --gres=gpu:a100:4 opd_mopd8.slurm
set -uo pipefail
RUN_CFG=${RUN_CFG:-mopd}
GPUS_PER_NODE=4
source /midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/env.sh
mkdir -p "$OPD_ROOT/outputs"

# ---- 互斥锁: mkdir 是原子的,先到的赢 ----
LOCK="$OPD_ROOT/outputs/.lock_${RUN_CFG}_8gpu"
if mkdir "$LOCK" 2>/dev/null; then
  echo "$SLURM_JOB_ID $(hostname) $(date)" > "$LOCK/owner"
  echo "[lock] 本作业 $SLURM_JOB_ID 抢到锁,取消同名的其他排队作业"
  for j in $(squeue -u "$USER" -h -n opd_mopd8 -o "%i" 2>/dev/null); do
    [ "$j" != "$SLURM_JOB_ID" ] && { echo "[lock]   scancel $j"; scancel "$j"; }
  done
else
  OWNER=$(cat "$LOCK/owner" 2>/dev/null)
  if [ "${OWNER%% *}" != "$SLURM_JOB_ID" ]; then
    echo "[lock] 锁已被作业 ${OWNER} 持有,本作业退出"; exit 0
  fi
  echo "[lock] 本作业是锁的持有者(requeue 后重入),继续"
fi

cd "$OPD_REPO"   # config 用 os.getcwd() 拼 dataset 路径
NNODES=${SLURM_NNODES:-1}
NPROC=$(( NNODES * GPUS_PER_NODE ))
HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
HEAD_IP=$(getent hosts "$HEAD" | awk '{print $1}')
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-48} / GPUS_PER_NODE ))
export NCCL_DEBUG=WARN
export OPD_RESUME=auto          # 超时/抢占后 requeue 能从最新 checkpoint 接上

echo "[mopd8] cfg=$RUN_CFG nodes=$NNODES nproc=$NPROC head=$HEAD($HEAD_IP) job=$SLURM_JOB_ID $(date)"
nvidia-smi -L
NG=$(nvidia-smi -L | wc -l)
[ "$NG" -ne "$GPUS_PER_NODE" ] && { echo "[guard] 需要 $GPUS_PER_NODE 卡,实得 $NG -> requeue"; sleep 60; scontrol requeue "$SLURM_JOB_ID"; exit 0; }

srun --ntasks-per-node=1 --export=ALL bash -c "
  source $OPD_ROOT/env.sh
  export OPD_RESUME=auto
  cd \$OPD_REPO
  echo \"[node \$SLURM_NODEID] \$(hostname) \$(nvidia-smi -L | wc -l) GPU\"
  python -m accelerate.commands.launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_machines $NNODES --num_processes $NPROC --machine_rank \$SLURM_NODEID \
    --main_process_ip $HEAD_IP --main_process_port 29500 \
    -m scripts.train_sd3_opd --config config/opd.py:$RUN_CFG
"
echo "[mopd8] exit=$? $(date)"
