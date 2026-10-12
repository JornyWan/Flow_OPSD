#!/bin/bash
#SBATCH --job-name=opd_train
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
#
# DiffusionOPD / CA-OPD 训练。原服务器是 8×H200 单节点;本集群单节点最多 4 卡,
# 所以用 2 节点 × 4 卡 = 8 进程复刻同样的 effective batch (8×bs3×3batch = 72 samples/epoch)。
#
# 用法:  RUN_CFG=mopd sbatch opd_train.slurm
#        RUN_CFG=mopd_caopd sbatch opd_train.slurm
#        单节点 4 卡冒烟:  RUN_CFG=mopd sbatch --nodes=1 opd_train.slurm   (注意:effective batch 减半,只用于验证跑通)
set -uo pipefail
RUN_CFG=${RUN_CFG:-mopd}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}

source /midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/env.sh
mkdir -p "$OPD_ROOT/outputs"
cd "$OPD_REPO"   # config 里多处用 os.getcwd() 拼 dataset 路径,必须在 repo 根目录跑

NNODES=${SLURM_NNODES:-1}
NPROC=$(( NNODES * GPUS_PER_NODE ))
HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
HEAD_IP=$(getent hosts "$HEAD" | awk '{print $1}')
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-48} / GPUS_PER_NODE ))
export NCCL_DEBUG=WARN

echo "[opd] cfg=$RUN_CFG nodes=$NNODES gpus/node=$GPUS_PER_NODE nproc=$NPROC head=$HEAD($HEAD_IP) job=$SLURM_JOB_ID $(date)"
nvidia-smi -L

# GPU 降级守卫 (SLURM 有时给不满卡)
NG=$(nvidia-smi -L | wc -l)
[ "$NG" -ne "$GPUS_PER_NODE" ] && { echo "[guard] 需要 $GPUS_PER_NODE 卡,实得 $NG -> requeue"; sleep 60; scontrol requeue "$SLURM_JOB_ID"; exit 0; }

srun --ntasks-per-node=1 --export=ALL bash -c "
  source $OPD_ROOT/env.sh
  cd \$OPD_REPO
  echo \"[rank-node \$SLURM_NODEID] host=\$(hostname)\"
  python -m accelerate.commands.launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_machines $NNODES \
    --num_processes $NPROC \
    --machine_rank \$SLURM_NODEID \
    --main_process_ip $HEAD_IP \
    --main_process_port 29500 \
    -m scripts.train_sd3_opd --config config/opd.py:$RUN_CFG
"
echo "[opd] exit=$? $(date)"
