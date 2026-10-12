#!/bin/bash
#SBATCH --job-name=opd_smoke
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=2:00:00
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
#
# 冒烟测试:单节点 4 卡跑 mopd(3 teacher,把 pickscore/ocr/geneval 三条 reward 路径全走一遍)。
# 只验证"跑得通",不出数 —— effective batch 是正式跑(8进程)的一半。
set -uo pipefail
RUN_CFG=${RUN_CFG:-mopd}
source /midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/env.sh
cd "$OPD_REPO"
NPROC=$(nvidia-smi -L | wc -l)
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-48} / NPROC ))

echo "[smoke] cfg=$RUN_CFG nproc=$NPROC host=$(hostname) job=$SLURM_JOB_ID $(date)"
nvidia-smi -L
echo "[smoke] --- GPU 上的环境自检 ---"
python - <<'PY'
import torch
print("  torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| ngpu", torch.cuda.device_count(), "|", torch.cuda.get_device_name(0))
x=torch.randn(64,64,device="cuda"); print("  cuda matmul ok:", float((x@x).sum()))
import mmcv, mmcv._ext, mmdet; print("  mmcv", mmcv.__version__, "mmdet", mmdet.__version__, "(_ext OK)")
import paddle; print("  paddle", paddle.__version__, "cuda:", paddle.is_compiled_with_cuda())
PY
[ $? -ne 0 ] && { echo "[smoke] 环境自检失败,不启动训练"; exit 42; }

echo "[smoke] --- 启动训练 ---"
python -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_machines 1 --num_processes $NPROC --main_process_port 29500 \
  -m scripts.train_sd3_opd --config "config/opd.py:$RUN_CFG"
echo "[smoke] exit=$? $(date)"
