#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=preempt_gpu
#SBATCH --qos=low
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=96G
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=p12tpd
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%A_%a.out
# §4.866 S0:TPD 零训练筛选。环境照抄 P10(scratchpad/p10.slurm)。数组下标 = prompt 分片。
set -uo pipefail
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
R=$B/CA-OPD/code/DiffusionOPSD-rpi
PY=$B/../conda_envs/envs/diffusionopd/bin/python
P=$R/data/pickapic/test.txt
TAG=${TAG:?}; LORA=${LORA:?}; NSHARD=${NSHARD:-1}; SH=${SLURM_ARRAY_TASK_ID:-0}
export HF_HUB_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPATH=$B/CA-OPD/code/shim_rpi:$R/src:$R
mkdir -p $B/outputs/p12
OUT=$B/outputs/p12/${TAG}_sh${SH}of${NSHARD}.json
[ -s "$OUT" ] && { echo "[p12] $OUT 已存在,跳过"; exit 0; }
echo "[p12] tag=$TAG shard=$SH/$NSHARD host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0) $(date)"
$PY $B/probes/p12_tpd.py --repo $R --lora $LORA --prompts $P --out $OUT \
  --shard $SH --nshard $NSHARD ${EXTRA:-} 2>&1 | grep --line-buffered -vE "it/s|%\|"
exit ${PIPESTATUS[0]}
