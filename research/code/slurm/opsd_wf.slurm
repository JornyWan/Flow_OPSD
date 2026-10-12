#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
#
# 消融: 把 OPSD 分支损失的归一化器冻结为常数(OPSD_WF_MODE=frozen),使损失由 1-齐次变 2-齐次,
# 从而让信赖域 rho **重新对更新产生一阶影响**。
# 判据: 在 frozen 下,rho=0.01 与 rho=0.10 的 12-epoch 训练 reward 增幅应当出现差异;
#       而 canonical(adaptive)下两者几乎重合(已实测)。
set -uo pipefail
OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
R=$OPD_ROOT/CA-OPD/reference_papers/DiffusionOPSD-main
SHIM=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad/opsd_shim
PY=$OPD_ROOT/../conda_envs/envs/diffusionopd/bin/python
REWARD=${REWARD:-pickscore}; UPDATES=${UPDATES:-12}
RHO=${RHO:-0.10}; WFMODE=${WFMODE:-frozen}
[ -f "$HOME/.secrets/hf_jorny.env" ] && . "$HOME/.secrets/hf_jorny.env" 2>/dev/null
cd "$R" || exit 1
NPROC=$(nvidia-smi -L | wc -l)
echo "[diag] 可见=$NPROC SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-?}"
case "$NPROC" in
  3|4|6|8) ;;
  *) _RS=${SLURM_RESTART_COUNT:-0}
     [ "$_RS" -ge 40 ] && { echo "[guard] nproc=$NPROC 不满足 48x24 分组,已重启 $_RS 次 -> 放弃"; exit 1; }
     echo "[guard] nproc=$NPROC 不满足 48x24 分组 -> requeue (第 $_RS 次)"; sleep 900
     scontrol requeue "$SLURM_JOB_ID"; exit 0;;
esac
export PYTHONPATH="$SHIM:$R"
export REWARD_CKPT_PATH=$OPD_ROOT/CA-OPD/code/DiffusionOPD-main/reward_ckpts
export PUBLIC_POLICY_WORLD_SIZE="$NPROC" PUBLIC_N_GPUS="$NPROC" PUBLIC_LAUNCH_WORLD_SIZE="$NPROC"
export ZIMAGE_HEAVY_BRIDGE=0 ZIMAGE_HEAVY_DIFF_BRIDGE=0 WANDB_MODE=offline
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-32} / NPROC ))
export OPSD_WF_MODE="$WFMODE"
_TAG=$(python3 -c "print(f'{int(round($RHO*1000)):03d}')")
OUT=$OPD_ROOT/outputs/opsd_wf_${WFMODE}_r${_TAG}
echo "[wf] reward=$REWARD rho=$RHO wf_mode=$WFMODE nproc=$NPROC out=$OUT $(date)"
"$PY" scripts/prepare_pickapic_prompts.py --quiet || exit 1
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  scripts/train_opsd_ri_sd3.py --config "config/public.py:sd35_${REWARD}" \
  --config.num_epochs="$UPDATES" --config.opsd.opa_rho="$RHO" --config.save_dir="$OUT" 2>&1 | grep -avE "[0-9]+%\|"
_RC=$?
echo "[wf] rho=$RHO wf=$WFMODE exit=$_RC $(date)"
exit $_RC
