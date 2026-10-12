#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
#
# Phase 1: 用**官方 DiffusionOPSD 代码**复现论文结果(不是我们的重写版)。
# 动机: RTC 的立论前提("98.7% 构造增益丢失")经查是把论文两个不同实验的数字相除得来的,
#       论文从未做这个除法。正确顺序是先忠实复现 OPSD、确认它如论文所说 work,再谈我们的 idea。
# 提交: SMOKE_TEST=1 sbatch --job-name=opsdsmk --gres=gpu:h200:3 opsd_repro.slurm
set -uo pipefail
OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
R=$OPD_ROOT/CA-OPD/reference_papers/DiffusionOPSD-main
SHIM=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad/opsd_shim
PY=$OPD_ROOT/../conda_envs/envs/diffusionopd/bin/python
REWARD=${REWARD:-pickscore}
UPDATES=${UPDATES:-100}
[ -f "$HOME/.secrets/hf_jorny.env" ] && . "$HOME/.secrets/hf_jorny.env" 2>/dev/null

cd "$R" || exit 1
NPROC=$(nvidia-smi -L | wc -l)
# 诊断: SLURM 声称分配的 gres 与作业内实际可见的 GPU 数多次不一致
# (squeue 显示 gres/gpu:h200:3,而 nvidia-smi -L 只见 2)。把两边都打出来定位真相。
echo "[diag] nvidia-smi 可见=$NPROC | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<未设>}"
echo "[diag] SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<未设>} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<未设>} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-<未设>}"
echo "[diag] scontrol 记录: $(scontrol show job $SLURM_JOB_ID 2>/dev/null | grep -oE 'TresPerNode=[^ ]*|GRES=[^ ]*' | tr '\n' ' ')"
# 官方 config/public.py 要求 48x24 分组:实测仅 3/4/6/8 卡可满足(1/2 卡直接报错)。
case "$NPROC" in
  3|4|6|8) ;;
  *) _RS=${SLURM_RESTART_COUNT:-0}
     # 集群上他人作业常占住 1~2 张 H200,使本节点无法凑够 3 张。这不是配置错误,是排队现实,
     # 应当**耐心等待**而非放弃:上限放宽到 40 次 x 15 分钟 ≈ 10 小时。
     if [ "$_RS" -ge 40 ]; then echo "[guard] nproc=$NPROC 不满足官方 48x24 分组且已重启 $_RS 次(约 10h) -> 放弃"; exit 1; fi
     echo "[guard] nproc=$NPROC 不满足官方 48x24 分组(需 3/4/6/8) -> requeue (第 $_RS 次,15 分钟后重试)"
     sleep 900; scontrol requeue "$SLURM_JOB_ID"; exit 0;;
esac
# 包名 diffusionopsd 映射到 src/(见 pyproject 的 package-dir),用符号链接垫片而非
# pip install -e —— 避免改动正在被其他作业使用的共享 conda env。
export PYTHONPATH="$SHIM:$R"
export PUBLIC_N_GPUS="$NPROC" PUBLIC_LAUNCH_WORLD_SIZE="$NPROC" PUBLIC_POLICY_WORLD_SIZE="$NPROC"
export ZIMAGE_HEAVY_BRIDGE=0 ZIMAGE_HEAVY_DIFF_BRIDGE=0
export WANDB_MODE=${WANDB_MODE:-offline}
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-32} / NPROC ))
export MASTER_PORT=$(( 29000 + SLURM_JOB_ID % 20000 ))
OUTPUT_DIR=${OUTPUT_DIR:-$OPD_ROOT/outputs/opsd_ls_sd35_${REWARD}}

echo "[opsd] reward=$REWARD nproc=$NPROC updates=$UPDATES smoke=${SMOKE_TEST:-0} host=$(hostname) job=$SLURM_JOB_ID $(date)"
nvidia-smi -L
"$PY" scripts/prepare_pickapic_prompts.py --quiet || exit 1
"$PY" scripts/check_reward_setup.py --reward "$REWARD" --backbone sd35 || exit 1

# ★ 新方法臂: opa_ls=K 逐样本线搜索(见 config/opsd_defaults.py 注释)。同时开 cert=1 以实测采纳率,
# 与 opsdc100(cert=1, ls=0) 构成单变量对照。
# canonical 配置是 opa_cert=0 -> `opa_ap[s:e] = 1.0` **硬编码无条件接受**,
# 即从不验证构造出的目标是否真的提升**端点** reward(只有查询点处的一阶保证)。
# opa_cert=1 会跑 exact-dpm2 后缀续解并给端点打分,得到**真实采纳率**。
# 判据: 若真实 acc 明显 < 1,则 canonical 路径确实在使用大量无效目标。
CMD=("$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC"
     scripts/train_opsd_ri_sd3.py
     --config "config/public.py:sd35_${REWARD}"
     --config.num_epochs="$UPDATES"
     --config.opsd.opa_cert=${OPA_CERT:-1}
     --config.opsd.opa_ls=${OPA_LS:-4}
     # opa_mb 是纯分块循环(逐样本独立),减小它只降显存不改结果 —— L40(44.4GiB) 上 mb=6 会 OOM
     --config.opsd.opa_mb=${OPA_MB:-6}
     --config.opsd.opa_mq_sigmas="${OPA_MQ:-}"
     --config.opsd.opa_mq_weight=${OPA_MQ_W:-0.3}
     --config.opsd.opa_ls_mode="${OPA_LS_MODE:-target}"
     --config.opsd.opa_dir_mode="${OPA_DIR:-grad}"
     --config.opsd.opa_topk=${OPA_TOPK:-0}
     --config.save_dir="$OUTPUT_DIR")
if [ "${SMOKE_TEST:-0}" = "1" ]; then
  CMD+=(--config.debug=true --config.sample.num_batches_per_epoch=1 --config.train.gradient_accumulation_steps=1)
fi
echo "[opsd] CMD: ${CMD[*]}"
"${CMD[@]}"
_RC=$?
echo "[opsd] reward=$REWARD exit=$_RC $(date)"
exit $_RC
