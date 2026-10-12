#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
#
# Gate A: 零方法探针 —— 测官方 Open3 的**标量求和**上升方向,是否在三个 reward 上同时改善。
# 判据与 kill 条件先于数据写定,见 codex_research/0040_GATEA_VECTOR_FEASIBILITY_PREREG.md。
#
# **这不是 smoke**:config / 数据 / batch / β / gs 全部与正式实验逐位相同(sd35_open3,
# pickapic 25415 条,1152 samples/epoch,β=0.1,gs=1.0)。只是 epoch 数取到"够统计"为止 ——
# 探针测的是**每个 on-policy state 上的目标几何**,不是训练终点,不需要跑满 300 步。
set -uo pipefail
OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
R=$OPD_ROOT/CA-OPD/code/DiffusionOPSD-rpi                 # fork(仅多一个 env 守卫的探针)
SHIM=$OPD_ROOT/CA-OPD/code/shim_rpi
PY=$OPD_ROOT/../conda_envs/envs/diffusionopd/bin/python
EPOCHS=${EPOCHS:-6}
OUTPUT_DIR=${OUTPUT_DIR:-$OPD_ROOT/outputs/gateA_open3_probe}
[ -f "$HOME/.secrets/hf_jorny.env" ] && . "$HOME/.secrets/hf_jorny.env" 2>/dev/null
cd "$R" || exit 1
NPROC=$(nvidia-smi -L | wc -l)
echo "[diag] nvidia-smi 可见=$NPROC | SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<未设>}"
echo "[diag] scontrol AllocTRES: $(scontrol show job $SLURM_JOB_ID 2>/dev/null | grep -oE 'AllocTRES=[^ ]*' | head -1)"
case "$NPROC" in
  3|4|6|8) ;;
  *) _RS=${SLURM_RESTART_COUNT:-0}
     if [ "$_RS" -ge 40 ]; then echo "[guard] nproc=$NPROC 且已重启 $_RS 次 -> 放弃"; exit 1; fi
     echo "[guard] nproc=$NPROC 不满足官方 48x24 分组 -> requeue(第 $_RS 次)"
     sleep 900; scontrol requeue "$SLURM_JOB_ID"; exit 0;;
esac
export PYTHONPATH="$SHIM:$R"
export PUBLIC_N_GPUS="$NPROC" PUBLIC_LAUNCH_WORLD_SIZE="$NPROC" PUBLIC_POLICY_WORLD_SIZE="$NPROC"
export ZIMAGE_HEAVY_BRIDGE=0 ZIMAGE_HEAVY_DIFF_BRIDGE=0
export OPSD_WF_MODE=adaptive        # 见 0041 §③:该 env 会改目标函数且不进 manifest,显式中和
export WANDB_MODE=${WANDB_MODE:-offline}
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-32} / NPROC ))
export MASTER_PORT=$(( 29000 + SLURM_JOB_ID % 20000 ))
export REWARD_CKPT_PATH=$OPD_ROOT/CA-OPD/code/DiffusionOPD-main/reward_ckpts
export RPI_A1=0                     # ⚠️ 探针**绝不能带干预** —— A1 会改 opa_ap,
                                    #    那样测到的就不是 baseline 的目标几何了。
                                    #    默认虽为 0,但 sbatch 用 --export=ALL,
                                    #    环境里若有残留就会被带进来,必须显式中和。
export RPI_GATEA_PROBE=1            # ← 唯一与 baseline 的差别
echo "[gateA] fork=$R epochs=$EPOCHS nproc=$NPROC host=$(hostname) job=$SLURM_JOB_ID $(date)"
echo "[gateA] forked_from=$(cat $R/FORKED_FROM_COMMIT.txt 2>/dev/null)"
"$PY" "$OPD_ROOT/preflight_check.py" --repo "$R" --config sd35_open3 --nproc "$NPROC" \
      --expect-fork-clean --allow-env RPI_GATEA_PROBE \
      || { echo "[gateA] ❌ preflight 未通过,放弃"; exit 42; }
"$PY" scripts/check_reward_setup.py --reward open3 --backbone sd35 || exit 1
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" \
     scripts/train_opsd_ri_sd3.py \
     --config "config/public.py:sd35_open3" \
     --config.num_epochs="$EPOCHS" \
     --config.save_dir="$OUTPUT_DIR"
_RC=$?
echo "[gateA] exit=$_RC $(date)"
ls -l "$OUTPUT_DIR/gateA_probe.jsonl" 2>/dev/null && \
  echo "[gateA] 探针记录行数: $(wc -l < "$OUTPUT_DIR/gateA_probe.jsonl")"
exit $_RC
