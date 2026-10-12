#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=preempt_gpu
#SBATCH --qos=low
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --mem=400G
#SBATCH --time=03:00:00
#SBATCH --exclusive --gres=gpu:h100:4
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
set -u
# code-correctness smoke for the **pushed** Flow_OPSD repo:跑 2 个 update,确认
# (a) MSP 路径不抛异常、(b) 生产端/消费端链条接上(否则脚本自己会 raise)、
# (c) canonical 路径(MSP_STATES 未设)与官方一致。不用于比数值。
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
# ⚠️ 绝对路径。初版用相对路径少算一层 `..`,readlink -f 返回空 ⇒ R="" ⇒
# `cd "" || exit 1` 是**空操作且返回 0**,于是在提交目录下跑、Python 找不到脚本(exit 2)。
R=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad/Flow_OPSD
[ -f "$R/scripts/train_opsd_ri_sd3.py" ] || { echo "[smoke] ❌ 仓库路径不对: $R"; exit 1; }
PY=$O/../conda_envs/envs/diffusionopd/bin/python
SHIM=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad/opsd_shim
export PYTHONPATH="$SHIM:$R"
export ZIMAGE_HEAVY_BRIDGE=0 ZIMAGE_HEAVY_DIFF_BRIDGE=0 WANDB_MODE=offline
export REWARD_CKPT_PATH=$O/CA-OPD/code/DiffusionOPD-main/reward_ckpts
export PUBLIC_POLICY_WORLD_SIZE=4 PUBLIC_N_GPUS=4 PUBLIC_LAUNCH_WORLD_SIZE=4
# 复用已就位的数据与权重(仓库本身不含它们)
# 建链接并**校验**(初版的 `|| true` 把失败吞了,导致空跑)
ln -sfn "$O/CA-OPD/code/DiffusionOPSD-rpi/data/pickapic" "$R/data/pickapic"
ln -sfn "$(readlink -f "$O/CA-OPD/code/DiffusionOPSD-rpi/reward_ckpts")" "$R/reward_ckpts"
for need in "$R/data/pickapic" "$R/reward_ckpts"; do
  [ -e "$need" ] || { echo "[smoke] ❌ 链接未建成: $need"; exit 1; }
done
echo "[smoke] 数据与权重链接就位"
cd "$R" || exit 1
for spec in "canonical:" "msp:3,6"; do
  tag=${spec%%:*}; states=${spec#*:}
  OUT=$O/outputs/smoke_flowopsd_${tag}
  rm -rf "$OUT"; mkdir -p "$OUT"
  echo "===== [$tag] MSP_STATES='${states}'  $(date) ====="
  MSP_STATES="$states" OPSD_SEED=20260830 \
  "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
      scripts/train_opsd_ri_sd3.py \
      --config config/public.py:sd35_open3 \
      --config.num_epochs=2 --config.save_freq=1 --config.save_dir="$OUT" 2>&1 \
    | grep -avE "[0-9]+%\|" | tail -40
  echo "[$tag] exit=$? ckpt=$(ls -d $OUT/checkpoints/checkpoint-* 2>/dev/null | wc -l)"
done
