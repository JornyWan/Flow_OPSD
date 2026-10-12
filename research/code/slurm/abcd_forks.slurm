#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=preempt_gpu
#SBATCH --qos=low
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --mem=300G
#SBATCH --gres=gpu:h100:4
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=abcd_forks
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
#
# §4.850 A→B→C→D 诊断的分叉:从 canonical ck300(o3canon300)各做**一次**真实更新。
# 同 ck300 权重 + 同 Adam 状态 + 同固定种子 ⇒ 同一批 prompt、同一 rollout;各 fork 只差 target 构造。
#   baseline = 官方 OPSD(所有 suffix 开关关闭)
#   qlook1   = 只把 oracle 换成 L1 一步前瞻(§4.844)
#   baseline 第二次(canonrep)= 零模型对照:测 GPU 非确定性带来的 fork 间差异
# 顺序执行,复用 ab_open3_v6.slurm 的全部守卫(preflight / nproc / 显存 / OUTPUT_DIR 锁)。
# 被抢占重排后:已完成的 fork(有 checkpoint-301/optimizer.pt)跳过;未完成的会从自身
# checkpoint-300(= ck300 原样)或 OPA_RESUME_DIR 重起,种子在启动时重设 ⇒ 结果不变。
set -uo pipefail
OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
# 源 checkpoint、fork 清单、卡数可由 env 覆盖(§4.853 在 qlook1 的 ck 上重做);默认 = §4.850 的原设置
CKSRC=${CKSRC:-$OPD_ROOT/outputs/ab_open3_baseline_s20260830_u300/checkpoints/checkpoint-300}
SPECS=${SPECS:-"baseline:20260830:abcd_canon_s20260830 qlook1:20260830:abcd_look1_s20260830 baseline:20260831:abcd_canon_s20260831 qlook1:20260831:abcd_look1_s20260831 baseline:20260830:abcd_canonrep_s20260830"}
FNP=${FNP:-4}
UPD=$(( $(python3 -c "import json;print(json.load(open('$CKSRC/trainer_state.json'))['global_step'])") + 1 ))
echo "[abcd] 源 $CKSRC → 各 fork 做 1 次更新到 checkpoint-$UPD,np=$FNP"
_RC=0
for spec in $SPECS; do
  IFS=: read -r arm seed name <<< "$spec"
  OUT=$OPD_ROOT/outputs/$name
  if [ -f "$OUT/checkpoints/checkpoint-$UPD/optimizer.pt" ]; then echo "[abcd] $name 已完成,跳过"; continue; fi
  # 被抢占后 requeue 的是**同一个 job id**:内层锁会把自己当成"别人在跑"而退出,先清掉自己的旧锁。
  _LK=/midtier/minilab/scratch/chw4026/tmp/ablock_$name
  [ "$(cat "$_LK/jobid" 2>/dev/null)" = "$SLURM_JOB_ID" ] && rm -rf "$_LK"
  echo "[abcd] ===== $name (arm=$arm seed=$seed) $(date)"
  ARM=$arm SEED=$seed UPDATES=$UPD OPA_RESUME_DIR=$CKSRC OUTPUT_DIR=$OUT EXPECT_NPROC=$FNP \
    bash $OPD_ROOT/slurm/ab_open3_v6.slurm
  rc=$?
  [ -f "$OUT/checkpoints/checkpoint-$UPD/optimizer.pt" ] || { echo "[abcd] ❌ $name 没有产出 checkpoint-$UPD (rc=$rc)"; _RC=1; }
done
echo "[abcd] done rc=$_RC $(date)"
exit $_RC
