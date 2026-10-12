#!/bin/bash
#SBATCH --job-name=smokecfg
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=1:30:00
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
#SBATCH --exclude=ai-gpu01,ai-gpu03   # 泄漏 GPU 进程(2.01/1.48GiB),实测导致 gcprobe OOM
# 用微型数据集快速验证一个配置能跑通(排队前的把关)。用法: RUN_CFG=xxx sbatch smoke_cfg.slurm
set -uo pipefail
RUN_CFG=${RUN_CFG:?需要 RUN_CFG}
source /midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/env.sh
cd "$OPD_REPO"
NPROC=$(nvidia-smi -L | wc -l)
# _scale_to_nproc 要求 nproc 整除 8。冒烟只验证代码路径,每卡显存占用与卡数无关(bs=3/卡),
# 所以 1/2/4 卡都行 —— 比强求 4 卡好排队得多。只有拿到 3 卡这种不整除的才 requeue。
case "$NPROC" in 1|2|4|8) ;; *) echo "[guard] nproc=$NPROC 不整除 8 -> requeue"; sleep 60; scontrol requeue "$SLURM_JOB_ID"; exit 0;; esac
export OPD_NPROC=$NPROC OPD_SMOKE=1 OPD_SMOKE_EPOCHS=${OPD_SMOKE_EPOCHS:-3} OPD_RESUME=none
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$(( 32 / NPROC ))
export MASTER_PORT=$(( 29000 + SLURM_JOB_ID % 20000 ))

# --- NFS 属性缓存自愈预检 ---
# 登录节点刚改完 config 就 sbatch,计算节点可能仍读到缓存里的旧文件。这里不是"等一等",
# 而是**直接校验我们依赖的那条性质**:smoke 下 eval_freq 必须是 0(否则会在 epoch 0 跑一次
# 最贵的在线 eval,在 48G 的 A40 上必 OOM)。不满足就强制刷新属性缓存后重试。
for _try in $(seq 1 12); do
  _ef=$(python -c "
import importlib.util,os
spec=importlib.util.spec_from_file_location('c','config/opd.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(m.get_config('$RUN_CFG').eval_freq)" 2>/dev/null)
  if [ "$_ef" = "0" ]; then
    [ "$_try" -gt 1 ] && echo "[preflight] 第 $_try 次尝试后 config 已刷新 (eval_freq=0)"
    break
  fi
  echo "[preflight] eval_freq='$_ef' 期望 0 -> config/opd.py 仍是 NFS 缓存的旧版,10s 后重试 ($_try/12)"
  ls -la config/opd.py scripts/train_sd3_opd.py >/dev/null 2>&1   # 强制刷新属性缓存
  sleep 10
done

# --- 预检: 等 NFS 属性缓存刷新,确认本节点能看到该配置 ---
# 登录节点刚改完 config 就 sbatch 会有竞态:计算节点的 NFS client 可能缓存了旧的文件属性,
# 表现为 KeyError: '<新配置名>'。这里最多重试 12 次(共 2 分钟),自愈而不是靠人工 sleep。
for _try in $(seq 1 12); do
  if python -c "
import importlib.util,sys
spec=importlib.util.spec_from_file_location('c','config/opd.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sys.exit(0 if '$RUN_CFG' in dir(m) else 1)" 2>/dev/null; then
    [ "$_try" -gt 1 ] && echo "[preflight] 第 $_try 次尝试后配置可见"
    break
  fi
  echo "[preflight] 配置 $RUN_CFG 尚不可见(NFS 缓存),10s 后重试 ($_try/12)"
  ls -la config/opd.py >/dev/null 2>&1   # 强制刷新属性缓存
  sleep 10
done


# --- 训练脚本新鲜度校验 ---
# 之前的预检只看 config/opd.py,不看 scripts/train_sd3_opd.py,于是"改完立刻提交"时
# 计算节点可能读到旧版训练脚本并顺利通过预检(实测编辑与作业启动只差 6 秒就中招)。
# RUN_REQUIRE_MARKER 传一段本次改动必然包含的字符串,轮询到出现为止。
if [ -n "${RUN_REQUIRE_MARKER:-}" ]; then
  for _t in $(seq 1 18); do
    if grep -qF "$RUN_REQUIRE_MARKER" scripts/train_sd3_opd.py 2>/dev/null; then
      [ "$_t" -gt 1 ] && echo "[preflight] 第 $_t 次尝试后训练脚本已刷新"
      break
    fi
    echo "[preflight] 训练脚本尚未包含标记(NFS 缓存),10s 后重试 ($_t/18)"
    ls -la scripts/train_sd3_opd.py >/dev/null 2>&1
    sleep 10
    [ "$_t" -eq 18 ] && { echo "[preflight] 训练脚本始终未刷新,放弃"; exit 1; }
  done
fi

echo "[smoke] cfg=$RUN_CFG nproc=$NPROC host=$(hostname) $(date)"
# 只删**本次配置**的 smoke 目录 —— 通配 "logs/[SMOKE]"* 会删掉并行 smoke 作业的产物
# (实测: RTC-RAWB 跑完 100 epoch 的 checkpoint 被后启动的 smoke 整个删除)。
_SMK_DIR=$(python -c "
import importlib.util
spec=importlib.util.spec_from_file_location('c','config/opd.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(m.get_config('$RUN_CFG').save_dir)" 2>/dev/null)
if [ -n "$_SMK_DIR" ] && [ "${_SMK_DIR#logs/\[SMOKE\]}" != "$_SMK_DIR" ]; then
  rm -rf "$_SMK_DIR" 2>/dev/null; echo "[smoke] 已清理本次目录: $_SMK_DIR"
else
  echo "[smoke] 跳过清理: ${_SMK_DIR:-<空>}"
fi
python -m accelerate.commands.launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_machines 1 --num_processes $NPROC --main_process_port $MASTER_PORT \
  -m scripts.train_sd3_opd --config config/opd.py:$RUN_CFG
_RC=$?
echo "[smoke] cfg=$RUN_CFG exit=$_RC $(date)"
# 必须把训练的退出码传给 SLURM: 否则脚本以 echo 的 0 结束,
# OOM/崩溃的作业在 sacct 里显示为 COMPLETED(实测 mech_a40 OOM 却报 COMPLETED)。
exit $_RC
