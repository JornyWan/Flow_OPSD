#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
#SBATCH --exclude=ai-gpu01,ai-gpu03   # 泄漏 GPU 进程(2.01/1.48GiB),实测导致 gcprobe OOM
#
# 单个实验 arm:单节点 4 卡。
# 上游配置按 8 卡写(bs3 x 8 x 3batch = 72 samples/epoch);这里 OPD_NPROC=4 让 config/opd.py 的
# _scale_to_nproc 把 num_batches_per_epoch 翻倍(3->6),得到 4 x 3 x 6 = 72,梯度与 8 卡严格等价。
# 提交: RUN_CFG=mopd_caopd_tau02 sbatch --job-name=tau02 --gres=gpu:h200:4 opd_arm.slurm
set -uo pipefail
RUN_CFG=${RUN_CFG:?必须指定 RUN_CFG}
source /midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/env.sh
mkdir -p "$OPD_ROOT/outputs"
cd "$OPD_REPO"          # config 用 os.getcwd() 拼 dataset 路径

NPROC=$(nvidia-smi -L | wc -l)
export OPD_NPROC=$NPROC          # 驱动 config 里的 nbpe 缩放
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-32} / NPROC ))
export OPD_RESUME=auto           # 超时/抢占 requeue 后从最新 train_state.pt 接上
export OPD_EVAL_FREQ=${OPD_EVAL_FREQ:-150}   # 在线 eval 纯观测且占 ~60% 墙钟;save_freq 不变
export MASTER_PORT=$(( 29000 + SLURM_JOB_ID % 20000 ))   # 同节点多作业并行时避免端口冲突

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


echo "[arm] cfg=$RUN_CFG nproc=$NPROC host=$(hostname) job=$SLURM_JOB_ID port=$MASTER_PORT $(date)"
nvidia-smi -L
# 2/3/4 卡都能保持 samples/epoch=72 且 micro-batch 总数=24(见 config/opd.py 的 _scale_to_nproc),
# 故三者梯度等价、结果可互相比较。只有落到 1 卡或 >4 才需要 requeue。
# 早先硬性要求 4 卡,在 SLURM 只给 3 卡时会无限重排(实测重排 2 次仍是 3 卡,
# 因为另一用户占着该节点第 4 张卡),而 --exclusive 的预计启动时间是 3 天后。
# OPD_ALLOW_1GPU=1 时允许单卡:实测单 teacher 配置在 1 卡下 nbpe=24,
# samples/epoch=72 且 micro-batch 总数=24 —— 与 2/4 卡**完全等效**(guard 自己的判据即此)。
# 不默认放开,是因为 SLURM 会降配(申请 4 卡给 1 卡),其他臂会因此静默慢 4 倍而无人察觉;
# 真正的兼容性保护是下面那个"用实际 NPROC 载入配置"的检查,与卡数范围无关。
_MIN_GPU=2; [ "${OPD_ALLOW_1GPU:-0}" = "1" ] && _MIN_GPU=1
if [ "$NPROC" -lt "$_MIN_GPU" ] || [ "$NPROC" -gt 4 ]; then
  _RS=${SLURM_RESTART_COUNT:-0}
  echo "[guard] 需 ${_MIN_GPU}~4 卡实得 $NPROC -> requeue (累计 $_RS 次)"
  sleep 900
  scontrol requeue "$SLURM_JOB_ID"; exit 0
fi
[ "$NPROC" -ne 4 ] && echo "[guard] 实得 $NPROC 卡(非 4);config 会把 nbpe 调为 $((24/NPROC)),samples/epoch 仍为 72,与 4 卡梯度等价"
# 配置兼容性检查(比卡数范围更可靠):
# 多 teacher 臂要求 nbpe 能被 teacher 数整除 —— 3 teacher + 3 卡 -> nbpe=8, 8%3!=0 -> 断言失败。
# 早先把 guard 从"必须 4 卡"放宽到 2~4 卡,对单 teacher 的 RTC 臂成立,但**过度泛化**到了多 teacher 臂,
# 实测 horizon2/ste/retain/poe125 四个臂在 3 卡下秒退(AssertionError)。
# 不硬编码 teacher 数,直接用实际 NPROC 载入配置:载不动就换卡数重排。
if ! python -c "
import os,importlib.util
os.environ['OPD_NPROC']='$NPROC'
spec=importlib.util.spec_from_file_location('c','config/opd.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.get_config('$RUN_CFG')" 2>/dev/null; then
  _RS=${SLURM_RESTART_COUNT:-0}
  if [ "$_RS" -ge 5 ]; then
    echo "[guard] nproc=$NPROC 与 $RUN_CFG 不兼容,已重启 $_RS 次 -> 放弃(请改 --gres 卡数)"; exit 1
  fi
  echo "[guard] nproc=$NPROC 与配置 $RUN_CFG 不兼容(如 nbpe 不能被 teacher 数整除) -> requeue (第 $_RS 次)"
  sleep 300; scontrol requeue "$SLURM_JOB_ID"; exit 0
fi
python -c "
import os,importlib.util
spec=importlib.util.spec_from_file_location('c','config/opd.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c=m.get_config('$RUN_CFG')
print(f'[arm] 有效配置: bs={c.sample.train_batch_size} nbpe={c.sample.num_batches_per_epoch} gas={c.train.gradient_accumulation_steps} -> samples/epoch={c.sample.train_batch_size*$NPROC*c.sample.num_batches_per_epoch} (目标 72)')
"

python -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_machines 1 --num_processes $NPROC --main_process_port $MASTER_PORT \
  -m scripts.train_sd3_opd --config config/opd.py:$RUN_CFG
_RC=$?
echo "[arm] cfg=$RUN_CFG exit=$_RC $(date)"
# 把训练退出码传给 SLURM,否则崩溃的 arm 在 sacct 里显示 COMPLETED(实测 smoke 已踩)。
# 链路用 afterany,故非零退出不会卡住后续作业;auto_eval 监视也只看作业是否离开队列。
exit $_RC
