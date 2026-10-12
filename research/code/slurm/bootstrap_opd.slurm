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
#
# BootstrapOPD / Ephemeral Expert:交替执行「从当前 student 重训短程微专家」与「OPD 巩固」。
#
#   round 1 : 用**原始 teacher** 跑 E 个 OPD epoch(等价于 mopd@E,保证起点不吃亏)
#   round r>1: 用当前 student S 重训 3 个微专家 -> 以它们为 teacher 再跑 E 个 OPD epoch
#
# 总 OPD epoch 数 = R*E,取 4*200=800,与其它 arm 的预算完全一致,可直接对比 mopd@800。
# 提交: sbatch --job-name=bootopd --gres=gpu:h200:4 bootstrap_opd.slurm
set -uo pipefail
source /midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/env.sh
cd "$OPD_REPO"

NPROC=$(nvidia-smi -L | wc -l)
export OPD_NPROC=$NPROC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-32} / NPROC ))
# smoke 只验证链路,每卡显存与卡数无关(bs 固定),1/2/4 卡都行;正式跑必须 4 卡才与其它 arm 等价。
if [ "${OPD_SMOKE:-0}" = "1" ]; then
  case "$NPROC" in 1|2|4|8) ;; *) echo "[guard] nproc=$NPROC 不整除 8 -> requeue"; sleep 60; scontrol requeue "$SLURM_JOB_ID"; exit 0;; esac
else
  [ "$NPROC" -ne 4 ] && { echo "[guard] 需 4 卡实得 $NPROC -> requeue"; sleep 60; scontrol requeue "$SLURM_JOB_ID"; exit 0; }
fi
# OPD_SMOKE=1 时 config 会给 run_name 加 "[SMOKE] " 前缀,save_dir 随之改变;
# 这里必须用同一个前缀拼路径,否则 latest_ckpt 找不到刚训好的专家。
PFX=""; [ "${OPD_SMOKE:-0}" = "1" ] && PFX="[SMOKE] "

R=${BOOT_ROUNDS:-4}
E=${BOOT_OPD_EPOCHS:-200}
export EPH_ON=1
export EPH_EPOCHS=${EPH_EPOCHS:-25}
STATE="$OPD_ROOT/outputs/bootstrap_state${OPD_SMOKE:+_smoke}"; mkdir -p "$STATE"
BASE_PORT=$(( 29000 + SLURM_JOB_ID % 20000 ))
_p=0

echo "[boot] R=$R E=$E EPH_EPOCHS=$EPH_EPOCHS host=$(hostname) job=$SLURM_JOB_ID $(date)"

# 取某个 save_dir 下 step 最大的 checkpoint 的 lora 路径(run_name 含空格和方括号,全程加引号;
# 只有 * 参与 glob,引号内的 [ ] 是字面量 —— 这正是之前 auto-resume 踩过的坑)。
latest_ckpt() {
  local d="$1" best="" bn=-1 c n
  for c in "$d"/checkpoints/checkpoint-*; do
    [ -d "$c/lora" ] || continue
    n="${c##*-}"; case "$n" in ''|*[!0-9]*) continue ;; esac
    [ "$n" -gt "$bn" ] && { bn="$n"; best="$c/lora"; }
  done
  printf '%s' "$best"
}

launch() {  # $1=module  $2=config.py:name
  _p=$(( _p + 1 ))
  python -m accelerate.commands.launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_machines 1 --num_processes "$NPROC" \
    --main_process_port $(( BASE_PORT + _p )) \
    -m "$1" --config "$2"
}

S=""   # 当前 student 的 LoRA 路径;空 = 从 base SD3.5 开始
for (( r=1; r<=R; r++ )); do
  TAG="r$r"; export BOOT_TAG="$TAG"
  echo "===== [boot] round $TAG  student='${S:-<base>}'  $(date) ====="

  if [ "$r" -gt 1 ]; then
    [ -n "$S" ] || { echo "[boot] round $TAG 缺少 student,中止"; exit 1; }
    export CASCADE_PREV_LORA="$S"
    # aes/ocr 用 GRPO-Guard(CFG 4.5),geneval 用 NFT(CFG-free 1.0) —— 与各自 teacher 槽位的
    # guidance_scale 严格一致,否则 teacher 目标落在训练分布之外,会把想法测成假阴性。
    for spec in "aes:scripts.baselines.train_sd3_GRPO_Guard_multireward" \
                "ocr:scripts.baselines.train_sd3_GRPO_Guard_multireward" \
                "geneval:scripts.baselines.train_nft_sd3_multireward"; do
      k="${spec%%:*}"; mod="${spec##*:}"
      if [ -f "$STATE/done_${TAG}_${k}" ]; then echo "[boot] 跳过已完成的专家 ${TAG}/${k}"; continue; fi
      # 上一次中断留下的 checkpoint 会被 latest_ckpt 按最大步号选中 -> 这一轮拿到的是**旧的**
      # 专家权重(smoke 里就出现过 checkpoint-15 盖掉本轮的 checkpoint-1)。既然已确定要重训,先清干净。
      _edir="logs/${PFX}Bootstrap-${TAG}_${k}/checkpoints"
      [ -d "$_edir" ] && { echo "[boot] 清理旧 checkpoint: $_edir"; rm -rf "$_edir"; }
      echo "[boot] --- 训练微专家 ${TAG}/${k} (${mod##*.}) ---"
      launch "$mod" "config/baselines.py:eph_${k}" || { echo "[boot] 专家 ${TAG}/${k} 失败"; exit 1; }
      [ -n "$(latest_ckpt "logs/${PFX}Bootstrap-${TAG}_${k}")" ] || { echo "[boot] 专家 ${TAG}/${k} 没产出 checkpoint"; exit 1; }
      touch "$STATE/done_${TAG}_${k}"
    done
    export AES_TEACHER_PATH="$(latest_ckpt "logs/${PFX}Bootstrap-${TAG}_aes")"
    export OCR_TEACHER_PATH="$(latest_ckpt "logs/${PFX}Bootstrap-${TAG}_ocr")"
    export GENEVAL_TEACHER_PATH="$(latest_ckpt "logs/${PFX}Bootstrap-${TAG}_geneval")"
    for v in AES_TEACHER_PATH OCR_TEACHER_PATH GENEVAL_TEACHER_PATH; do
      [ -n "${!v}" ] || { echo "[boot] $v 为空,中止"; exit 1; }
      echo "[boot] $v = ${!v}"
    done
  else
    echo "[boot] round 1 使用**原始** teacher(env.sh 默认值),等价于 mopd@$E"
  fi

  ODIR="logs/${PFX}[MOPD-BOOTSTRAP-${TAG} + SD3.5-M + PickScore+OCR+GenEval]"
  if [ -f "$STATE/done_${TAG}_opd" ]; then
    echo "[boot] 跳过已完成的 OPD ${TAG}"
  else
    export OPD_STUDENT_INIT="$S"
    export OPD_NUM_EPOCHS="$E"
    export OPD_RESUME=auto           # 本轮内被 requeue 可从 train_state.pt 接上
    launch scripts.train_sd3_opd "config/opd.py:mopd_bootstrap" || { echo "[boot] OPD $TAG 失败"; exit 1; }
    touch "$STATE/done_${TAG}_opd"
  fi

  S="$(latest_ckpt "$ODIR")"
  [ -n "$S" ] || { echo "[boot] round $TAG 的 OPD 没产出 checkpoint,中止"; exit 1; }
  echo "[boot] round $TAG 完成 -> student = $S"
done
echo "[boot] 全部 $R 轮完成,最终 student = $S $(date)"
