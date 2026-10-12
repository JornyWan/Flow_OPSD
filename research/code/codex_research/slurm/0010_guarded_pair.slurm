#!/bin/bash
#SBATCH --job-name=${GJOB:-guarded}
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=20:00:00
#SBATCH --open-mode=append
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/codex_research/runs/0010_guarded_pair/slurm_%x_%j.out

set -euo pipefail

OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
CODEX_ROOT="$OPD_ROOT/codex_research"
BASE_REPO="$OPD_ROOT/CA-OPD/code/DiffusionOPD-main"
METHOD_PATCH="$CODEX_ROOT/code/0002_constrained_retention.patch"
FIX_PATCH="$CODEX_ROOT/code/0003_fix_snr_init_order.patch"
RUN_ROOT="$CODEX_ROOT/runs/0010_guarded_pair/${GJOB:-guarded}"
ARTIFACT_DIR="$RUN_ROOT/artifacts/job_${SLURM_JOB_ID}"


source "$OPD_ROOT/env.sh"
export OPD_NUM_EPOCHS=${GEPOCHS:-200}
export OPD_EVAL_FREQ=0
export OPD_RESUME=none
export OPD_STUDENT_INIT="$OPD_ROOT/models/DiffusionOPD/Student/lora"
export CODEX_CR_RUN_DIR="$ARTIFACT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NPROC=$(nvidia-smi -L | wc -l)
# 必须恰好 2 卡,不能放宽:
#  - nbpe = 24/nproc,而 codex_cr_smoke 有 3 个 teacher,要求 nbpe % 3 == 0
#    -> 2卡 nbpe=12 ✓ | 3卡 nbpe=8 ✗(8%3!=0,秒退 AssertionError) | 4卡 nbpe=6 ✓
#  - 下方不变量还硬性断言 nbpe==12 且 gas==12,只有 2 卡满足。
# 我曾把它放宽到 2~4 并用 3 卡提交(job 1035930/1035931),那是错的。
# 允许 1/2/4 卡:nbpe=24/nproc 分别为 24/12/6,都能被 3 个 teacher 整除,
# 且 bs*nproc*nbpe 恒为 72 -> 三者训练等效。3 卡不行(nbpe=8, 8%3!=0)。
case "$NPROC" in
    1|2|4) ;;
    *) echo "[CODEX-0010] expected 1, 2 or 4 visible GPUs, got $NPROC (3 卡 nbpe=8 不能被 3 个 teacher 整除)"; exit 40 ;;
esac
export OPD_NPROC="$NPROC"
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-32} / NPROC ))
export MASTER_PORT=$(( 29000 + SLURM_JOB_ID % 20000 ))

mkdir -p "$ARTIFACT_DIR"
WORK_DIR=$(mktemp -d "$TMPDIR/codex0003-${SLURM_JOB_ID}.XXXXXX")
case "$WORK_DIR" in
    "$TMPDIR"/codex0003-"${SLURM_JOB_ID}".*) ;;
    *) echo "[CODEX-0003] refusing unexpected temporary path: $WORK_DIR"; exit 41 ;;
esac
cleanup() {
    rm -rf -- "$WORK_DIR"
}
trap cleanup EXIT

echo "[CODEX-0003] job=$SLURM_JOB_ID host=$(hostname) start=$(date --iso-8601=seconds)"
echo "[CODEX-0003] base=$BASE_REPO method_patch=$METHOD_PATCH fix_patch=$FIX_PATCH"
echo "[CODEX-0003] work=$WORK_DIR artifacts=$ARTIFACT_DIR"
echo "[CODEX-0003] visible_gpus=$NPROC CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L

EXPECTED_CONFIG_SHA=f55f19c0351ed8bf56117fc925a8419b461828a87260ae2729c2cdd3880db36f
EXPECTED_TRAINER_SHA=0f50637504eee01bfcfc1aaa306442d44fa77b00f5e1e284563650aee604d015
EXPECTED_METHOD_TRAINER_SHA=8d8e7552c929cde22307f48b0d38f92f6d329e77d475a1506a57092cd555dc0e
ACTUAL_CONFIG_SHA=$(sha256sum "$BASE_REPO/config/opd.py" | awk '{print $1}')
# 基底 trainer 用**pin 住的副本**,不是 live 文件:live 的 scripts/train_sd3_opd.py 已被
# CODEX-0009(逐 teacher KL 日志)与 CODEX-0007(branch_steps_direct 计数)修改,hash 必然失配。
# 这两处改动服务于 RTC 主线,与 guarded 实验无关;pin 副本与 codex smoke 验证过的版本逐字节相同,
# 故 EXPECTED_* 三个 hash 全部原样保留 —— 不去迁就改动而放宽校验。
PIN_TRAINER="$CODEX_ROOT/code/0002_pinned_base_train_sd3_opd.py"
ACTUAL_TRAINER_SHA=$(sha256sum "$PIN_TRAINER" | awk '{print $1}')
if [ "$ACTUAL_CONFIG_SHA" != "$EXPECTED_CONFIG_SHA" ] || [ "$ACTUAL_TRAINER_SHA" != "$EXPECTED_TRAINER_SHA" ]; then
    echo "[CODEX-0003] source hash mismatch; refusing to run"
    echo "[CODEX-0003] config expected=$EXPECTED_CONFIG_SHA actual=$ACTUAL_CONFIG_SHA"
    echo "[CODEX-0003] trainer expected=$EXPECTED_TRAINER_SHA actual=$ACTUAL_TRAINER_SHA"
    exit 42
fi

mkdir -p "$WORK_DIR/config" "$WORK_DIR/scripts" "$WORK_DIR/flow_grpo"
cp -a "$BASE_REPO/config/." "$WORK_DIR/config/"
cp -a "$BASE_REPO/scripts/." "$WORK_DIR/scripts/"
cp -a "$PIN_TRAINER" "$WORK_DIR/scripts/train_sd3_opd.py"
cp -a "$BASE_REPO/flow_grpo/." "$WORK_DIR/flow_grpo/"
cp -a "$BASE_REPO/setup.py" "$WORK_DIR/setup.py"

patch --dry-run -d "$WORK_DIR" -p1 < "$METHOD_PATCH"
patch -d "$WORK_DIR" -p1 < "$METHOD_PATCH"
METHOD_TRAINER_SHA=$(sha256sum "$WORK_DIR/scripts/train_sd3_opd.py" | awk '{print $1}')
if [ "$METHOD_TRAINER_SHA" != "$EXPECTED_METHOD_TRAINER_SHA" ]; then
    echo "[CODEX-0003] 0002 overlay produced unexpected trainer hash: $METHOD_TRAINER_SHA"
    exit 43
fi
patch --dry-run -d "$WORK_DIR" -p1 < "$FIX_PATCH"
patch -d "$WORK_DIR" -p1 < "$FIX_PATCH"

python -m py_compile "$WORK_DIR/config/opd.py" "$WORK_DIR/scripts/train_sd3_opd.py"
python "$CODEX_ROOT/code/0002_test_constraint_math.py"

export PYTHONPATH="$WORK_DIR:$BASE_REPO:${PYTHONPATH:-}"
cd "$WORK_DIR"

python - <<'PY'
import importlib.util
import os

spec = importlib.util.spec_from_file_location("codex0003_config", "config/opd.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
config = module.get_config("codex_cr_smoke")
assert [x["name"] for x in config.train.teachers] == ["pickscore", "ocr", "geneval"]
# 原断言写死 nbpe==12/gas==12,那只是 2 卡时的取值,不是方法不变量。
# 真正必须成立的两条:每 epoch 恰好 72 个样本,且 nbpe 能被 teacher 数整除(否则 teacher 路由不均)。
_nproc = int(os.environ["OPD_NPROC"])
_nbpe = config.sample.num_batches_per_epoch
assert config.sample.train_batch_size * _nproc * _nbpe == 72, (config.sample.train_batch_size, _nproc, _nbpe)
assert _nbpe % len(config.train.teachers) == 0, (_nbpe, len(config.train.teachers))
assert config.train.gradient_accumulation_steps == _nbpe, (config.train.gradient_accumulation_steps, _nbpe)
# smoke 原本硬断言 num_epochs==3;0010 要跑 GEPOCHS(=200)个 epoch,
# 故改为校验"配置确实收到了所请求的 epoch 数",而不是删掉这条不变量。
assert config.num_epochs == int(os.environ["OPD_NUM_EPOCHS"]), config.num_epochs
assert config.eval_freq == 0 and config.save_freq == 1
assert config.train.lora_path == os.environ["OPD_STUDENT_INIT"]
assert config.train.cr_reward_teacher_idx == 0
assert list(config.train.cr_retention_teacher_indices) == [1, 2]
assert all(not x["proxy_reward"] for x in config.train.teachers)
print("[CODEX-0003] config invariants: PASS")
print(f"[CODEX-0003] effective: bs={config.sample.train_batch_size} nproc={_nproc} nbpe={_nbpe} "
      f"gas={config.train.gradient_accumulation_steps} samples_per_epoch=72 epochs={config.num_epochs}")
PY

sha256sum "$BASE_REPO/config/opd.py" "$BASE_REPO/scripts/train_sd3_opd.py" "$METHOD_PATCH" "$FIX_PATCH"
sha256sum "$WORK_DIR/config/opd.py" "$WORK_DIR/scripts/train_sd3_opd.py"
echo "[CODEX-0003] launch: accelerate nproc=$NPROC port=$MASTER_PORT config=codex_cr_smoke"

set +e
python -m accelerate.commands.launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_machines 1 --num_processes "$NPROC" --main_process_port "$MASTER_PORT" \
    -m scripts.train_sd3_opd --config config/opd.py:codex_cr_smoke
RC=$?
set -e
echo "[CODEX-0003] exit=$RC end=$(date --iso-8601=seconds) artifacts=$ARTIFACT_DIR"
exit "$RC"
