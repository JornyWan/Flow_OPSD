#!/bin/bash
#SBATCH --job-name=cdx0002cr
#SBATCH --account=minilab
#SBATCH --partition=preempt_gpu
#SBATCH --qos=low
#SBATCH --nodelist=ai-gpu14
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/codex_research/runs/0002_cr_smoke/slurm_%j.out

set -euo pipefail

OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
CODEX_ROOT="$OPD_ROOT/codex_research"
BASE_REPO="$OPD_ROOT/CA-OPD/code/DiffusionOPD-main"
PATCH_FILE="$CODEX_ROOT/code/0002_constrained_retention.patch"
RUN_ROOT="$CODEX_ROOT/runs/0002_cr_smoke"
ARTIFACT_DIR="$RUN_ROOT/artifacts/job_${SLURM_JOB_ID}"

# env.sh checks WANDB_FORCE_OFFLINE while constructing WANDB_MODE.
export WANDB_FORCE_OFFLINE=1
source "$OPD_ROOT/env.sh"
export OPD_NUM_EPOCHS=3
export OPD_EVAL_FREQ=0
export OPD_RESUME=none
export OPD_STUDENT_INIT="$OPD_ROOT/models/DiffusionOPD/Student/lora"
export CODEX_CR_RUN_DIR="$ARTIFACT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NPROC=$(nvidia-smi -L | wc -l)
if [ "$NPROC" -ne 2 ]; then
    echo "[CODEX-0002] expected exactly 2 visible GPUs, got $NPROC"
    exit 40
fi
export OPD_NPROC="$NPROC"
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-32} / NPROC ))
export MASTER_PORT=$(( 29000 + SLURM_JOB_ID % 20000 ))

mkdir -p "$ARTIFACT_DIR"
WORK_DIR=$(mktemp -d "$TMPDIR/codex0002-${SLURM_JOB_ID}.XXXXXX")
case "$WORK_DIR" in
    "$TMPDIR"/codex0002-"${SLURM_JOB_ID}".*) ;;
    *) echo "[CODEX-0002] refusing unexpected temporary path: $WORK_DIR"; exit 41 ;;
esac
cleanup() {
    rm -rf -- "$WORK_DIR"
}
trap cleanup EXIT

echo "[CODEX-0002] job=$SLURM_JOB_ID host=$(hostname) start=$(date --iso-8601=seconds)"
echo "[CODEX-0002] base=$BASE_REPO patch=$PATCH_FILE work=$WORK_DIR artifacts=$ARTIFACT_DIR"
echo "[CODEX-0002] visible_gpus=$NPROC CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L

# The overlay is pinned to the audited source.  If Claude or anyone else edits
# either file, fail closed and require a newly reviewed patch instead of fuzzily
# applying old research logic to new code.
EXPECTED_CONFIG_SHA=f55f19c0351ed8bf56117fc925a8419b461828a87260ae2729c2cdd3880db36f
EXPECTED_TRAINER_SHA=0f50637504eee01bfcfc1aaa306442d44fa77b00f5e1e284563650aee604d015
ACTUAL_CONFIG_SHA=$(sha256sum "$BASE_REPO/config/opd.py" | awk '{print $1}')
ACTUAL_TRAINER_SHA=$(sha256sum "$BASE_REPO/scripts/train_sd3_opd.py" | awk '{print $1}')
if [ "$ACTUAL_CONFIG_SHA" != "$EXPECTED_CONFIG_SHA" ] || [ "$ACTUAL_TRAINER_SHA" != "$EXPECTED_TRAINER_SHA" ]; then
    echo "[CODEX-0002] source hash mismatch; refusing to run"
    echo "[CODEX-0002] config expected=$EXPECTED_CONFIG_SHA actual=$ACTUAL_CONFIG_SHA"
    echo "[CODEX-0002] trainer expected=$EXPECTED_TRAINER_SHA actual=$ACTUAL_TRAINER_SHA"
    exit 42
fi

mkdir -p "$WORK_DIR/config" "$WORK_DIR/scripts" "$WORK_DIR/flow_grpo"
cp -a "$BASE_REPO/config/." "$WORK_DIR/config/"
cp -a "$BASE_REPO/scripts/." "$WORK_DIR/scripts/"
cp -a "$BASE_REPO/flow_grpo/." "$WORK_DIR/flow_grpo/"
cp -a "$BASE_REPO/setup.py" "$WORK_DIR/setup.py"

patch --dry-run -d "$WORK_DIR" -p1 < "$PATCH_FILE"
patch -d "$WORK_DIR" -p1 < "$PATCH_FILE"
python -m py_compile "$WORK_DIR/config/opd.py" "$WORK_DIR/scripts/train_sd3_opd.py"
python "$CODEX_ROOT/code/0002_test_constraint_math.py"

export PYTHONPATH="$WORK_DIR:$BASE_REPO:${PYTHONPATH:-}"
cd "$WORK_DIR"

# Config-level fail-closed checks.  With two GPUs, nbpe=12 and total global
# samples/epoch remains 3 * 2 * 12 = 72, exactly matching the upstream 8-GPU run.
python - <<'PY'
import importlib.util
import os

spec = importlib.util.spec_from_file_location("codex0002_config", "config/opd.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
config = module.get_config("codex_cr_smoke")
assert [x["name"] for x in config.train.teachers] == ["pickscore", "ocr", "geneval"]
assert config.sample.num_batches_per_epoch == 12
assert config.train.gradient_accumulation_steps == 12
assert config.sample.train_batch_size * int(os.environ["OPD_NPROC"]) * config.sample.num_batches_per_epoch == 72
assert config.num_epochs == 3 and config.eval_freq == 0 and config.save_freq == 1
assert config.train.lora_path == os.environ["OPD_STUDENT_INIT"]
assert config.train.cr_reward_teacher_idx == 0
assert list(config.train.cr_retention_teacher_indices) == [1, 2]
assert all(not x["proxy_reward"] for x in config.train.teachers)
print("[CODEX-0002] config invariants: PASS")
print("[CODEX-0002] effective: bs=3 nproc=2 nbpe=12 samples_per_epoch=72 epochs=3")
PY

sha256sum "$BASE_REPO/config/opd.py" "$BASE_REPO/scripts/train_sd3_opd.py" "$PATCH_FILE"
echo "[CODEX-0002] launch: accelerate nproc=$NPROC port=$MASTER_PORT config=codex_cr_smoke"

set +e
python -m accelerate.commands.launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_machines 1 --num_processes "$NPROC" --main_process_port "$MASTER_PORT" \
    -m scripts.train_sd3_opd --config config/opd.py:codex_cr_smoke
RC=$?
set -e
echo "[CODEX-0002] exit=$RC end=$(date --iso-8601=seconds) artifacts=$ARTIFACT_DIR"
exit "$RC"
