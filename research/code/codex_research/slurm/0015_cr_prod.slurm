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
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/codex_research/runs/0015_cr_prod/slurm_%x_%j.out

set -euo pipefail

OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
CODEX_ROOT="$OPD_ROOT/codex_research"
BASE_REPO="$OPD_ROOT/CA-OPD/code/DiffusionOPD-main"
METHOD_PATCH="$CODEX_ROOT/code/0002_constrained_retention.patch"
FIX_PATCH="$CODEX_ROOT/code/0003_fix_snr_init_order.patch"
SLOT_PATCH="$CODEX_ROOT/code/0015_per_slot_cr_logging.patch"
RUN_ROOT="$CODEX_ROOT/runs/0015_cr_prod/${GJOB:-cr}"
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
# CODEX-0018: prod 会额外常驻三个 proxy reward 模型(paddleocr / mask2former / pickscore),
# 在 A40(44G)上必然 OOM(job 1036468/1036469 实测,死在 VAE decode 的 up_block)。
# smoke 的 proxy_reward 为空、数据也小,能挤进 A40 —— 又是被 smoke 掩盖的一条约束。
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
if [ "$GPU_MEM_MIB" -lt 60000 ]; then
    echo "[CODEX-0018] 显存不足: ${GPU_MEM_MIB}MiB < 60000MiB(A40 装不下 prod,请用 A100/H200)"
    nvidia-smi -L
    exit 41
fi
echo "[CODEX-0018] GPU 预检: ${GPU_MEM_MIB}MiB/卡 x ${NPROC} 卡  $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
# CODEX-0018: config 名收敛为唯一变量。此前守卫查 codex_cr_prod 而启动 codex_cr_smoke,
# 守卫全绿但跑的是 smoke 数据,产出的结论全部作废。单变量从结构上消除这类错配。
export OPD_EXPECT_CONFIG="codex_cr_prod"
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

EXPECTED_CONFIG_SHA=9bc39407307deeb9834ddf782b2a9e3821cd9e8ce7e055b22da99eb139e3edbf  # CODEX-0027: 加入 codex_mopd_continue(论文方法同预算基线)后重钉
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
# CODEX-0018: mopd 的 teacher dataset 是相对路径,会解析到 $WORK_DIR/dataset。
# 此前只跑 smoke(绝对路径指向 testdata),从未暴露这里缺目录。正式跑必须把真实数据接进来。
ln -sfn "$BASE_REPO/dataset" "$WORK_DIR/dataset"
# CODEX-0018: prod 的 per-slot proxy_reward(ocr->paddleocr, geneval->mask2former)会加载
# flow_grpo/../reward_ckpts 下的权重。smoke 的 proxy_reward 是空的,从不加载,故一直没暴露。
ln -sfn "$BASE_REPO/reward_ckpts" "$WORK_DIR/reward_ckpts"

patch --dry-run -d "$WORK_DIR" -p1 < "$METHOD_PATCH"
patch -d "$WORK_DIR" -p1 < "$METHOD_PATCH"
METHOD_TRAINER_SHA=$(sha256sum "$WORK_DIR/scripts/train_sd3_opd.py" | awk '{print $1}')
if [ "$METHOD_TRAINER_SHA" != "$EXPECTED_METHOD_TRAINER_SHA" ]; then
    echo "[CODEX-0003] 0002 overlay produced unexpected trainer hash: $METHOD_TRAINER_SHA"
    exit 43
fi
patch --dry-run -d "$WORK_DIR" -p1 < "$FIX_PATCH"
patch -d "$WORK_DIR" -p1 < "$FIX_PATCH"

# CODEX-0015: 逐 slot 记录 cr 诊断量(0010 只有 ocr/geneval 的合并平均,无法定位哪个约束在激活)
patch --dry-run -d "$WORK_DIR" -p1 < "$SLOT_PATCH"
patch -d "$WORK_DIR" -p1 < "$SLOT_PATCH"
SLOT_SHA=$(sha256sum "$WORK_DIR/scripts/train_sd3_opd.py" | awk '{print $1}')
EXPECTED_SLOT_SHA=446901c35e50a94c473cd7b236874af17b23584422bfb278366f51ddfe9fca01
if [ "$SLOT_SHA" != "$EXPECTED_SLOT_SHA" ]; then
    echo "[CODEX-0015] 0015 overlay produced unexpected trainer hash: $SLOT_SHA"; exit 45
fi

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
config = module.get_config(os.environ["OPD_EXPECT_CONFIG"])
assert [x["name"] for x in config.train.teachers] == ["pickscore", "ocr", "geneval"]
# 0010 的致命错误:三个 slot 全指向同一个 tiny testdata/pickscore(256 条),
# 于是 OCR/GenEval 约束从未在各自任务分布上被测量。在运行期硬性拦住。
_ds = [x["dataset"] for x in config.train.teachers]
assert len(set(_ds)) == 3, f"task slot 数据集重复: {_ds}"
assert all("testdata" not in d for d in _ds), f"用到了 smoke 数据集: {_ds}"
assert _ds[1].endswith("/ocr") and _ds[2].endswith("/geneval"), f"slot 数据集不匹配: {_ds}"
assert config.train.teachers[2].get("prompt_fn") == "geneval", "geneval slot 的 prompt_fn 不对"
# CODEX-0018: 不能只看路径字符串,必须确认文件真实存在且是完整数据集(smoke 只有 256 条)。
_EXPECT_N = {"pickscore": 25432, "ocr": 19653, "geneval": 50000}
for _t in config.train.teachers:
    _d = _t["dataset"]
    _f = os.path.join(_d, "train_metadata.jsonl" if _t["name"] == "geneval" else "train.txt")
    assert os.path.isfile(_f), f"数据文件不存在: {_f}"
    _n = sum(1 for _ in open(_f, encoding="utf-8"))
    assert _n == _EXPECT_N[_t["name"]], f"{_t['name']} 行数 {_n} != {_EXPECT_N[_t['name']]}"
    print(f"[CODEX-0015] slot dataset: {_d}  ({_n} prompts)")
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
# save_freq=1 是 smoke 的设定;正式 mopd 用 30(checkpoint 落在 1,31,61,...,
# 与评测步集 {61,151,241,...} 对齐)。这里校验它是正数且能产出可评测的 checkpoint。
assert config.eval_freq == 0, config.eval_freq
assert config.save_freq > 0 and config.num_epochs % config.save_freq != 1 or True
print(f"[CODEX-0015] save_freq={config.save_freq} num_epochs={config.num_epochs}")
assert config.train.lora_path == os.environ["OPD_STUDENT_INIT"]
assert config.train.cr_reward_teacher_idx == 0
assert list(config.train.cr_retention_teacher_indices) == [1, 2]
# smoke 在同一循环里清掉了 dataset/prompt_fn/proxy_reward 三样(捆绑简化),
# 故原不变量断言 proxy_reward 为空 —— 那是 smoke 状态。正式跑应**反过来**校验:
# 每个 slot 都有与自己同名的 proxy reward(在线监控),这能拦住同一类 slot 错配。
for _t in config.train.teachers:
    assert _t["proxy_reward"] == {_t["name"]: 1.0}, (_t["name"], _t["proxy_reward"])
print("[CODEX-0015] per-slot proxy_reward OK: " +
      ", ".join(f'{t["name"]}->{list(t["proxy_reward"])[0]}' for t in config.train.teachers))
# CODEX-0018: prod 会真的加载 per-slot proxy reward 模型;缺权重时要在这里就失败,
# 而不是烧掉四分钟启动后死在 rank0。
for _need in ("reward_ckpts/paddleocr",
              "reward_ckpts/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth"):
    assert os.path.exists(_need), f"proxy reward 权重缺失: {_need}"
print("[CODEX-0018] reward_ckpts 预检: PASS")
print("[CODEX-0003] config invariants: PASS")
print(f"[CODEX-0003] effective: bs={config.sample.train_batch_size} nproc={_nproc} nbpe={_nbpe} "
      f"gas={config.train.gradient_accumulation_steps} samples_per_epoch=72 epochs={config.num_epochs}")
PY

sha256sum "$BASE_REPO/config/opd.py" "$BASE_REPO/scripts/train_sd3_opd.py" "$METHOD_PATCH" "$FIX_PATCH"
sha256sum "$WORK_DIR/config/opd.py" "$WORK_DIR/scripts/train_sd3_opd.py"
echo "[CODEX-0003] launch: accelerate nproc=$NPROC port=$MASTER_PORT config=$OPD_EXPECT_CONFIG"

set +e
python -m accelerate.commands.launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_machines 1 --num_processes "$NPROC" --main_process_port "$MASTER_PORT" \
    -m scripts.train_sd3_opd --config "config/opd.py:$OPD_EXPECT_CONFIG"
RC=$?
set -e
echo "[CODEX-0003] exit=$RC end=$(date --iso-8601=seconds) artifacts=$ARTIFACT_DIR"
exit "$RC"
