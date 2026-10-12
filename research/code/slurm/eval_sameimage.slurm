#!/bin/bash
#SBATCH --job-name=evSI
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:2
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=8:00:00
#SBATCH --array=0-0
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%A_%a.out
# CODEX-0028: same-image 评测 —— 在**同一批生成图像**上同时算任务指标与 PickScore/HPS。
# 回答 codex 0024 §6 的第一优先问题:crg 是只在 held-out prompt 分布上退化,
# 还是连它正在保护的 task images 也变差?
set -uo pipefail
OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
source "$OPD_ROOT/env.sh"
cd "$OPD_ROOT/CA-OPD/code/DiffusionOPD-main"
: "${SI_TSV:?需 SI_TSV}"; : "${SI_DS:?需 SI_DS}"; : "${SI_EVAL_REWARDS:?需 SI_EVAL_REWARDS}"
export SI_EVAL_REWARDS
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$SI_TSV")
NAME=$(echo "$LINE" | cut -f1); CKPT=$(echo "$LINE" | cut -f2)
GS=$(echo "$LINE" | cut -f3);   NS=$(echo "$LINE" | cut -f4)
[ -z "$NAME" ] && { echo "[SI] 无目标"; exit 0; }
NPROC=$(nvidia-smi -L | wc -l)
PORT=$(( 20000 + SLURM_JOB_ID % 40000 ))   # 不按 array 号,避免同节点撞端口
CKARG=(); [ "$CKPT" != "base" ] && CKARG=(--checkpoint_path "$CKPT")
OUT="$OPD_ROOT/outputs/eval/${NAME}_${SI_DS}"
if [ -s "$OUT/average_scores.json" ]; then echo "[SI] 已有结果,跳过"; exit 0; fi
mkdir -p "$OUT"
echo "[SI] $NAME / $SI_DS  rewards=$SI_EVAL_REWARDS  nproc=$NPROC host=$(hostname) $(date)"
python -m torch.distributed.run --nproc_per_node=$NPROC --master_port=$PORT scripts/evaluation.py \
  "${CKARG[@]}" --model_type sd3 --dataset "$SI_DS" --guidance_scale "$GS" \
  --max_samples "$NS" --mixed_precision fp16 --output_dir "$OUT" \
  2>&1 | grep -avE "[0-9]+%\|"
_RC=${PIPESTATUS[0]}
_FAIL=0
[ "$_RC" -ne 0 ] && { echo "[SI] FAILED rc=$_RC"; _FAIL=1; }
[ -f "$OUT/average_scores.json" ] || { echo "[SI] 无 average_scores.json"; _FAIL=1; }
echo "[SI] --- $NAME/$SI_DS ---"; cat "$OUT/average_scores.json" 2>/dev/null | tr -d '\n '; echo
echo "[SI] DONE fail=$_FAIL $(date)"
exit $_FAIL
