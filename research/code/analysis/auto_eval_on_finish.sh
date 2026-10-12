#!/bin/bash
# 等某个训练 arm 结束 -> 自动按其 checkpoint 排轨迹评测(A40)。
# 用法: auto_eval_on_finish.sh <jobid> <run_name_dir> <tag> <steps...>
set -uo pipefail
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
JID=$1; RUNDIR=$2; TAG=$3; shift 3; STEPS="$@"
echo "[auto] 等作业 $JID 结束..."
while squeue -j "$JID" -h >/dev/null 2>&1 && [ -n "$(squeue -j "$JID" -h 2>/dev/null)" ]; do sleep 120; done
echo "[auto] $JID 已结束 $(date)"
# CODEX-0018: CR 系的 save_dir 是持久 artifact 绝对路径,不在 logs/ 下。
case "$RUNDIR" in
    /*) CK="$RUNDIR/checkpoints" ;;
    *)  CK="$O/CA-OPD/code/DiffusionOPD-main/logs/$RUNDIR/checkpoints" ;;
esac
ls "$CK" >/dev/null 2>&1   # 刷 NFS 属性缓存
T=/midtier/minilab/scratch/chw4026/tmp/${TAG}_targets.tsv
: > "$T"
for s in $STEPS; do
  [ -d "$CK/checkpoint-$s" ] && printf "%s_%s\t%s\t4.5\t256\n" "$TAG" "$s" "$CK/checkpoint-$s" >> "$T"
done
n=$(wc -l < "$T")
echo "[auto] 目标 $n 个: $(cut -f1 "$T" | tr '\n' ' ')"
[ "$n" -eq 0 ] && { echo "[auto] ❌ 没找到 checkpoint,退出"; exit 1; }
S=$O/slurm/eval_${TAG}.slurm
sed -e "s|/midtier/minilab/scratch/chw4026/tmp/eval_targets.tsv|$T|" \
    -e "s|^#SBATCH --array=.*|#SBATCH --array=0-$((n-1))%4|" \
    -e "s|#SBATCH --job-name=evalall|#SBATCH --job-name=ev${TAG}|" \
    $O/slurm/eval_all.slurm > "$S"
chmod +x "$S"
J=$("$O/sbatch_retry.sh" --tries 120 --sleep 60 "$S")
echo "[auto] 已提交评测作业 $J"
