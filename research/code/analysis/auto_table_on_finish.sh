#!/bin/bash
# 臂跑完 -> 等评测产出 -> 立刻出全轴对比表。
# 用法: auto_table_on_finish.sh <jobid> <arm> <base> <arm_updates> <step> <run_dir>
set -uo pipefail
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
PY=$O/../conda_envs/envs/diffusionopd/bin/python
JID=$1; ARM=$2; BASE=$3; AUPD=$4; STEP=$5; RUNDIR=$6
OUT=$O/outputs/TABLE_${ARM}_vs_${BASE}_ck${STEP}.txt

echo "[table] 等训练作业 $JID ($ARM) ..."
while [ -n "$(squeue -j "$JID" -h 2>/dev/null)" ]; do sleep 180; done
echo "[table] $JID 已结束 $(date)"
sacct -j "$JID" -o JobID%12,State%14,ExitCode,Elapsed -n | head -2

ls "$RUNDIR" >/dev/null 2>&1                       # 刷 NFS
CK="$RUNDIR/checkpoints/checkpoint-${STEP}/lora/adapter_config.json"
if [ ! -f "$CK" ]; then
  echo "[table] ❌ 无 checkpoint-$STEP —— **训练没跑到终点**,不是'跑完了'。"
  echo "        真实进度:"
  if [ -d "$RUNDIR/checkpoints" ]; then
    ls "$RUNDIR/checkpoints" 2>/dev/null | tail -3 | sed 's/^/          ckpt: /'
  else
    echo "          (连 checkpoints/ 目录都没有 —— 训练在第一次存档前就死了)"
  fi
  # 从 metrics 读真实跑到第几个 epoch,比"有没有 ckpt"更能说明问题
  if [ -f "$RUNDIR/metrics.jsonl" ]; then
    "$PY" - "$RUNDIR/metrics.jsonl" <<'PYEOF'
import json,sys
m=-1; n=0
for l in open(sys.argv[1]):
    n+=1
    try: r=json.loads(l)
    except Exception: continue
    if "epoch" in r and "reward_avg" in r: m=max(m,r["epoch"])
print(f"          metrics {n} 行,最后一个有 reward_avg 的 epoch = {m}")
PYEOF
  else
    echo "          (无 metrics.jsonl)"
  fi
  tail -25 "$O/outputs/slurm_"*"_${JID}.out" 2>/dev/null
  exit 1
fi

# 等两边的评测 json 都出现(评测由 watch_ab_ckpt_hipri.sh 排;最多等 6 小时)
case "$BASE" in np3) BTAG=ab_basenp3 ;; *) BTAG=ab_base ;; esac
A_JSON=$O/outputs/opsd_eval_ab_${ARM}_ck${STEP}_raw.json
B_JSON=$O/outputs/opsd_eval_${BTAG}_ck${STEP}_raw.json
echo "[table] 等评测:$(basename "$A_JSON") 与 $(basename "$B_JSON")"
for i in $(seq 1 120); do
  ls "$O/outputs" >/dev/null 2>&1
  [ -f "$A_JSON" ] && [ -f "$B_JSON" ] && break
  sleep 180
done
miss=0
[ -f "$A_JSON" ] || { echo "[table] ❌ 缺 $A_JSON"; miss=1; }
[ -f "$B_JSON" ] || { echo "[table] ❌ 缺 $B_JSON"; miss=1; }
[ "$miss" -eq 1 ] && { echo "[table] 评测未齐,不出表(宁可没有,也不要口径不全的表)"; exit 1; }

echo "[table] === 全轴对比表 $(date) ===" | tee "$OUT"
"$PY" "$O/ab_compare.py" --arm "$ARM" --base "$BASE" --arm-updates "$AUPD" --step "$STEP" 2>&1 | tee -a "$OUT"
echo "[table] 已写 $OUT"
