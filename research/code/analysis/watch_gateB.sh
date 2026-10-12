#!/bin/bash
# 等 Gate B 探针作业结束 -> 刷 NFS 缓存 -> 跑判读。用法: watch_gateB.sh <jobid> <probe_dir>
set -uo pipefail
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
JID=$1; DIR=$2
echo "[gateB-watch] 等作业 $JID ($(basename "$DIR")) ..."
while [ -n "$(squeue -j "$JID" -h 2>/dev/null)" ]; do sleep 180; done
echo "[gateB-watch] $JID 已结束 $(date)"
ls "$DIR" >/dev/null 2>&1                       # 刷 NFS 属性缓存(先 ls)
F="$DIR/gateB_noise.jsonl"
[ -f "$F" ] || F="$DIR/rtcReach_probe.jsonl"   # 同一监视也服务 RTC 探针
if [ ! -f "$F" ]; then
  echo "[gateB-watch] ❌ 无 $F —— 作业结束但**没产出指标**,须查真因,不要当成'跑完了'"
  sacct -j "$JID" -o JobID%12,State%14,ExitCode,Elapsed -n | head -3
  tail -25 "$O/outputs/slurm_"*"_${JID}.out" 2>/dev/null
  exit 1
fi
n=$(wc -l < "$F"); n=${n:-0}                    # grep/wc 空文件的双行陷阱,显式兜底
echo "[gateB-watch] 记录数 = $n"
"$O/../conda_envs/envs/diffusionopd/bin/python" $([ "$(basename "$F")" = "rtcReach_probe.jsonl" ] && echo "$O/rtcReach_analyze.py" || echo "$O/gateB_analyze.py") --path "$F"
