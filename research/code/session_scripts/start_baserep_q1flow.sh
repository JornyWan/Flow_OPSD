#!/bin/bash
# GPU 放开:gpu06(A100×4 全空)、gpu14(H100×4 全空),两节点 IDLE ⇒ 不抢占任何人。
#  ① baserep  —— 纯重复 run,测 run-to-run 噪声(§4.549 记为项目最大方法学空缺)。A100 配对 baseAA。
#  ② q1flow   —— §4.516/§4.519 的后缀 solver 探针,UPDATES=6。原 launcher 在等 H200,改派 H100。
# 用 v5(§4.558:不原地改可能正在被执行的脚本)。
set -u
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
S=$(dirname "$0"); SELF=$$
cd "$B" || exit 1

echo "== ① baserep @ A100(gpu06)=="
OUT1="$B/outputs/ab_open3_baserep_s20260830_u100"
REQ=4 ACCEPT="3 4" SCRIPT=slurm/ab_open3_v5.slurm \
  nohup bash "$S/wait_and_launch_v4.sh" baserep 20260830 100 "$OUT1" a100 3 \
        sablab-gpu sablab "ai-gpu06" > "$S/wl_baserep.log" 2>&1 &
echo "   launcher pid=$!"
for ck in 20 50 100; do
  nohup bash "$S/watch_srun_eval_long2.sh" baserep "$OUT1" "$ck" > "$S/se2_baserep_ck${ck}.log" 2>&1 &
  echo "   watcher ck$ck pid=$!  (CK_MULT=1 ⇒ ck = 外层 epoch)"
done

echo "== ② q1flow @ H100(gpu14)=="
# 先停掉在等 H200 的旧 launcher,避免两处同时起(锁会挡,但省一次空转)
k=0
while read -r pid args; do
  [ "$pid" = "$SELF" ] || [ "$pid" = "$PPID" ] && continue
  case "$args" in *wait_and_launch_v4.sh\ q1flow\ *) kill "$pid" 2>/dev/null && k=$((k+1));; esac
done < <(ps -u chw4026 -o pid=,args=)
echo "   停掉旧 H200 launcher: $k 个"
OUT2="$B/outputs/ab_open3_q1flow_s20260830_u6"
REQ=4 ACCEPT="3 4" SCRIPT=slurm/ab_open3_v5.slurm \
  nohup bash "$S/wait_and_launch_v4.sh" q1flow 20260830 6 "$OUT2" h100 3 \
        minilab-gpu minilab "ai-gpu14" > "$S/wl_q1flow_h100.log" 2>&1 &
echo "   launcher pid=$!  (探针产物是 gateQ1_probe.jsonl,不需评测 watcher)"
