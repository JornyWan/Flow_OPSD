#!/bin/bash
# 等某个训练作业写出指定 checkpoint,然后**主动取消**它,把算力让给后续实验。
# 用途:1037080 是 300-update 未播种跑;用户已把标准改为 100 updates。
# 让它跑到 ck100 再停 —— 既拿到一个**未播种的 Open3@100**(与播种 baseline 对照可估 RNG 波动),
# 又不必占着 3 张 H200 到明早 07:05。
# 用法: stop_at_ckpt.sh <jobid> <ckroot> <step>
set -uo pipefail
JID=$1; CKROOT=$2; STEP=$3
CK="$CKROOT/checkpoint-$STEP"
echo "[stop] 等 $CK 落盘后取消 job $JID  $(date)"
while true; do
  [ -z "$(squeue -j "$JID" -h 2>/dev/null)" ] && { echo "[stop] job $JID 已自行结束,无需取消 $(date)"; exit 0; }
  ls "$CKROOT" >/dev/null 2>&1                      # 刷 NFS 属性缓存(必需)
  if [ -f "$CK/lora/adapter_config.json" ]; then
    sleep 90                                        # 宽限:等 ema/optimizer 等文件写完
    ls "$CK" >/dev/null 2>&1
    echo "[stop] $CK 就绪,内容: $(ls "$CK" 2>/dev/null | tr '\n' ' ')"
    scancel "$JID" && echo "[stop] 已取消 $JID $(date)"
    sleep 5; scancel "$JID" 2>/dev/null              # requeue TERM-trap 会吃掉第一个信号
    exit 0
  fi
  sleep 120
done
