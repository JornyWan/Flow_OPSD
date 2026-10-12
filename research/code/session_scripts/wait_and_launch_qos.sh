#!/bin/bash
# 等到**某个节点真有 N 张空闲**再定点申请,避免 salloc→scancel 空转。
# 背景(§4.382):本集群资源不足时会**截断授予**(要 3 给 2)而不是排队;
# 且已确认账户无 TRES 限制(GrpTRES/MaxTRES 全空)⇒ 只能自己先核对节点空闲量。
# 空闲量必须读 scontrol 的 AllocTRES:squeue 里别人的作业可能报无类型的 gres/gpu:1,
# 用 'gpu:类型:N' 的正则会漏算占用、高估空闲。
# 用法: wait_and_launch.sh ARM SEED UPDATES OUTDIR GPUTYPE 最少空闲数 PART ACCT "节点列表"
#   env: REQ(申请张数,默认 4=整节点)  ACCEPT(可接受的实得张数集合,默认 "3 4")
# 固定动作(3)的铁律:申请 h200:4/a100:4 + EXPECT_NPROC="3 4" —— 要整节点卡数时 SLURM 不会半给。
set -u
ARM=$1; SEED=$2; UPDATES=$3; OUTDIR=$4; TYP=$5; NGPU=$6; PART=$7; ACCT=$8; NODES=$9
REQ=${REQ:-4}                 # 申请整节点卡数:SLURM 对"要全部"不会半给
ACCEPT=${ACCEPT:-"3 4"}       # 实得落在此集合内即接受(EXPECT_NPROC 同值)
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$B" || exit 1
free_on(){ local n=$1 s tot used
  s=$(scontrol show node "$n" 2>/dev/null) || return 1
  echo "$s" | grep -q "Gres=gpu:${TYP}:" || return 1
  tot=$(echo "$s" | grep -oP "Gres=gpu:${TYP}:\K[0-9]+" | head -1)
  used=$(echo "$s" | grep -oP 'AllocTRES=[^ ]*gres/gpu[^=]*=\K[0-9]+' | head -1); used=${used:-0}
  echo $(( ${tot:-0} - used )); }
for t in $(seq 1 2000); do
  for n in $NODES; do
    f=$(free_on "$n") || continue
    [ "${f:-0}" -ge "$NGPU" ] || continue
    echo "[wl:$ARM] $n 有 $f 张空闲 -> 申请 $NGPU 张 $(date)"
    # --immediate=90:资源拿不到就快速失败,不要阻塞。
    # 背景:被 reservation 占住的节点在 scontrol 的 AllocTRES 里仍显示空闲,
    # 轮询会误判为可用;不加 --immediate 时 salloc 会无限期挂住,整个排队器卡死。
    OUT=$(salloc --no-shell --immediate=90 --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --mem=300G \
          --time=24:00:00 --gres="gpu:${TYP}:${REQ}" --nodelist="$n" \
          --partition="$PART" --account="$ACCT" ${QOS:+--qos=$QOS} --job-name="$ARM" 2>&1)
    echo "$OUT"
    J=$(echo "$OUT" | grep -oP 'job allocation \K[0-9]+' | head -1)
    [ -n "$J" ] || { sleep 60; continue; }
    GOT=$(scontrol show job "$J" 2>/dev/null | grep -oP 'AllocTRES=[^ ]*gres/gpu[^=]*=\K[0-9]+' | head -1)
    _ok=0; for _a in $ACCEPT; do [ "${GOT:-0}" = "$_a" ] && _ok=1; done
    if [ "$_ok" != 1 ]; then
      echo "[wl:$ARM] ⚠️ 实得 ${GOT:-0} 张,不在 {$ACCEPT} 内 -> scancel $J"; scancel "$J"; sleep 60; continue
    fi
    echo "[wl:$ARM] ✅ job=$J 实得 $GOT 张 @ $n(申请 $REQ),起 step"
    export ARM SEED UPDATES OUTPUT_DIR="$OUTDIR" EXPECT_NPROC="$ACCEPT"
    # ⚠️ step 必须显式带 --gres=实得张数:否则它继承的是 job 的**申请值**(TresPerNode),
    # 与实得不符就报 "Invalid generic resource (gres) specification",
    # 而分配还在 ⇒ 作业 RUNNING 却空转占卡。2026-09-08 实测:申请4/实得3 即触发。
    # ⚠️ 不能用 exec:`salloc --no-shell` 的分配**不会**随 step 结束而释放,
    #    2026-09-09 实测一个被锁挡下的副本 step 秒退,分配却继续占 3 张 A100(要到 24h 时限)。
    #    所以 step 结束后必须显式 scancel。
    srun --jobid="$J" --nodes=1 --ntasks=1 --gres="gpu:${TYP}:${GOT}" bash slurm/ab_open3.slurm
    _rc=$?
    echo "[wl:$ARM] step 结束 rc=$_rc -> scancel $J 释放分配"
    scancel "$J" 2>/dev/null
    exit $_rc
  done
  sleep 120
done
echo "[wl:$ARM] ❌ 超时未等到 $NGPU 张空闲"
