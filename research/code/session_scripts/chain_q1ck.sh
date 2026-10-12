#!/bin/bash
# §4.853 链:fork job 结束 → 第一段还原(自检不过即停)→ 算 canoneq 的 k → 提交评测数组。用法:chain_q1ck.sh <ck> <fork_jobid>
set -uo pipefail
ck=$1; jid=$2
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
PY=$O/../conda_envs/envs/diffusionopd/bin/python
cd $O
until [ -z "$(squeue -j $jid -h 2>/dev/null)" ]; do sleep 60; done
S=outputs/ab_open3_qlook1_s20260830_u100/checkpoints
U=$((ck + 1)); F=""
for s in 20260830 20260831; do for a in canon look1; do
  d=outputs/q1ck${ck}_${a}_s$s/checkpoints/checkpoint-$U
  [ -f $d/optimizer.pt ] || { echo "[chain ck$ck] ❌ 缺 $d"; exit 1; }
  F="$F${F:+,}q1ck${ck}_${a}_s$s=$d"
done; done
$PY probes/p11_abcd_dirs.py --pre $S/checkpoint-$ck --ref $S/checkpoint-$((ck - 10)) --forks "$F" \
    --out outputs/p11/dirs_q1ck$ck.pt 2>&1 | grep "p11-dirs" || exit 1
[ -f outputs/p11/dirs_q1ck$ck.pt ] || { echo "[chain ck$ck] ❌ 第一段没写输出(自检不过)"; exit 1; }
C=$(python3 - <<E
import json; s=json.load(open("outputs/p11/dirs_q1ck$ck.pt.json"))["forks"]
c=["M:0"]
for sd in ("20260830","20260831"):
    k=s[f"q1ck${ck}_look1_s{sd}"]["norm_delta"]/s[f"q1ck${ck}_canon_s{sd}"]["norm_delta"]
    c += [f"q1ck${ck}_canon_s{sd}:1", f"q1ck${ck}_canon_s{sd}:{k:.4f}", f"q1ck${ck}_look1_s{sd}:1"]
print(",".join(c))
E
)
echo "[chain ck$ck] combos=$C"
TAG=q1ck$ck NSHARD=25 PROMPTS=$O/CA-OPD/code/DiffusionOPSD-rpi/data/pickapic/test.txt DIRS=$O/outputs/p11/dirs_q1ck$ck.pt \
  COMBOS="$C" BS=8 EXTRA="--selftest" PRE_LORA=$O/$S/checkpoint-$ck/lora \
  sbatch --export=ALL --array=0-24 --gres=gpu:1 --exclude=ai-gpu01,ai-gpu03,ai-gpu04 --time=01:30:00 slurm/p11_abcd_eval.slurm
