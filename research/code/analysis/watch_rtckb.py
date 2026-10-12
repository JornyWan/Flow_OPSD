#!/usr/bin/env python
"""监控 rtckb: 打印原始序列(不是汇总统计 —— 汇总曾把 V 形回升误读为发散)。
主要信号是 kbest_idx / kbest_gain: 若它们持续走高,说明模型在远离高 reward 区。"""
import sys, glob, statistics as st
O = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related"
sys.path.insert(0, O)
from read_wandb import read
p = sys.argv[1] if len(sys.argv) > 1 else \
    "/midtier/minilab/scratch/chw4026/.cache/wandb/wandb/offline-run-20260826_032007-gyjeluyk"
s = read(glob.glob(p + "/*.wandb")[0])
for k in ("monitor_pickscore", "rtc_kbest_idx", "rtc_kbest_gain", "rtc_loss", "grad_norm"):
    v = [x for _, x in s.get(k, [])]
    if not v: continue
    tail = v[-16:]
    print(f"{k:18s} n={len(v):3d}  最近16: " + " ".join(f"{x:.3f}" for x in tail))
    if len(v) >= 10:
        a, b = st.mean(v[:5]), st.mean(v[-5:])
        print(f"{'':18s}   首5均={a:.4f}  末5均={b:.4f}  变化={b-a:+.4f}")
