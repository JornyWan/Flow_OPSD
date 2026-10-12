#!/usr/bin/env python
"""对 gc_probe 的各方向做配对检验。

配对而非独立检验: 同一 epoch 的所有方向共用同一批 prompt 与同一个 y0,
批间差异(gc_F_noop 在 0.79~0.83 波动)是主要方差来源,配对能把它消掉。
"""
import sys, glob, os, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_wandb import read

path = sys.argv[1]
if os.path.isdir(path):
    c = glob.glob(os.path.join(path, "*.wandb")); path = c[0]
s = read(path)
V = {k.replace("gc_G_", ""): [v for _, v in s[k]] for k in s if k.startswith("gc_G_")}
if not V:
    print("没有 gc_G_* 数据"); sys.exit(1)
n = min(len(v) for v in V.values())
print(f"n = {n}\n")
print(f"{'方向':12s} {'mean':>10s} {'sd':>9s} {'se':>9s}")
for k in sorted(V, key=lambda x: -st.mean(V[x])):
    v = V[k][:n]; m = st.mean(v); sd = st.stdev(v) if n > 1 else 0.0
    print(f"{k:12s} {m:+10.5f} {sd:9.5f} {sd/n**0.5:9.5f}")

pairs = [a.split("-") for a in sys.argv[2:]] or [
    ["tproj", "teacher"], ["tproj", "tsum"], ["teacher", "tsum"],
    ["rtpbest", "teacher"], ["rtpbest", "rtprand"], ["teacher", "rand"],
]
print(f"\n{'配对':26s} {'diff':>10s} {'se':>9s} {'t':>8s}  判定")
for a, b in pairs:
    if a not in V or b not in V: continue
    d = [V[a][i] - V[b][i] for i in range(n)]
    m = st.mean(d)
    if n < 2: continue
    se = st.stdev(d) / n ** 0.5
    t = m / se if se > 0 else float("inf")
    verdict = "显著" if abs(t) > 2.5 else "打平(噪声内)"
    print(f"{a+' - '+b:26s} {m:+10.5f} {se:9.5f} {t:+8.1f}  {verdict}")
