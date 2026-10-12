#!/usr/bin/env python
"""cert=1 vs cert=0 的训练 reward 轨迹配对对比(不依赖离线评测)。

官方 OPSD 每 epoch 往 metrics.jsonl 写 reward_avg。同 epoch 配对能消掉"训练进度"这个
主要方差来源;逐 epoch 差值再做符号检验 + 配对 t,两个都报。
"""
import json, os, sys, statistics as st
O = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs"
KEY = os.environ.get("KEY", "reward_avg")

def load(d):
    p = os.path.join(O, d, "metrics.jsonl")
    if not os.path.exists(p): return {}
    out, ep = {}, None
    for l in open(p):
        if not l.strip(): continue
        try: r = json.loads(l)
        except Exception: continue
        if "epoch" in r: ep = int(r["epoch"])          # epoch 与指标分行写,需承接
        if KEY in r and ep is not None: out[ep] = r[KEY]
    return out

runs = sys.argv[1:] or ["opsd_cert100_sd35_pickscore", "opsd_base100_sd35_pickscore"]
data = {d: load(d) for d in runs}
for d, v in data.items():
    print(f"{d:<34} n={len(v):>4}" + (f"  epoch {min(v)}~{max(v)}  末值={v[max(v)]:.4f}" if v else "  (无数据)"))
if len(runs) < 2 or not all(data.values()): sys.exit()
a, b = data[runs[0]], data[runs[1]]
common = sorted(set(a) & set(b))
if not common: sys.exit("\n无公共 epoch")
d = [a[e] - b[e] for e in common]
n = len(d); m = st.mean(d)
sd = st.stdev(d) if n > 1 else 0.0
se = sd / n ** 0.5 if n > 1 else 0.0
t = m / se if se > 0 else 0.0
pos = sum(1 for x in d if x > 0)
print(f"\n公共 epoch n={n} ({common[0]}~{common[-1]})")
print(f"  {runs[0]} 均值={st.mean([a[e] for e in common]):.4f}")
print(f"  {runs[1]} 均值={st.mean([b[e] for e in common]):.4f}")
print(f"  逐 epoch 差 (前−后): 均值={m:+.5f}  se={se:.5f}  t={t:+.2f}  同号 {pos}/{n}")
# 符号检验(正态近似),对非正态更稳
if n >= 6:
    z = (pos - n / 2) / (n / 4) ** 0.5
    print(f"  符号检验 z={z:+.2f}  {'显著' if abs(z) > 1.96 else '不显著'}")
print("  判据: |t|>2 且符号检验显著才算真差异;两者不一致时以符号检验为准(轨迹自相关会让 t 虚高)")
