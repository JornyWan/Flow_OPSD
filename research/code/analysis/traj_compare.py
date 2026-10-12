#!/usr/bin/env python3
"""训练曲线的臂间比较(**带自相关修正**)。

⚠️ 为什么需要它:对 100 个连续 epoch 做逐点配对 t 检验**违反独立性** ——
相邻 epoch 高度自相关,会把显著性放大。2026-09-01 实测 a3:
  朴素 t=+3.16(过 |t|>2.5) → lag-1 自相关 +0.447 → n_eff=38.2 → **修正后 t=+1.95,不显著**。

⚠️ 训练期 reward_avg **不是评测口径**,任何结论都只能是参考;
   正式判定看 ab_compare.py 的五轴表(单 checkpoint,不涉及自相关)。
"""
import argparse, io, json, math, os, random, statistics as st
O = os.path.dirname(os.path.abspath(__file__))

def traj(d):
    o = {}
    for l in io.open(f"{O}/outputs/{d}/metrics.jsonl"):
        try: r = json.loads(l)
        except Exception: continue
        if "reward_avg" in r and "epoch" in r: o[r["epoch"]] = r["reward_avg"]
    return o

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--block", type=int, default=10)
    a = ap.parse_args()
    A, B = traj(a.arm), traj(a.base)
    c = sorted(set(A) & set(B)); d = [A[e] - B[e] for e in c]
    n = len(d)
    if n < 5: print(f"  共同 epoch 仅 {n} 个,不做统计"); return
    m, sd = st.mean(d), st.stdev(d)
    print(f"  arm={a.arm}")
    print(f"  base={a.base}")
    print(f"  共同 epoch n={n}  均值差={m:+.5f}  sd={sd:.5f}")
    print(f"  逐点差(前 12 个): " + " ".join(f"{x:+.4f}" for x in d[:12]))
    print(f"  朴素 t = {m/(sd/n**0.5):+.2f}   ← **不可用**(相邻 epoch 自相关)")
    dm = [x - m for x in d]
    den = sum(x*x for x in dm)
    r1 = sum(dm[i]*dm[i+1] for i in range(n-1))/den if den > 0 else 0.0
    neff = n*(1-r1)/(1+r1) if r1 < 1 else float("nan")
    t = m/(sd/neff**0.5) if neff == neff and neff > 0 else float("nan")
    print(f"  lag-1 自相关 = {r1:+.3f}  ⇒ n_eff = {neff:.1f}")
    print(f"  **修正后 t = {t:+.2f}**  ⇒ " + ("显著(|t|>2.5)" if abs(t) > 2.5 else "**不显著**"))
    rng = random.Random(0); BL = a.block
    blocks = [d[i:i+BL] for i in range(0, n, BL)]
    cnt = 0
    for _ in range(20000):
        s = []
        while len(s) < n: s.extend(rng.choice(blocks))
        if st.mean(s[:n]) <= 0: cnt += 1
    print(f"  分块 bootstrap(块长 {BL}):P(均值≤0) = {cnt/20000:.4f}")
    h = n//2
    print(f"  前半 ep{c[0]}-{c[h-1]} 均值={st.mean(d[:h]):+.5f}   后半 ep{c[h]}-{c[-1]} 均值={st.mean(d[h:]):+.5f}")
    print("  ⚠️ 训练 reward 非评测口径;判定看 TABLE_*.txt 的五轴表。")

main()
