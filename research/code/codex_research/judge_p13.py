#!/usr/bin/env python
"""§4.869 PS1 判定器(数据前写定)。用法:judge_p13.py(读 outputs/p13/u1k_*.json 与 db_*.json)。"""
import glob, json, math
MDE = 0.0033
def load(tag):
    rows = []
    for f in sorted(glob.glob(f"outputs/p13/{tag}_sh*of*.json")): rows += json.load(open(f))["rows"]
    return rows
o3 = lambda c: c["pickscore"] / 26.0 + c["clipscore"] + c["hpsv2"]
ir = lambda c: c["imagereward"]
def st(v):
    n = len(v); m = sum(v) / n; sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else float("nan")
    return m, (m / (sd / math.sqrt(n)) if sd and sd > 0 else float("nan")), n
u = load("u1k"); db = load("db")
if not u: raise SystemExit("没有 u1k 数据")
by = {}
for r in u: by.setdefault(r["pi"], {})[r["noise"]] = r["cells"]
pis = sorted(p for p in by if 0 in by[p] and 1 in by[p])
cands = list(by[pis[0]][0].keys())
print(f"u1k prompts with A&B: {len(pis)} | candidates {len(cands)} | db rows {len(db)}")
for s in ("1", "3"):
    sets = {"S": [c for c in cands if c[1:3] in ("e1", "e2") and c.endswith("@" + s)],
            "Rd": [c for c in cands if c[1:3] == "e3" and c.endswith("@" + s)]}
    for nm, C in sets.items():
        if not C: continue
        gs, gx, gxi, win = [], [], [], {}
        for p in pis:
            A, Bc = by[p][0], by[p][1]
            best = max(["base"] + C, key=lambda k: o3(A[k]))
            win[best] = win.get(best, 0) + 1
            gs.append((o3(A[best]) - o3(A["base"])) / MDE)
            gx.append((o3(Bc[best]) - o3(Bc["base"])) / MDE)
            gxi.append(ir(Bc[best]) - ir(Bc["base"]))
        m1, t1, _ = st(gs); m2, t2, _ = st(gx); m3, t3, _ = st(gxi)
        print(f"s={s} {nm:2s}: G_search {m1:+.2f}×MDE | G_xfer(open3) {m2:+.2f}×MDE (t={t2:+.2f}) | G_xfer(imagereward) {m3:+.4f} (t={t3:+.2f}) | 赢家分布 {win}")
print("\n逐候选在噪声 A / B / db 上的均值增益(open3 ×MDE)与噪声 A→B 的跨噪声一致性:")
for c in cands:
    if c == "base": continue
    a_ = st([(o3(by[p][0][c]) - o3(by[p][0]["base"])) / MDE for p in pis])
    b_ = st([(o3(by[p][1][c]) - o3(by[p][1]["base"])) / MDE for p in pis])
    d_ = st([(o3(r["cells"][c]) - o3(r["cells"]["base"])) / MDE for r in db]) if db else (float("nan"),) * 3
    print(f"   {c:9s} A {a_[0]:+6.2f}(t={a_[1]:+5.1f})  B {b_[0]:+6.2f}(t={b_[1]:+5.1f})  db {d_[0]:+6.2f}(t={d_[1]:+5.1f})")
s1 = [c for c in cands if c[1:3] in ("e1", "e2") and c.endswith("@1")]
gx = []
for p in pis:
    A, Bc = by[p][0], by[p][1]
    best = max(["base"] + s1, key=lambda k: o3(A[k])); gx.append((o3(Bc[best]) - o3(Bc["base"])) / MDE)
m, t, n = st(gx)
print(f"\n主判定(§4.869 PS1):G_xfer(S, s=1) = {m:+.2f}×MDE, t={t:+.2f} ⇒ **{'进 PS2' if (m > 0 and t > 2) else '不进 PS2(先改设计)'}**")
