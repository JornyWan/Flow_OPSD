#!/usr/bin/env python
"""§4.870 PS2 判定器(数据前写定)。"""
import json, math, sys
MDE = 0.0033
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "outputs/p14/ps2.json"))
o3 = lambda c: c["pickscore"] / 26.0 + c["clipscore"] + c["hpsv2"]
def st(v):
    n = len(v); m = sum(v) / n; sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    return m / MDE, m / (sd / math.sqrt(n)), n
print("更新信息:", {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in d["info"].items()})
R = {}
for s in ("u1kA", "u1kB", "db"):
    rows = [r for r in d["rows"] if r["set"] == s]
    print(f"\n[{s}] n={len(rows)}")
    for a_, b_ in (("Sm", "M"), ("Sq", "M"), ("Rm", "M"), ("CANON", "M"), ("Sm", "CANON"), ("Sm", "Sq"), ("Sm", "Rm"),
                   ("M", "base"), ("CANON", "base"), ("Sm", "base")):
        if a_ not in rows[0]["cells"] or b_ not in rows[0]["cells"]: continue
        m, t, n = st([o3(r["cells"][a_]) - o3(r["cells"][b_]) for r in rows])
        mi, ti, _ = st([(r["cells"][a_]["imagereward"] - r["cells"][b_]["imagereward"]) * MDE for r in rows])
        R[(s, a_, b_)] = (m, t)
        print(f"   {a_:5s} − {b_:5s}  open3 {m:+6.2f}×MDE (t={t:+5.2f}) | imagereward {mi*1:+.4f} (t={ti:+5.2f})")
dbt, ut, at_ = R[("db", "Sm", "M")], R[("u1kB", "Sm", "M")], R[("u1kA", "Sm", "M")]
if dbt[0] > 0 and dbt[1] > 2 and ut[0] > 0 and ut[1] > 1: v = "L1 学进去且泛化"
elif dbt[1] < -2: v = "L4 有害"
elif at_[1] > 2 and abs(dbt[1]) < 2: v = "L2 只在样本内"
elif abs(at_[1]) < 2: v = "L3 没学进去"
else: v = "未判"
print(f"\n主判定(§4.870 PS2):**{v}**")
