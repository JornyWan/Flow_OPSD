#!/usr/bin/env python
"""§4.853 判定器(数据前写定)。在 qlook1 的 ck50 / ck80 上:
主 ΔD_eq = R(look1|1) − R(canon|k)(canon 的本批梯度放大到 look1 的步长),副 ΔD = R(look1|1) − R(canon|1)。
分类同 §4.850 §八:pos 两 seed 都 t>2 | neg 都 t<−2 | null 都 |t|<2 且 |均值|<0.3×MDE | mixed 其余。
S1 ΔD_eq,50∈{pos,null} 且 ΔD_eq,80=neg | S2 两者都∈{pos,null} | S3 ΔD_eq,50=neg | S4 其余。"""
import glob, json, math, os
MDE = 0.0033
SEEDS = ("20260830", "20260831")
LEVS = ("B", "C-B", "P", "C", "D")


def load(ck):
    fs = sorted(glob.glob(f"outputs/p11/q1ck{ck}_sh*of*.json"))
    if not fs: return None, None, False
    rows = []
    for f in fs: rows += json.load(open(f))["rows"]
    st = json.load(open(f"outputs/p11/dirs_q1ck{ck}.pt.json"))["forks"]
    k = {s: "%g" % float("%.4f" % (st[f"q1ck{ck}_look1_s{s}"]["norm_delta"] / st[f"q1ck{ck}_canon_s{s}"]["norm_delta"]))
         for s in SEEDS}
    return rows, k, len(fs) == 25


def o3(c): return c["pickscore"] + c["clipscore"] + c["hpsv2"]


def R(rows, cell, lev, i):
    if lev == "C-B": return R(rows, cell, "C", i) - R(rows, cell, "B", i)
    return o3(rows[i]["cells"][f"{cell}|{lev}"])


def stat(v):
    n = len(v); m = sum(v) / n; sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    return m / MDE, m / (sd / math.sqrt(n))


def cls(ms):
    if any(m is None for m in ms): return "缺"
    if all(t > 2 for _, t in ms): return "pos"
    if all(t < -2 for _, t in ms): return "neg"
    if all(abs(t) < 2 and abs(x) < 0.3 for x, t in ms): return "null"
    return "mixed"


fmt = lambda m: "%+6.2f(t=%+5.1f)" % m
res, complete = {}, True
for ck in (50, 60, 70, 80):
    rows, k, done = load(ck)
    if rows is None:
        print(f"ck{ck}: ⏳ 无数据"); complete = False; continue
    complete &= done
    N = range(len(rows))
    print(f"\nck{ck}: 图 {len(rows)}{'' if done else '  ⚠️ 分片未齐'} | k(canoneq) = {k}")
    for L in LEVS:
        eq = [stat([R(rows, f"q1ck{ck}_look1_s{s}|1", L, i) - R(rows, f"q1ck{ck}_canon_s{s}|{k[s]}", L, i) for i in N]) for s in SEEDS]
        nat = [stat([R(rows, f"q1ck{ck}_look1_s{s}|1", L, i) - R(rows, f"q1ck{ck}_canon_s{s}|1", L, i) for i in N]) for s in SEEDS]
        res[(ck, L)] = eq
        print(f"  {L:3s} 等范数(主) " + "  ".join(fmt(m) for m in eq) + f" ⇒ {cls(eq):5s} | 自然 " + "  ".join(fmt(m) for m in nat) + f" ⇒ {cls(nat)}")
    for a in ("canon", "look1"):
        print(f"  G_{a}(D, 本批梯度本身) " + "  ".join(fmt(stat([R(rows, f'q1ck{ck}_{a}_s{s}|1', 'D', i) - R(rows, 'M|0', 'D', i) for i in N])) for s in SEEDS))

if complete and (50, "D") in res and (80, "D") in res:
    c50, c80 = cls(res[(50, "D")]), cls(res[(80, "D")])
    if c50 == "neg": v = "S3 ck50 就已更差"
    elif c50 in ("pos", "null") and c80 == "neg": v = "S1 状态依赖的翻转"
    elif c50 in ("pos", "null") and c80 in ("pos", "null"): v = "S2 一步看不出"
    else: v = "S4 未判"
    print(f"\n主判定(§4.853):ΔD_eq,50={c50} ΔD_eq,80={c80} ⇒ **{v}**")
else:
    print("\n主判定:⏳ 不判(数据未齐)")
