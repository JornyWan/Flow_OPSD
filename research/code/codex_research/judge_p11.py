#!/usr/bin/env python
"""§4.850 §八 判定器(数据前写定)。用法:judge_p11.py <tag>
格:`{name}|{k}|{L}`,work = θ300 + Δ0 + k·δ_name;M = θ300 + Δ0。open3 = pick/26 + clip + hps,单位 ×MDE。
主对比 ΔX = R(look1_s|1|X) − R(canon_s|1|X)(逐图配对)。副:ΔX_eq(look1_s|k_eq − canon_s|1)、G_f = R(f|1) − R(M)。
分类(每 seed paired t):pos 两 seed 都 t>2 | neg 都 t<−2 | null 都 |t|<2 且 |均值|<0.3 | mixed 其余。
O1 ΔB∈{null,neg} | O2 ΔB=pos 且 ΔD∈{null,neg} | O3 ΔB=pos 且 ΔD=pos | O4 其余。
K4:canonrep_s20260830 − canon_s20260830 的 ΔB、ΔD 必须 null(单 seed)。"""
import glob, json, math, sys
MDE = 0.0033
SEEDS = ("20260830", "20260831")
KEQ = {"20260830": "0.6234", "20260831": "0.6081"}
LEVS = ("B", "C-B", "P", "C", "D")
tag = sys.argv[1] if len(sys.argv) > 1 else "main"

files = sorted(glob.glob(f"outputs/p11/{tag}_sh*of*.json"))
if not files: sys.exit(f"没有 outputs/p11/{tag}_sh*.json")
rows, metas = [], []
for f in files:
    d = json.load(open(f)); rows += d["rows"]; metas.append(d["meta"])
nsh = max(int(f.split("of")[-1].split(".")[0]) for f in files)
done = len(files) == nsh
print(f"[{tag}] 分片 {len(files)}/{nsh} | 图 {len(rows)} | i*={metas[0]['istar']} σ={metas[0]['sigma_istar']:.4f}"
      + ("" if done else "   ⚠️ 分片未齐,只看不判"))
cells = set(rows[0]["cells"])
N = range(len(rows))


def o3(c): return c["pickscore"] + c["clipscore"] + c["hpsv2"]


def R(cell, lev, i):
    if lev == "C-B":
        return R(cell, "C", i) - R(cell, "B", i)
    c = rows[i]["cells"]
    return o3(c["Pre"]) if cell == "Pre" else o3(c[f"{cell}|{lev}"])


def stat(v):
    n = len(v); m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    return m / MDE, (m / (sd / math.sqrt(n)) if sd > 0 else float("nan"))


def has(cell): return f"{cell}|D" in cells


def diff(a, b, lev): return stat([R(a, lev, i) - R(b, lev, i) for i in N])


def cls(ms):
    if any(m is None for m in ms): return "缺"
    if all(t > 2 for _, t in ms): return "pos"
    if all(t < -2 for _, t in ms): return "neg"
    if all(abs(t) < 2 and abs(x) < 0.3 for x, t in ms): return "null"
    return "mixed"


fmt = lambda m: "%+6.2f(t=%+5.1f)" % m if m else "   —   "
print("\n各组合相对 Pre(×MDE;C-B 行 = 该组合的 C−B,Pre 的 C−B 恒为 0):")
for c in sorted({k.rsplit("|", 1)[0] for k in cells if k != "Pre"}):
    print(f"  {c:26s} " + " ".join(f"{L}={fmt(diff(c, 'Pre', L))}" for L in LEVS))
    add = stat([R(c, "D", i) - R(c, "P", i) - R(c, "C", i) + R("Pre", "D", i) for i in N])
    print(f"  {'':26s} 可加性 (D−Pre)−(P−Pre)−(C−Pre) = {add[0]:+.2f}×MDE")

if has("M|0"):
    print("\nG_f = R(θ301_f) − R(M)(给定动量后,本批梯度本身的作用):")
    for s in SEEDS:
        for f in ("canon", "look1"):
            c = f"{f}_s{s}|1"
            if has(c): print(f"  {c:26s} " + " ".join(f"{L}={fmt(diff(c, 'M|0', L))}" for L in LEVS))

res, reseq = {}, {}
for L in LEVS:
    res[L] = [diff(f"look1_s{s}|1", f"canon_s{s}|1", L) if has(f"look1_s{s}|1") and has(f"canon_s{s}|1") else None for s in SEEDS]
    reseq[L] = [diff(f"look1_s{s}|{KEQ[s]}", f"canon_s{s}|1", L) if has(f"look1_s{s}|{KEQ[s]}") and has(f"canon_s{s}|1") else None
                for s in SEEDS]
print("\nΔX = look1 − canon(自然步长,主)                     ΔX_eq(等范数,副)")
for L in LEVS:
    print(f"  {L:3s} " + "  ".join(fmt(m) for m in res[L]) + f"  ⇒ {cls(res[L]):5s}   |  "
          + "  ".join(fmt(m) for m in reseq[L]) + f"  ⇒ {cls(reseq[L])}")

ok4 = None
if has("canonrep_s20260830|1") and has("canon_s20260830|1"):
    k4 = {L: diff("canonrep_s20260830|1", "canon_s20260830|1", L) for L in ("B", "D")}
    ok4 = all(abs(t) < 2 and abs(x) < 0.3 for x, t in k4.values())
    print("\nK4 零模型 canonrep − canon:" + "  ".join(f"{L} {fmt(v)}" for L, v in k4.items()) + ("  ✅" if ok4 else "  ❌ 不判"))


def verdict(r):
    cB, cD = cls(r["B"]), cls(r["D"])
    if cB in ("null", "neg"): return cB, cD, "O1 target 没在 query 处兑现"
    if cB == "pos" and cD in ("null", "neg"): return cB, cD, "O2 兑现了、被共享参数抵消"
    if cB == "pos" and cD == "pos": return cB, cD, "O3 一步 policy 改进成立"
    return cB, cD, "O4 未判"


if done and ok4:
    cB, cD, v = verdict(res); _, _, veq = verdict(reseq)
    print(f"\n主判定(§4.850 §八):ΔB={cB} ΔD={cD} ⇒ **{v}**   | 等范数读法:{veq}")
else:
    print(f"\n主判定:⏳ 不判(分片齐={done},K4={ok4})")
