#!/usr/bin/env python
"""§4.880 判定器(数据前写定):C/M 去噪前后缀交换。
o3 逐图 = pickscore/26 + clipscore + hpsv2(与 vs_base / matched 口径同);ΔX = mean(X − Y)/MDE,t = 配对 t。显著 = ≥2.5 且 t>2。
切点选择:每个 NFE 在 drawbench 前 256 条上取 4 对 checkpoint 的 mean(MC − CC) 最大的 cut(并列取大 cut)。
主读数:unseen1k 全集、选定 cut。每对 a=MC−CC、b=MC−MM、c=CM−CC:
  S1 a 显著正 且 b 显著正(前段收益可分离,完整 MSP 的后段把它吃掉一部分)| S2 a 显著正、b 不显著正(前段即全部)
  S3 a 非显著正 且 c 显著正(收益在 MSP 的后段)| S0 其余(无可利用)。
聚合:≥3/4 对同格 ⇒ 该格;否则 Sx。NFE 4 与 40 分别判。另报 IR(imagereward)同号与否。"""
import glob, json, math, os
MDE = 0.0033
PAIRS = {"s1_100": ("o3canon300_100", "o3msp300_100"), "s1_300": ("o3canon300_300", "o3msp300_300"),
         "s2_100": ("canon300s2_100", "msp300s2_100"), "s2_300": ("canon300s2_300", "msp300s2_300")}
CUTS = ("0.9", "0.7", "0.45")
o3 = lambda r, i: r["pickscore"][i] / 26 + r["clipscore"][i] + r["hpsv2"][i]

def ref(tag, s, n):
    d = json.load(open(f"outputs/matched_pi_{tag}_default_{s}_flow{n}.json"))["per_image"]
    return {k: [x[k] for x in d] for k in ("pickscore", "clipscore", "hpsv2", "imagereward")}

def swap(pair, s):
    cells, idx = {}, []
    for f in sorted(glob.glob(f"outputs/p15/{pair}_{s}_b*.json")):
        d = json.load(open(f)); lo, hi = d["meta"]["index_range"]; idx += list(range(lo, hi))
        for k, v in d["cells"].items():
            c = cells.setdefault(k, {r: {} for r in v["scores"]})
            for r, col in v["scores"].items():
                for j, x in enumerate(col): c[r][lo + j] = x
    return cells, sorted(idx)

def diff(A, B, ia, ib, key=o3):
    x = [key(A, i) - key(B, j) for i, j in zip(ia, ib)]
    n = len(x); m = sum(x) / n; sd = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1))
    return m / MDE, m / (sd / math.sqrt(n)), n

def as_lists(c, idx):        # 交换格 dict(index→值) → 与 idx 对齐的列表
    return {r: [c[r][i] for i in idx] for r in c}

sp = lambda v: v and v[0] >= 2.5 and v[1] > 2
f = lambda v: "%+6.2f(t%+5.1f)" % v[:2] if v else "   ⏳        "
ir = lambda A, i: A["imagereward"][i]
for n in (4, 40):
    # 切点选择(drawbench 前 256)
    score = {}
    for cut in CUTS:
        vals = []
        for p, (tc, tm) in PAIRS.items():
            cells, idx = swap(p, "drawbench"); k = f"MC|{cut}|{n}"
            if k not in cells or len(idx) < 256: vals = None; break
            C = ref(tc, "drawbench", n); MC = as_lists(cells[k], idx)
            vals.append(diff(MC, C, range(len(idx)), idx)[0])
        score[cut] = None if vals is None else sum(vals) / len(vals)
    if any(v is None for v in score.values()):
        print(f"NFE={n}: ⏳ drawbench 选切点数据未齐"); continue
    cut = max(CUTS, key=lambda c: (score[c], float(c)))
    print(f"\nNFE={n}: 切点选择(drawbench-256 上 mean(MC−CC))" + "  ".join(f"cut {c}: {score[c]:+.2f}" for c in CUTS) + f"  ⇒ **cut={cut}**")
    verdicts = []
    for p, (tc, tm) in PAIRS.items():
        cells, idx = swap(p, "unseen1k")
        need = (f"MC|{cut}|{n}", f"CM|{cut}|{n}")
        if any(k not in cells for k in need) or len(idx) < 1000:
            print(f"  {p}: ⏳ unseen1k 未齐({len(idx)}/1000)"); verdicts.append(None); continue
        C, M = ref(tc, "unseen1k", n), ref(tm, "unseen1k", n)
        MC, CM = as_lists(cells[need[0]], idx), as_lists(cells[need[1]], idx)
        r = range(len(idx))
        a, b, c, mm = diff(MC, C, r, idx), diff(MC, M, r, idx), diff(CM, C, r, idx), diff(M, C, idx, idx)
        ia, ic = diff(MC, C, r, idx, ir), diff(CM, C, r, idx, ir)
        cell = "S1" if sp(a) and sp(b) else "S2" if sp(a) else "S3" if sp(c) else "S0"
        verdicts.append(cell)
        print(f"  {p}: MM−CC {f(mm)} | a=MC−CC {f(a)} | b=MC−MM {f(b)} | c=CM−CC {f(c)} | IR(MC−CC) {ia[0]*MDE:+.3f} IR(CM−CC) {ic[0]*MDE:+.3f} ⇒ {cell}")
    if None in verdicts: print(f"  主判定 NFE={n}:⏳"); continue
    top = max(set(verdicts), key=verdicts.count)
    print(f"  主判定(§4.880)NFE={n}:**{top if verdicts.count(top) >= 3 else 'Sx 混合'}**({verdicts})")
