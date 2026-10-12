import json, os, math, glob, re, sys
MDE = 0.0033
O = "outputs"
def per(f):
    d = json.load(open(f"{O}/{f}.json")); pi = d["per_image"]
    return [r["pickscore"]/26 + r["clipscore"] + r["hpsv2"] for r in pi], [r.get("imagereward") for r in pi]
BASE = {(s, n): per(f"matched_dflt_basemodel_none_{s}_flow{n}") for s in ("drawbench", "unseen1k") for n in (40, 4)}
def d(tag, ck, s, n):
    f = f"matched_pi_{tag}_{ck}_default_{s}_flow{n}"
    if not os.path.exists(f"{O}/{f}.json"): return None
    a, ai = per(f); b, bi = BASE[(s, n)]
    if len(a) != len(b): return None
    x = [p - q for p, q in zip(a, b)]; m = sum(x)/len(x)
    xi = [p - q for p, q in zip(ai, bi) if p is not None and q is not None]
    return m / MDE, (sum(xi)/len(xi) if xi else float("nan"))
tags = sys.argv[1].split(",")
for tag in tags:
    cks = sorted({int(m.group(1)) for f in glob.glob(f"{O}/matched_pi_{tag}_*_default_drawbench_flow40.json")
                  for m in [re.search(rf"matched_pi_{re.escape(tag)}_(\d+)_default", f)] if m})
    if not cks: print(f"{tag:14s} —"); continue
    out = []
    for ck in cks:
        c = [d(tag, ck, s, n) for s, n in (("drawbench", 40), ("unseen1k", 40), ("drawbench", 4), ("unseen1k", 4))]
        fmt = lambda v: ("%+6.1f" % v[0]) if v else "   —  "
        fir = lambda v: ("%+.3f" % v[1]) if v else "  —  "
        out.append(f"ck{ck}: o3 f40 {fmt(c[0])}/{fmt(c[1])}  f4 {fmt(c[2])}/{fmt(c[3])}  IR f40 {fir(c[0])}/{fir(c[1])}")
    print(f"{tag:14s} " + "\n               ".join(out))
