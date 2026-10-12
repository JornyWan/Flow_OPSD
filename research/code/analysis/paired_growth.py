#!/usr/bin/env python3
"""官方口径(default/EMA, DrawBench, flow-40)逐图配对比较。
不能只看 exit=0:先核 per_image 条数与 prompt 对齐,再做配对检验。"""
import json, os, math, sys, statistics as st
O = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs"
AXES = ("hpsv2", "clipscore", "aesthetic", "pickscore", "imagereward")

def load(tag):
    p = os.path.join(O, f"matched_pi_{tag}_default_drawbench_flow40.json")
    if not os.path.exists(p):
        return None, f"缺文件 {os.path.basename(p)}"
    d = json.load(open(p))
    pi = d.get("per_image") or []
    if len(pi) < 900:
        return None, f"per_image 只有 {len(pi)} 条"
    rows = {}
    for r in pi:
        i = r.get("index")
        if i is None or not all(a in r for a in ("pickscore", "clipscore", "hpsv2")):
            continue
        rows[i] = r
    if len(rows) < 900:
        return None, f"完整记录只有 {len(rows)} 条"
    return rows, None

def obj(r):
    return r["pickscore"] / 26.0 + r["clipscore"] + r["hpsv2"]

def paired(a_tag, b_tag, axis=None):
    """b − a(b 是实验臂)"""
    A, ea = load(a_tag)
    B, eb = load(b_tag)
    if ea: return None, f"{a_tag}: {ea}"
    if eb: return None, f"{b_tag}: {eb}"
    keys = sorted(set(A) & set(B))
    if len(keys) < 900:
        return None, f"共同 prompt 只有 {len(keys)} 条"
    f = (lambda r: r[axis]) if axis else obj
    d = [f(B[k]) - f(A[k]) for k in keys]
    n = len(d); m = st.mean(d)
    sd = st.pstdev(d) * math.sqrt(n / (n - 1))
    se = sd / math.sqrt(n)
    t = m / se if se > 0 else float("nan")
    win = sum(1 for x in d if x > 0) / n
    return dict(n=n, delta=m, se=se, t=t, win=win), None

if __name__ == "__main__":
    base, arm = sys.argv[1], sys.argv[2]
    r, err = paired(base, arm)
    if err: print(f"  ❌ {err}"); sys.exit(1)
    sig = "✅ 显著" if abs(r["delta"]) > 0.0033 and abs(r["t"]) > 2 else "· 未过 MDE"
    print(f"  {arm} − {base}:  Δ={r['delta']:+.4f}  t={r['t']:+.2f}  胜率={r['win']:.1%}  n={r['n']}  {sig}")
    for ax in AXES:
        ra, e = paired(base, arm, ax)
        if ra: print(f"     {ax:<12} Δ={ra['delta']:+.5f}  t={ra['t']:+7.2f}  胜率={ra['win']:.1%}")
