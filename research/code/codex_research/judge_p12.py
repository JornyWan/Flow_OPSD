#!/usr/bin/env python
"""§4.866 S0 判定器(数据前写定)。用法:judge_p12.py <tag>(canon100 | msp100)
主读数:σ≈0.858 与 0.602、r=0.10;G_X = 逐 prompt 增益均值,配对 t(n 个 prompt)。
PASS:两态都 tpd8−msp>0 且 t>2、tpd8−grad>0 且 t>2、tpd8 增益为正的比例 ≥70% | TIE:两态 |t(tpd8−msp)|<2
FAIL:任一态 tpd8−msp<0 且 t<−2 | MIXED:其余。"""
import glob, json, math, sys
tag = sys.argv[1] if len(sys.argv) > 1 else "canon100"
rows = []
fs = sorted(glob.glob(f"outputs/p12/{tag}_sh*of*.json"))
for f in fs: rows += json.load(open(f))["rows"]
if not rows: sys.exit(f"没有 outputs/p12/{tag}_sh*.json")
nsh = max(int(f.split("of")[-1].split(".")[0]) for f in fs)
print(f"[{tag}] 分片 {len(fs)}/{nsh} | 行 {len(rows)}" + ("" if len(fs) == nsh else "  ⚠️ 未齐,不判"))
DIRS = ("grad", "msp", "cons", "tpd8", "tpd4", "tpd8f", "tpd8h", "rand")
def st(v):
    n = len(v); m = sum(v) / n; sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else float("nan")
    return m, (m / (sd / math.sqrt(n)) if sd and sd > 0 else float("nan")), n
by = {}
for r in rows: by.setdefault(r["state"], []).append(r)
res = {}
for s in sorted(by, reverse=True):
    R = sorted(by[s], key=lambda r: r["prompt_i"])
    print(f"\nσ≈{s}(i={R[0]['i']},σ={R[0]['sigma']:.3f},n={len(R)})  宽度 {sum(r['width'] for r in R)/len(R):.3f}  "
          f"‖x̄−y0‖/‖y0‖ {sum(r['mean_vs_y0'] for r in R)/len(R):.3f}")
    for rad in ("0.05", "0.1"):
        print(f"  r={rad}: " + "  ".join("%s %+.4f" % (k, st([r[f'gain_{k}_{rad}'] for r in R])[0]) for k in DIRS))
    for a_, b_ in (("tpd8", "msp"), ("tpd8", "grad"), ("tpd8", "tpd8h"), ("tpd8", "tpd4"), ("tpd8", "tpd8f"), ("msp", "grad"), ("msp", "cons")):
        m, t, n = st([r[f"gain_{a_}_0.1"] - r[f"gain_{b_}_0.1"] for r in R])
        print(f"    {a_} − {b_:6s} (r=0.1) {m:+.4f}  t={t:+.2f}")
    pos = sum(r["gain_tpd8_0.1"] > 0 for r in R) / len(R)
    res[s] = dict(tm=st([r["gain_tpd8_0.1"] - r["gain_msp_0.1"] for r in R]),
                  tg=st([r["gain_tpd8_0.1"] - r["gain_grad_0.1"] for r in R]), pos=pos)
    print(f"    tpd8 增益为正的比例 {pos:.2f};cos(tpd8,msp) {sum(r['cos']['tpd8|msp'] for r in R)/len(R):+.3f}  "
          f"cos(tpd8,grad) {sum(r['cos']['tpd8|grad'] for r in R)/len(R):+.3f}")
keys = [s for s in res if abs(s - 0.858) < 1e-6 or abs(s - 0.602) < 1e-6]
if len(fs) < nsh or len(keys) < 2:
    print("\n主判定:⏳ 不判"); sys.exit(0)
if all(res[s]["tm"][0] > 0 and res[s]["tm"][1] > 2 and res[s]["tg"][0] > 0 and res[s]["tg"][1] > 2 and res[s]["pos"] >= 0.7 for s in keys):
    v = "PASS(进 S1)"
elif any(res[s]["tm"][0] < 0 and res[s]["tm"][1] < -2 for s in keys): v = "FAIL(这一形式的 TPD 停)"
elif all(abs(res[s]["tm"][1]) < 2 for s in keys): v = "TIE(S1 降优先级)"
else: v = "MIXED(报用户)"
print(f"\n主判定(§4.866 S0,{tag}):**{v}**")
