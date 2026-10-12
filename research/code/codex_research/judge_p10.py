#!/usr/bin/env python
"""P10 判定器(§4.842 §四)。数据前写好。主半径 0.1;逐 query σ。
F≫c  := mean(gain_F) ≥ 1.3·mean(gain_c) 且 配对 t(F−c) > 2
L1 拿回 ≥30% := mean(L1−c) ≥ 0.3·mean(F−c) 且 t(L1−c) > 2
H1 ∃σ:F≫c 且 L1 拿回≥30% | H2 ∃σ:F≫c 但都没拿回 | H4 ∃σ:t(F−c)<−2
H3 ∀σ:gain_F<1.1·gain_c 或 t(F−c)≤2 | H5 其余
"""
import json, sys, math, statistics as st
f = sys.argv[1] if len(sys.argv) > 1 else "outputs/p10_suffix.json"
D = json.load(open(f)); R = D["rows"]
def tpair(a, b):
    d = [x - y for x, y in zip(a, b)]; m = st.mean(d); s = st.stdev(d) if len(d) > 1 else float("nan")
    return m, (m / (s / math.sqrt(len(d))) if s and s > 0 else float("nan"))
print("n_rows=%d  prompts=%d" % (len(R), len({r["prompt_i"] for r in R})))
print("Euler 自检 rel_err:", ["%.2e" % c["rel_err_vs_step"] for c in D["checks"]])
qs = sorted({r["qt"] for r in R}); flags = []
for qt in qs:
    g = [r for r in R if r["qt"] == qt]; sq = st.mean(r["sigma_q"] for r in g)
    print("\n── query σ≈%.4f (目标 %.3f),n=%d ──" % (sq, qt, len(g)))
    for rad in ("0.05", "0.1"):
        G = {k: [r["gain_%s_%s" % (k, rad)] for r in g] for k in ("c", "L1", "F", "rand")}
        mF_c, tF_c = tpair(G["F"], G["c"]); mL_c, tL_c = tpair(G["L1"], G["c"])
        print("  r=%-4s 均值增益: c=%+.5f L1=%+.5f F=%+.5f rand=%+.5f | F−c=%+.5f(t=%+.2f) L1−c=%+.5f(t=%+.2f)"
              % (rad, st.mean(G["c"]), st.mean(G["L1"]), st.mean(G["F"]), st.mean(G["rand"]), mF_c, tF_c, mL_c, tL_c))
        if rad == "0.1":
            mc, mF = st.mean(G["c"]), st.mean(G["F"])
            Fgg = (mF >= 1.3 * mc) and tF_c > 2
            ret = (mL_c >= 0.3 * mF_c) and tL_c > 2
            weak = not ((mF < 1.1 * mc) or tF_c <= 2)
            flags.append(dict(qt=qt, Fgg=Fgg, ret=ret, neg=tF_c < -2, h3=(mF < 1.1 * mc) or tF_c <= 2))
            print("        ⇒ F≫c=%s  L1拿回≥30%%=%s  t(F−c)<−2=%s" % (Fgg, ret, tF_c < -2))
    print("  cos 中位: c·F=%+.3f  L1·F=%+.3f  c·L1=%+.3f" % tuple(st.median(r[k] for r in g) for k in ("cos_c_F", "cos_L1_F", "cos_c_L1")))
    for k in ("c", "F"):
        lo = st.mean(r["local_%s_0.1" % k] for r in g); en = st.mean(r["gain_%s_0.1" % k] for r in g)
        print("  代理→终点(r=0.1,%s):局部 %+.5f → 终点 %+.5f(保持率 %.0f%%)" % (k, lo, en, 100 * en / lo if lo else float("nan")))
if any(x["Fgg"] for x in flags):
    v = "H1" if any(x["Fgg"] and x["ret"] for x in flags) else "H2"
elif any(x["neg"] for x in flags):
    v = "H4"
elif all(x["h3"] for x in flags):
    v = "H3"
else:
    v = "H5"
print("\n主判定(§4.842 §四):**%s**   逐σ:%s" % (v, ["σ%.3f:F≫c=%s,拿回=%s" % (x["qt"], x["Fgg"], x["ret"]) for x in flags]))
