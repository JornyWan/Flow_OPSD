#!/usr/bin/env python
"""Gate Q1 分析:suffix action-value 梯度是否修复 local→suffix 落差。
判据见 0059 §③(launch 前写定)。用法: gateQ1_analyze.py [--probe P] | --selftest"""
import argparse, json, os, statistics as st, sys, random

O = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related"
K3 = ("pickscore", "clipscore", "hpsv2")
TH = {"a_neg_drop": 0.50, "c_mean_gain": 0.15}     # 0059 §③,不得事后修改


def gains(rows, d):
    """每条 state 上,方向 d 经 exact suffix 的三轴和增益(相对锚点 F_old(s,y0))。"""
    return [sum(r["dir"][d][k] - r["r0_suffix"][k] for k in K3) for r in rows if d in r["dir"]]


def per_axis(rows, d, k):
    return [r["dir"][d][k] - r["r0_suffix"][k] for r in rows if d in r["dir"]]


def pc(v, q):
    v = sorted(v); return v[min(int(q * len(v)), len(v) - 1)]


def analyze(rows, label=""):
    n = len(rows)
    G = {d: gains(rows, d) for d in ("clean", "q", "rand")}
    print(f"{label}  n={n} 条 state")
    print(f"  {'方向':<8}{'均值增益':>12}{'中位':>12}{'p10':>12}{'负增益率':>12}")
    neg = {}
    for d in ("clean", "q", "rand"):
        if not G[d]: continue
        neg[d] = sum(1 for v in G[d] if v < 0) / len(G[d])
        print(f"  {d:<8}{st.mean(G[d]):>12.5f}{pc(G[d],.5):>12.5f}{pc(G[d],.10):>12.5f}{neg[d]:>11.1%}")

    ok = {}
    # Q1-a 负增益率相对 clean 降低 ≥50%
    if neg.get("clean", 0) > 0:
        drop = (neg["clean"] - neg["q"]) / neg["clean"]
    else:
        drop = float("nan")
    ok["a"] = drop >= TH["a_neg_drop"]
    print(f"\n  Q1-a 负增益率 {neg.get('clean',float('nan')):.1%} → {neg.get('q',float('nan')):.1%}"
          f"  降低 {drop:+.1%}  (阈值 ≥{TH['a_neg_drop']:.0%})  {'✅' if ok['a'] else '❌'}")
    # Q1-b p10 由负/近零转正
    p10c, p10q = pc(G["clean"], .10), pc(G["q"], .10)
    ok["b"] = (p10c <= 0) and (p10q > 0)
    print(f"  Q1-b p10  {p10c:+.5f} → {p10q:+.5f}   (需由 ≤0 变 >0)  {'✅' if ok['b'] else '❌'}")
    # Q1-c 均值增益提高 ≥15%
    mc, mq = st.mean(G["clean"]), st.mean(G["q"])
    rel = (mq - mc) / abs(mc) if mc else float("nan")
    ok["c"] = rel >= TH["c_mean_gain"]
    print(f"  Q1-c 均值增益 {mc:.5f} → {mq:.5f}  提高 {rel:+.1%}  (阈值 ≥{TH['c_mean_gain']:.0%})  {'✅' if ok['c'] else '❌'}")
    # Q1-d 三轴无系统性显著下降(配对 t)
    bad = []
    for k in K3:
        a, b = per_axis(rows, "clean", k), per_axis(rows, "q", k)
        d_ = [y - x for x, y in zip(a, b)]
        t = st.mean(d_) / (st.stdev(d_) / len(d_) ** .5) if len(d_) > 2 and st.stdev(d_) > 0 else 0.0
        flag = "↓显著" if t < -2.5 else ("↑显著" if t > 2.5 else "打平")
        if t < -2.5: bad.append(k)
        print(f"  Q1-d {k:<10} q−clean 配对 t={t:+6.2f}  {flag}")
    ok["d"] = not bad
    print(f"       => {'✅ 无系统性下降' if ok['d'] else '❌ 下降轴: '+','.join(bad)}")

    cos = [r["cos_clean_q"] for r in rows if "cos_clean_q" in r]
    if cos:
        print(f"\n  cos(d_clean, d_Q) 中位 = {st.median(cos):.4f}  范围[{min(cos):.3f},{max(cos):.3f}]")
        print("     ⚠️ 若接近 1 ⇒ 后缀 Jacobian 近似恒等,Q-OPSD 退化为 canonical(0059 §④a)")
    gn = [r["gnorm"] for r in rows if "gnorm" in r]
    if gn:
        nan = sum(1 for g in gn if any(v != v for v in g.values()))
        print(f"  梯度范数 clean 中位={st.median([g['clean'] for g in gn]):.3e}  "
              f"q 中位={st.median([g['q'] for g in gn]):.3e}  NaN 条数={nan}")

    allok = all(ok.values())
    print(f"\n  ⇒ Q1-a~d {'全部通过 —— 可跑 Q1-e(换 seed)' if allok else '未全过 ⇒ 按 0059 §③ 终止 Q-OPSD,不调 rho 复活'}")
    return allok, {"drop": drop, "rel": rel, "p10q": p10q}


def _mk(n, cln, q, rnd, seed=0):
    rnd_ = random.Random(seed); rows = []
    for i in range(n):
        r0 = {k: 1.0 for k in K3}
        def mk(mu):
            return {k: 1.0 + rnd_.gauss(mu / 3, 0.02) for k in K3}
        rows.append({"epoch": i // 50, "kq": 8, "sigma_q": .278,
                     "cos_clean_q": rnd_.uniform(.2, .6),
                     "gnorm": {"clean": 1.0, "q": 1.2},
                     "r0_suffix": r0,
                     "dir": {"clean": mk(cln), "q": mk(q), "rand": mk(rnd)}})
    return rows


def selftest():
    print("=== 自检A(正例):构造 d_Q 明显优于 d_clean,应判通过 ===")
    okA, _ = analyze(_mk(400, cln=0.02, q=0.09, rnd=0.0, seed=1), "  [合成A]")
    print("\n=== 自检B(负例):构造 d_Q ≈ d_clean,应判不通过 ===")
    okB, _ = analyze(_mk(400, cln=0.05, q=0.051, rnd=0.0, seed=2), "  [合成B]")
    print(f"\n  A 判通过: {'✅' if okA else '❌'}    B 判不通过: {'✅' if not okB else '❌'}")
    return okA and (not okB)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=f"{O}/outputs/gateQ1_probe/gateQ1_probe.jsonl")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(0 if selftest() else 1)
    d = os.path.dirname(a.probe)
    if os.path.isdir(d): os.listdir(d)
    if not os.path.exists(a.probe): print(f"⏳ 尚未产出: {a.probe}"); sys.exit(2)
    rows = [json.loads(l) for l in open(a.probe) if l.strip()]
    if not rows: print("⏳ 文件为空"); sys.exit(2)
    print("=" * 76); print("Gate Q1 分析(判据见 0059 §③,launch 前写定)"); print("=" * 76)
    analyze(rows, "")
