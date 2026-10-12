#!/usr/bin/env python
"""Gate Q1b 分析:多步长回溯能否修好尾部。判据见 0061 §⑤(先写后测)。
用法: gateQ1b_analyze.py [--probe P] | --selftest"""
import argparse, json, os, statistics as st, sys, random
O = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related"
K3 = ("pickscore", "clipscore", "hpsv2")
ALPHAS = (1.0, 0.5, 0.25, 0.125)          # 从大到小
TH = {"a_neg": 0.05, "c_mean": 0.15, "d_noop": 0.15}   # 0061 §⑤,不得事后修改
TAU = 0.0                                  # 0061 §⑤ 补记,判定只认 τ=0


def _g(row, d, k, alpha=None):
    """方向 d 在步长 alpha 下第 k 轴的 exact-suffix 增益。"""
    if alpha is None or alpha == 1.0:
        v = row["dir"][d][k]
    else:
        v = row["q_alpha"][str(alpha)][k]
    return v - row["r0_suffix"][k]


def backtrack(row, tau=TAU):
    """取最大可行 α;无可行则 no-op。返回 (alpha 或 None, 三轴和增益)。"""
    for a in ALPHAS:
        if a != 1.0 and "q_alpha" not in row:
            continue
        per = [_g(row, "q", k, a) for k in K3]
        if all(v >= -tau for v in per) and sum(per) > 0:
            return a, sum(per)
    return None, 0.0                       # no-op:不动,增益 0


def analyze(rows, label=""):
    n = len(rows)
    clean = [sum(_g(r, "clean", k) for k in K3) for r in rows]
    q1 = [sum(_g(r, "q", k) for k in K3) for r in rows]
    bt = [backtrack(r) for r in rows]
    qb = [g for _, g in bt]
    noop = sum(1 for a, _ in bt if a is None) / n
    pc = lambda v, q: sorted(v)[min(int(q * len(v)), len(v) - 1)]
    print(f"{label}  n={n}")
    print(f"  {'方案':<16}{'均值':>11}{'p10':>11}{'负增益率':>11}")
    for nm, v in (("clean(canonical)", clean), ("q  α=1 固定", q1), ("q  回溯(Q1b)", qb)):
        print(f"  {nm:<16}{st.mean(v):>11.5f}{pc(v,.10):>11.5f}{sum(1 for x in v if x<0)/len(v):>10.1%}")
    negb = sum(1 for x in qb if x < 0) / n
    p10b = pc(qb, .10)
    relb = (st.mean(qb) - st.mean(clean)) / abs(st.mean(clean))
    ok = {"a": negb <= TH["a_neg"], "b": p10b > 0,
          "c": relb >= TH["c_mean"], "d": noop <= TH["d_noop"]}
    print()
    print(f"  Q1b-a 回溯后负增益率 = {negb:.1%}   (阈值 ≤{TH['a_neg']:.0%})  {'✅' if ok['a'] else '❌'}")
    print(f"  Q1b-b 回溯后 p10     = {p10b:+.5f}  (需 >0)          {'✅' if ok['b'] else '❌'}")
    print(f"  Q1b-c 均值相对 clean = {relb:+.1%}  (阈值 ≥{TH['c_mean']:.0%})  {'✅' if ok['c'] else '❌'}")
    print(f"  Q1b-d no-op 率       = {noop:.1%}   (阈值 ≤{TH['d_noop']:.0%})  {'✅' if ok['d'] else '❌'}")
    cnt = {}
    for a, _ in bt:
        cnt[a] = cnt.get(a, 0) + 1
    print("\n  选中的 α 分布: " + ", ".join(
        f"{'no-op' if k is None else k}={v}({v/n:.0%})" for k, v in
        sorted(cnt.items(), key=lambda x: (x[0] is None, -(x[0] or 0)))))
    print("\n  τ 敏感性(仅供参考,判定只认 τ=0):")
    for tau in (0.0, 0.002, 0.005):
        b2 = [backtrack(r, tau) for r in rows]
        g2 = [g for _, g in b2]
        no2 = sum(1 for a, _ in b2 if a is None) / n
        print(f"    τ={tau:<6} 负增益率={sum(1 for x in g2 if x<0)/n:6.1%}  "
              f"no-op={no2:6.1%}  均值={st.mean(g2):+.5f}")
    allok = all(ok.values())
    print(f"\n  ⇒ Q1b {'全部通过 —— 回溯确实修好了尾部,可进 Gate Q2' if allok else '未全过'}")
    return allok


def selftest():
    def mk(n, tail_frac, rescuable, seed=0):
        rnd = random.Random(seed); rows = []
        for i in range(n):
            r0 = {k: 1.0 for k in K3}
            bad = i < int(n * tail_frac)
            base = -0.02 if bad else 0.03
            row = {"epoch": 0, "r0_suffix": r0,
                   "dir": {"clean": {k: 1.0 + base * 0.6 for k in K3},
                           "q": {k: 1.0 + base for k in K3},
                           "rand": {k: 1.0 - 0.02 for k in K3}},
                   "q_alpha": {}}
            for a in (0.5, 0.25, 0.125):
                # 可救:缩步后转正;不可救:一直为负
                v = (0.01 if (bad and rescuable) else base * a)
                row["q_alpha"][str(a)] = {k: 1.0 + v for k in K3}
            rows.append(row)
        return rows
    print("=== 自检A(正例):20% 尾部,缩步后可救 ⇒ 应通过 ===")
    okA = analyze(mk(500, .20, True, 1), "  [合成A]")
    print("\n=== 自检B(负例):40% 尾部,缩步也救不回 ⇒ no-op 率过高,应不通过 ===")
    okB = analyze(mk(500, .40, False, 2), "  [合成B]")
    print(f"\n  A 通过: {'✅' if okA else '❌'}   B 不通过: {'✅' if not okB else '❌'}")
    return okA and not okB


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=f"{O}/outputs/gateQ1b_probe/gateQ1_probe.jsonl")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(0 if selftest() else 1)
    d = os.path.dirname(a.probe)
    if os.path.isdir(d): os.listdir(d)
    if not os.path.exists(a.probe): print(f"⏳ 尚未产出: {a.probe}"); sys.exit(2)
    rows = [json.loads(l) for l in open(a.probe) if l.strip()]
    rows = [r for r in rows if "q_alpha" in r]
    if not rows: print("⏳ 无 q_alpha 数据"); sys.exit(2)
    print("=" * 72); print("Gate Q1b 分析(判据见 0061 §⑤,先写后测)"); print("=" * 72)
    analyze(rows, "")
