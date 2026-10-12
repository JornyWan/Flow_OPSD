#!/usr/bin/env python
"""Cert 探针分析:canonical OPSD 的 rollout 可达率。判据见 0054 §②(launch 前写定)。

用法: certP_analyze.py [--probe PATH] | --selftest
"""
import argparse, json, os, statistics as st, sys, random

O = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related"
# 0053 的**门控版**同区间参照(pickscore, opa_cert=1):前 20 updates 0.9521 -> 0.8677
GATED_REF = {"first10": 0.9521, "second10": 0.8677}
DROP_REF = GATED_REF["first10"] - GATED_REF["second10"]      # 0.0844


def rates(rows):
    """逐 epoch 的认证率(正/负分支)。"""
    by = {}
    for r in rows:
        by.setdefault(r["epoch"], []).append(r)
    out = {}
    for e, rs in by.items():
        m = rs[0]["margin"]
        out[e] = (st.mean(1.0 if x["R_plus"] >= x["R_old"] + m else 0.0 for x in rs),
                  st.mean(1.0 if x["R_minus"] <= x["R_old"] - m else 0.0 for x in rs),
                  len(rs))
    return out


def slope(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def trend(xs, ys, n_perm=5000, seed=3):
    rnd = random.Random(seed)
    s0 = slope(xs, ys)
    ys2 = list(ys)
    c = 0
    for _ in range(n_perm):
        rnd.shuffle(ys2)
        if abs(slope(xs, ys2)) >= abs(s0):
            c += 1
    p = c / n_perm
    h = len(xs) // 2
    s1, s2 = slope(xs[:h], ys[:h]), slope(xs[h:], ys[h:])
    ratio = max(abs(s1), abs(s2)) / max(1e-12, min(abs(s1), abs(s2)))
    ok = p < 0.05 and s1 * s2 > 0 and ratio < 3
    return s0, p, s1, s2, ratio, ok


def analyze(rows, label=""):
    R = rates(rows)
    eps = sorted(R)
    print(f"{label}  n={len(rows)} 条 state,epoch {eps[0]}–{eps[-1]}")
    print(f"  {'epoch':<10}{'可达率(+)':>12}{'认证率(−)':>12}{'n':>8}")
    for lo in range(0, max(eps) + 1, 10):
        ch = [e for e in eps if lo <= e < lo + 10]
        if not ch:
            continue
        ap = st.mean(R[e][0] for e in ch); am = st.mean(R[e][1] for e in ch)
        n = sum(R[e][2] for e in ch)
        print(f"  {str(lo)+'-'+str(lo+9):<10}{ap:>12.4f}{am:>12.4f}{n:>8}")

    f10 = [e for e in eps if e < 10]; s10 = [e for e in eps if 10 <= e < 20]
    if not (f10 and s10):
        print("\n  ⏳ 不足 20 epoch,无法判 C1"); return None
    a1 = st.mean(R[e][0] for e in f10); a2 = st.mean(R[e][0] for e in s10)
    drop = a1 - a2
    print(f"\n  C1: 本探针(未门控) {a1:.4f} → {a2:.4f}  降幅 {drop:+.4f}")
    print(f"      0053 门控版参照   {GATED_REF['first10']:.4f} → {GATED_REF['second10']:.4f}  降幅 {DROP_REF:+.4f}")
    frac = drop / DROP_REF if DROP_REF else float("nan")
    print(f"      降幅之比 = {frac:.2f}")
    if frac >= 0.7:
        v = "**H1 成立** —— 下降是内在的,'目标不可达'得到支持"
    elif frac <= 0.3:
        v = "**H2 成立** —— 下降主要由门控造成,**0053 §③ 的结论必须撤回**"
    else:
        v = "**介于两者之间** —— 两者都有贡献,按比例报告,不选边(0054 §② 已先写死)"
    print(f"      ⇒ {v}")

    xs = eps; ys = [R[e][0] for e in eps]
    s0, p, s1, s2, ratio, ok = trend(xs, ys)
    print(f"\n  C2 趋势: 斜率={s0:.3e}  置换 p={p:.4f}  前半={s1:.3e} 后半={s2:.3e} 量级比={ratio:.1f}")
    print(f"      ⇒ {'✅ 下降趋势成立' if ok else '❌ 三条判据未全过,不写趋势'}")
    return frac


def selftest():
    """已知答案的正反两测。

    ⚠️ 自检只验 **C1**(降幅之比 → H1/H2)。合成数据是**阶跃**(前10 恒 0.95、后10 恒 0.87),
    不是线性趋势,所以 C2 在自检里必然不过 —— 那是判据在正确地拒绝把阶跃当线性趋势,
    **不是 bug**。真实数据(门控版 10 个箱单调递减)才适用 C2。
    """
    m = 0.005
    def mk(ep, rate, n=200, seed=0):
        rnd = random.Random(seed + ep)
        out = []
        for i in range(n):
            good = rnd.random() < rate
            out.append({"epoch": ep, "kq": 8, "sigma_q": 0.278, "margin": m,
                        "R_old": 1.0, "R_plus": 1.0 + (2 * m if good else -m),
                        "R_minus": 1.0 - 2 * m})
        return out
    print("=== 自检 A(正例):构造成 0.95 -> 0.87(与门控版同幅),应判 H1 ===")
    rowsA = [r for e in range(20) for r in mk(e, 0.95 if e < 10 else 0.87)]
    fA = analyze(rowsA, "  [合成A]")
    print("\n=== 自检 B(负例):构造成 0.95 平坦不降,应判 H2 ===")
    rowsB = [r for e in range(20) for r in mk(e, 0.95)]
    fB = analyze(rowsB, "  [合成B]")
    okA, okB = (fA is not None and fA >= 0.7), (fB is not None and fB <= 0.3)
    print(f"\n  A 判 H1: {'✅' if okA else '❌'}   B 判 H2: {'✅' if okB else '❌'}")
    return okA and okB


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=f"{O}/outputs/certP_probe_open3/certP_probe.jsonl")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    d = os.path.dirname(a.probe)
    if os.path.isdir(d):
        os.listdir(d)                       # 刷 NFS
    if not os.path.exists(a.probe):
        print(f"⏳ 探针文件尚未产出: {a.probe}"); sys.exit(2)
    rows = [json.loads(l) for l in open(a.probe) if l.strip()]
    if not rows:
        print("⏳ 探针文件为空"); sys.exit(2)
    print("=" * 74); print("Cert 探针分析(判据见 0054 §②,launch 前写定)"); print("=" * 74)
    analyze(rows, "")
