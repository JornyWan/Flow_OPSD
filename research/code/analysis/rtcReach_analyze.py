#!/usr/bin/env python3
"""RTC 可达性判读。判据见 codex_research/0067 §③(经 §⑦d 修订),先写后测。

ratio = R(g)/E_u[R(u)],R(d)=‖Jᵀd‖²/‖d‖²。
  ratio > 1 ⇒ reward 方向比随机方向**更**可达;< 1 ⇒ 更不可达。
  J 整体收缩时分子分母同步 ⇒ ratio 不变(零模型内建)。

自检: python3 rtcReach_analyze.py --selftest
"""
import argparse, io, json, os, random, statistics as st, sys


def load(p):
    return [json.loads(l) for l in io.open(p) if l.strip()]


def slope(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def trend(ys, nperm=20000, seed=0):
    """项目既定三判据:置换 p<0.05 且 前后半同号 且 幅度比<3。"""
    rng = random.Random(seed)
    xs = list(range(len(ys)))
    b = slope(xs, ys)
    c = sum(1 for _ in range(nperm) if abs(slope(xs, rng.sample(ys, len(ys)))) >= abs(b))
    h = len(ys) // 2
    b1, b2 = slope(xs[:h], ys[:h]), slope(xs[h:], ys[h:])
    same = (b1 > 0) == (b2 > 0)
    ratio = max(abs(b1), abs(b2)) / max(min(abs(b1), abs(b2)), 1e-12)
    return dict(slope=b, p=c / nperm, b1=b1, b2=b2, same=same, ratio=ratio,
                ok=(c / nperm < 0.05 and same and ratio < 3))


def report(path):
    R = load(path)
    if not R:
        print(f"  (无记录: {path})"); return
    # [[print-the-series-first]] 序列先于汇总
    print("  逐 epoch 原始序列(先看这个):")
    print("    ep    : " + " ".join(f"{r['epoch']:>7}" for r in R))
    print("    ratio : " + " ".join(f"{r['ratio']:>7.3f}" for r in R))
    print("    R(g)  : " + " ".join(f"{r['R_g']:>7.1e}" for r in R))
    print("    E[R(u)]: " + " ".join(f"{r['R_u_mean']:>6.1e}" for r in R))
    if any("ratio_perm" in r for r in R):
        print("    ratio_perm(**主判据**,零模型=batch 移位的真实 reward 梯度):")
        print("           " + " ".join(f"{r.get('ratio_perm', float('nan')):>7.3f}" for r in R))
    else:
        print("    ⚠️ 无 ratio_perm ⇒ 这批数据只有**各向同性**零模型,按 0067 §⑨ 不足以判读 RTC 命题。")
    if any("ratio_R1" in r for r in R):
        print("    ratio_R1(0067 §⑩ 判别 A/B:单样本,和里无跨样本项):")
        print("           " + " ".join(f"{r.get('ratio_R1', float('nan')):>7.3f}" for r in R))
        import statistics as _st
        _rp = [r["ratio_perm"] for r in R[1:] if "ratio_perm" in r]
        _r1 = [r["ratio_R1"] for r in R[1:] if "ratio_R1" in r]
        # 纪律修补 2026-09-01:A/B 判词此前打在 C-d(n≥25)之前,n=12 就下了结论。
        # 与 C-a/C-b 同标准:样本不足一律拒绝打判词,只显示序列。
        if len(R) < 25:
            print(f"    ⚠️ n={len(R)}<25 ⇒ **A/B 归因拒绝打判词**(只显示上面的序列)")
        elif len(_r1) >= 3 and len(_rp) >= 3:
            m1, mp = _st.median(_r1), _st.median(_rp)
            print(f"    去 epoch0 后中位: ratio_R1={m1:.3f}  ratio_perm={mp:.3f}")
            if m1 < 0.7:
                print("    ⇒ 单样本也 ≪1 ⇒ **机制 (A) 逐样本衰减**(RTC 命题的尖锐版)")
            elif m1 > 0.85:
                print("    ⇒ 单样本≈1 而整批 ≪1 ⇒ **机制 (B) 跨样本抵消**(冲突的加强版)")
            else:
                print("    ⇒ 单样本居中(0.7~0.85)⇒ **两种机制并存,不可单归一因**")
    print("    ⚠️ epoch 0 是 LoRA 初始化瞬态,单独看待。")
    n = len(R)
    rt = [r["ratio"] for r in R]
    print(f"\n  n_updates={n}  K={R[0].get('K','?')}")
    # C-e:分母的相对标准误
    se = [r["R_u_sd"] / (r["K"] ** 0.5) / r["R_u_mean"] for r in R if r.get("K") and r["R_u_mean"] > 0]
    if se:
        print(f"  C-e 分母相对标准误 SE/均值 中位={st.median(se):.1%}  "
              + ("✅ <20%" if st.median(se) < 0.20 else "❌ ≥20%,K 不足"))
    # §4.100 预写:LoRA 从零长起的"苏醒期"必须与真实趋势分开。
    # E[R(u)] 在苏醒期单调爬升(实测 ~5 倍),会把任何比值的趋势染色。
    ru = [r["R_u_mean"] for r in R]
    warm = 0
    for i in range(1, len(ru)):
        if ru[i] > ru[i - 1]:
            warm = i
        else:
            break
    print(f"  苏醒期识别:E[R(u)] 从 ep0 起单调上升到 ep{warm}(共 {warm+1} 点)"
          f" ⇒ 趋势检验将**同时**报告 全段 / 去苏醒期 两个口径")
    if n < 25:
        print(f"  ⚠️ C-d 不过(n={n}<25)—— **拒绝打判词**"); return
    tr = trend(rt)
    print(f"  C-a 趋势: 斜率={tr['slope']:+.4f}/ep  置换p={tr['p']:.4f}"
          f"{'✅' if tr['p']<0.05 else '❌'}  前后半 {tr['b1']:+.4f}/{tr['b2']:+.4f}"
          f" 同号={'✅' if tr['same'] else '❌'} 幅度比={tr['ratio']:.1f}{'✅' if tr['ratio']<3 else '❌'}"
          f"  ⇒ {'成立' if tr['ok'] else '**不成立**'}")
    # 去苏醒期后重做 C-a(阈值不变,只换样本段)
    cut = max(warm + 1, 1)
    if n - cut >= 8:
        rt2 = rt[cut:]
        tr2 = trend(rt2)
        print(f"  C-a(去苏醒期,从 ep{cut} 起,n={len(rt2)}): 斜率={tr2['slope']:+.4f}/ep"
              f"  置换p={tr2['p']:.4f}{'✅' if tr2['p']<0.05 else '❌'}"
              f"  前后半 {tr2['b1']:+.4f}/{tr2['b2']:+.4f} 同号={'✅' if tr2['same'] else '❌'}"
              f"  ⇒ {'成立' if tr2['ok'] else '**不成立**'}")
        if tr["ok"] != tr2["ok"]:
            print("    ⚠️ **两个口径判定不一致** ⇒ 趋势由苏醒期驱动,不得声称训练期趋势")
    else:
        print(f"  C-a(去苏醒期): 剩 {n-cut} 点 <8,**不做**(样本不足)")
    D = rt[0] / rt[-1] if rt[-1] else float("inf")
    lvl = "主要成因(D≥3)" if D >= 3 else ("部分成因(1.5≤D<3)" if D >= 1.5 else "**不是成因(D<1.5)**")
    print(f"  C-b 幅度: 首={rt[0]:.3f} 末={rt[-1]:.3f}  D={D:.2f} ⇒ {lvl}")
    # C-c:分子分母各自趋势
    tg, tu = trend([r["R_g"] for r in R]), trend([r["R_u_mean"] for r in R])
    print(f"  C-c 分量: R(g) 趋势 p={tg['p']:.4f}{'(降)' if tg['slope']<0 else '(升)'}"
          f"   E[R(u)] 趋势 p={tu['p']:.4f}{'(降)' if tu['slope']<0 else '(升)'}")
    if tg["ok"] and tu["ok"] and not tr["ok"]:
        print("    ⇒ 两分量同步变化而 ratio 平 ⇒ **是 J 整体收缩,不是目标不可达**(0067 §④b)")
    elif tr["ok"] and tr["slope"] < 0:
        print("    ⇒ ratio 显著下降 ⇒ **支持 RTC 命题**(0067 §④c)")
    else:
        print("    ⇒ 无趋势 ⇒ 可达性不是成因(0067 §④a)")


def selftest():
    """合成三种情形,验证判词正确。"""
    print("=" * 68); print("rtcReach 分析器自检(合成已知情形)"); print("=" * 68)
    rng = random.Random(1); ok = True

    def mk(ratio_fn, n=30, noise=0.05):
        return [{"epoch": i, "upd": i, "R_g": ratio_fn(i) * 1e-3, "R_u_mean": 1e-3,
                 "R_u_sd": 1e-4, "K": 8,
                 "ratio": ratio_fn(i) * rng.uniform(1 - noise, 1 + noise)} for i in range(n)]

    cases = [
        ("ratio 显著下降(RTC 命题成立)", lambda i: 1.0 - 0.025 * i, "支持 RTC 命题"),
        ("ratio 持平(可达性非成因)",     lambda i: 1.0,             "无趋势"),
    ]
    for lbl, fn, want in cases:
        print(f"\n  --- {lbl} ---")
        import io as _io, contextlib
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            p = "/tmp/_rt.jsonl"
            import tempfile
            fd = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
            for r in mk(fn): fd.write(json.dumps(r) + "\n")
            fd.close(); report(fd.name); os.unlink(fd.name)
        out = buf.getvalue()
        hit = want in out
        ok &= hit
        print("   " + [l for l in out.splitlines() if "⇒ **" in l or "⇒ 支持" in l or "⇒ 无趋势" in l][-1].strip())
        print(f"   期望包含 {want!r}: {'✅' if hit else '❌'}")

    print("\n  --- 反向:n<25 必须拒绝打判词 ---")
    import tempfile, contextlib, io as _io
    fd = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for r in mk(lambda i: 1.0 - 0.025 * i, n=10): fd.write(json.dumps(r) + "\n")
    fd.close()
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf): report(fd.name)
    os.unlink(fd.name)
    hit = "拒绝打判词" in buf.getvalue()
    ok &= hit
    print(f"   {'✅ 正确拒绝' if hit else '❌ 竟然下了结论'}")
    print("\n" + ("  ✅ 自检全部通过" if ok else "  ❌ 自检未通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    p = a.path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "outputs", "rtcReach_probe", "rtcReach_probe.jsonl")
    if not os.path.exists(p): print(f"  (尚无数据: {p})"); sys.exit(0)
    report(p)
