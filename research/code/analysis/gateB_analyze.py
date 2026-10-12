#!/usr/bin/env python3
"""Gate B 分析:半批梯度噪声尺度 / 临界批量。判据见 codex_research/0064 §③(先写后测)。

B_noise = B·(‖ḡ_full‖² − ḡ_A·ḡ_B)/(ḡ_A·ḡ_B),B = bs×world×accum。
自检:python3 gateB_analyze.py --selftest   (合成已知真值,双向验证)
"""
import argparse, io, json, os, sys
import numpy as np


def load(path):
    recs = []
    with io.open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def bnoise_series(recs):
    """返回 (B_eff, num[], den[])。num/den 分开保留,聚合时用比值-和(与功效分析同口径)。"""
    if not recs:
        return None, np.array([]), np.array([])
    B = int(recs[0]["bs"]) * int(recs[0]["world"]) * int(recs[0]["accum"])
    for r in recs:
        b = int(r["bs"]) * int(r["world"]) * int(r["accum"])
        if b != B:
            raise SystemExit(f"[gateB] B_eff 在记录间不一致({B} vs {b}) —— 拒绝聚合")
    num = np.array([B * (r["nrm_full2"] - r["dot_ab"]) for r in recs], float)
    den = np.array([r["dot_ab"] for r in recs], float)
    return B, num, den


def boot_ci(num, den, nboot=4000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(num)
    idx = rng.integers(0, n, size=(nboot, n))
    r = num[idx].sum(1) / den[idx].sum(1)
    lo, md, hi = np.percentile(r, [2.5, 50, 97.5])
    return float(lo), float(md), float(hi)


def verdict(B, num, den, quiet=False):
    """返回 (判词, 详情dict)。阈值全部来自 0064 §③,不在此处调整。"""
    n = len(num)
    d = {"n": n, "B_eff": B}
    pos = float((den > 0).mean()) if n else 0.0
    d["dot_pos_rate"] = pos
    lo, md, hi = boot_ci(num, den) if n >= 2 else (float("nan"),) * 3
    d.update(ci_lo=lo, med=md, ci_hi=hi)
    d["width_ratio"] = hi / lo if lo > 0 else float("inf")

    # 前半/后半漂移(B-c)
    if n >= 4:
        h = n // 2
        d["med_first"] = boot_ci(num[:h], den[:h])[1]
        d["med_second"] = boot_ci(num[h:], den[h:])[1]
        d["drift"] = d["med_second"] / d["med_first"] if d["med_first"] > 0 else float("inf")
    else:
        d["med_first"] = d["med_second"] = d["drift"] = float("nan")

    # B-d 健全性(report() 会先拦一次;此处保证 verdict 单独调用也正确)
    if pos < 0.90:
        return f"❌ B-d 不过(ḡ_A·ḡ_B>0 占比 {pos:.0%} < 90%)⇒ 见 0064 §④(a)", d
    # B-b 精度
    if n < 25 or not (d["width_ratio"] < 3):
        return "⚠️ B-b 不过 —— 拒绝打判词", d
    # B-a 方向
    if hi < B:
        return "已在临界批量之上 ⇒ idea #8 判负(0064 §④b)", d
    if lo > 2 * B:
        return "明确低于临界 ⇒ 4× batch 有据(0064 §④c,须算力配平)", d
    return "未决(CI 跨过 B_eff)—— 按 0064 §③ 不得凭点估计二选一", d


def report(path, quiet=False):
    recs = load(path)
    B, num, den = bnoise_series(recs)
    if B is None:
        print(f"  (无记录: {path})")
        return None
    pos = float((den > 0).mean())
    if pos < 0.90:
        print(f"  ❌ B-d 不过:ḡ_A·ḡ_B>0 仅 {pos:.0%} (<90%) ⇒ ‖G‖² 不可分辨,见 0064 §④(a)")
        return None
    # [[print-the-series-first]] 2026-09-01 一夜因"只读汇总量"栽了三次(§4.75/§4.87a/§4.89)。
    # 结构性修法:**逐点序列先于任何汇总量输出**,不给"只看中位/前后半比"的机会。
    import math as _m2
    print("  逐 epoch 原始序列(先看这个,再看下面的汇总):")
    print("    ep : " + " ".join(f"{r['epoch']:>6}" for r in recs))
    print("    Bn : " + " ".join(f"{B*(r['nrm_full2']-r['dot_ab'])/r['dot_ab']:>6.0f}" for r in recs))
    print("    实现率: " + " ".join(
        f"{_m2.sqrt(r['dot_ab']/r['nrm_full2'])/_m2.sqrt(1+B*(r['nrm_full2']-r['dot_ab'])/r['dot_ab']):>5.1%}"
        for r in recs))
    print("    ⚠️ epoch 0 是 LoRA 初始化瞬态(B 矩阵初始为零),判读时应单独看待。")
    v, d = verdict(B, num, den)
    splits = {}
    for r in recs:
        splits[r.get("split", "random")] = splits.get(r.get("split", "random"), 0) + 1
    if len(splits) > 1:
        print(f"  ⚠️ 记录混用了不同的半批切分口径 {splits} —— 不可混合聚合(0064 §⑦/§⑧)")
    else:
        print(f"  半批切分口径 = {list(splits)[0]}"
              + ("  (随机切分只能判'加每 prompt 图数',见 0064 §⑦)" if list(splits)[0] == "random" else ""))
    print(f"  n_updates={d['n']}  B_eff={d['B_eff']}  内积为正占比={d['dot_pos_rate']:.0%}")
    print(f"  B_noise 中位={d['med']:.0f}  95%CI=[{d['ci_lo']:.0f}, {d['ci_hi']:.0f}]  hi/lo={d['width_ratio']:.2f}")
    # §4.84:实现率 ≈ ‖G‖²/(‖ḡ‖·E‖g‖)。本探针直接测得 ‖G‖²=dot_ab、‖ḡ‖²=nrm_full2,
    # 且 E‖g‖² = ‖G‖²(1+B_noise) ⇒ 实现率 = sqrt(dot/nrm) / sqrt(1+B_noise)。
    import math as _m
    _rr = [_m.sqrt(r["dot_ab"] / r["nrm_full2"]) / _m.sqrt(1.0 + B * (r["nrm_full2"] - r["dot_ab"]) / r["dot_ab"])
           for r in recs if r["dot_ab"] > 0]
    if _rr:
        _rr = sorted(_rr)
        print(f"  §4.84 由本探针直接推出的**预测实现率** = {_rr[len(_rr)//2]:.2%} "
              f"[{_rr[0]:.2%}, {_rr[-1]:.2%}]   (§4.4 实测 1.04~3.22%)")
        if list(splits)[0] == "random":
            print(f"     ⚠️ 随机切分低估噪声(0064 §⑦)⇒ 此值**偏高**;按 prompt 切分的版本才是可比口径。")
    if d["n"] >= 4:
        print(f"  B-c 漂移: 前半={d['med_first']:.0f}  后半={d['med_second']:.0f}  比={d['drift']:.2f}"
              + ("   ⚠️ 后半≥2×前半,'固定 batch'提法本身有问题" if d["drift"] >= 2 else ""))
    print(f"  ⇒ {v}")
    return d


def selftest():
    """合成已知真值,双向验证:'高于临界'与'低于临界'两种情形都要判对。"""
    rng = np.random.default_rng(7)
    B, accum, bs, world = 1152, 48, 8, 3
    d_eff = 600      # §4.79 实测 k_eff 悲观端
    ok = True

    def synth(Bn_true, n):
        """按 ḡ_A/ḡ_B 的真实分布造 dot_ab 与 nrm_full2(‖G‖=1)。"""
        vh = (Bn_true / d_eff) * 2.0 / B
        dot = 1.0 + rng.normal(scale=np.sqrt(2 * vh), size=n) \
                  + rng.normal(scale=np.sqrt(d_eff) * vh, size=n)
        fn2 = rng.chisquare(d_eff, size=n) * (vh / 2)
        nrm = 1.0 + rng.normal(scale=np.sqrt(2 * (vh / 2)), size=n) + fn2
        return [{"epoch": 0, "upd": i, "dot_ab": float(dot[i]), "nrm_full2": float(nrm[i]),
                 "accum": accum, "half": accum // 2, "bs": bs, "world": world} for i in range(n)]

    print("=" * 72)
    print("Gate B 分析器自检(合成已知真值;判据 0064 §③)")
    print("=" * 72)
    cases = [
        (400,  "已在临界批量之上", "真值 400 ≪ B_eff=1152 ⇒ 应判负"),
        (835,  "已在临界批量之上", "点估计 835 < 1152 ⇒ 应判负"),
        (1152, "未决",             "真值恰在临界 ⇒ 应判未决(不许硬判)"),
        (4131, "明确低于临界",     "真值 4131 ≫ 2×1152 ⇒ 应判正"),
    ]
    for Bn, expect, why in cases:
        recs = synth(Bn, 100)
        Bx, num, den = bnoise_series(recs)
        v, dd = verdict(Bx, num, den)
        hit = expect in v
        ok &= hit
        print(f"  真值={Bn:>5}  估计={dd['med']:>6.0f} CI=[{dd['ci_lo']:.0f},{dd['ci_hi']:.0f}]"
              f"  判词={v.split(' ⇒')[0][:22]:<24} {'✅' if hit else '❌'}  ({why})")

    print("\n  --- 反向:样本不足 / 精度不够时必须**拒绝打判词** ---")
    recs = synth(4131, 10)
    Bx, num, den = bnoise_series(recs)
    v, dd = verdict(Bx, num, den)
    hit = "B-b 不过" in v
    ok &= hit
    print(f"  n=10(<25 门槛): 判词={v[:30]:<32} {'✅ 正确拒绝' if hit else '❌ 竟然下了结论'}")

    print("\n  --- 反向:零模型 G=0(纯噪声)⇒ 内积不为正,B-d 必须拦下 ---")
    n = 100
    dot = rng.normal(scale=1.0, size=n)      # 无真信号 ⇒ 内积在 0 附近正负各半
    recs = [{"epoch": 0, "upd": i, "dot_ab": float(dot[i]), "nrm_full2": 5.0,
             "accum": accum, "half": 24, "bs": bs, "world": world} for i in range(n)]
    Bx, num, den = bnoise_series(recs)
    posr = float((den > 0).mean())
    hit = posr < 0.90
    ok &= hit
    print(f"  内积为正占比={posr:.0%} (<90%) {'✅ B-d 正确拦下' if hit else '❌ 未拦下'}")

    print("\n  --- 反向:B_eff 在记录间不一致必须报错,不得静默聚合 ---")
    recs = synth(1152, 30); recs[5]["world"] = 4
    try:
        bnoise_series(recs); print("  ❌ 竟然接受了不一致的 B_eff"); ok = False
    except SystemExit:
        print("  ✅ 正确拒绝")

    print("\n" + ("  ✅ 自检全部通过" if ok else "  ❌ 自检未通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    p = a.path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "outputs", "gateB_probe", "gateB_noise.jsonl")
    if not os.path.exists(p):
        print(f"  (尚无数据: {p})"); sys.exit(0)
    report(p)
