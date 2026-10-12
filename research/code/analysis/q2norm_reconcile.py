#!/usr/bin/env python3
"""余弦法 vs 半批法的对账(EXPERIMENT_LOG §4.86/§4.86b)。

问题:ρ = E[cos] = ‖E[ĝ]‖²(先归一化再平均),而实现率公式要的是 ‖E[g]‖²/E‖g‖²。
      两者只在**逐样本梯度范数齐次**时相等,实测差 2~4 倍。
解法:探针记下逐样本范数 nrm_canon 后,可由 cos_ij·n_i·n_j 还原全部内积,
      于是能在**不假设范数齐次**的情况下直接估 ‖E[g]‖²/E‖g‖²。

⚠️ 只用**跨 prompt** 对:Q2c 的选样逻辑仍在,同 prompt 对被刻意超采样,
   且同 prompt 对共享 prompt 专属分量 p_c,会把 ‖G‖² 抬高(§4.81)。

自检: python3 q2norm_reconcile.py --selftest
"""
import argparse, io, json, os, sys
import numpy as np


def estimates(recs):
    """返回 (无偏信号占比, 余弦法 ρ, n_跨prompt对, n_记录)。"""
    per_sig = []; per_rho = []    # 逐记录的比值(正确口径)
    ip_sum = 0.0; ip_n = 0        # 跨 prompt 的 g_i·g_j
    cs_sum = 0.0                  # 跨 prompt 的 cos_ij
    n2_sum = 0.0; n2_n = 0        # ‖g_i‖²
    used = 0
    for r in recs:
        nr = r.get("nrm_canon")
        cs = r.get("cos_canon")
        ph = r.get("phash")
        if not nr or not cs:
            continue
        ns = len(nr)
        if len(cs) != ns * (ns - 1) // 2:
            continue
        used += 1
        n2_sum += float(np.sum(np.square(nr))); n2_n += ns
        _ip = []; _co = []
        k = 0
        for a in range(ns):
            for b in range(a + 1, ns):
                if ph is None or ph[a] != ph[b]:      # 只用跨 prompt 对
                    ip_sum += cs[k] * nr[a] * nr[b]
                    cs_sum += cs[k]
                    _ip.append(cs[k] * nr[a] * nr[b]); _co.append(cs[k])
                    ip_n += 1
                k += 1
        if _ip:
            per_sig.append(float(np.mean(_ip)) / (float(np.sum(np.square(nr))) / ns))
            per_rho.append(float(np.mean(_co)))
    if ip_n == 0 or n2_n == 0:
        return None
    # 用**聚合**口径(Σnum/Σden):它对"信号占比跨记录恒定"是无偏的,且不引入小样本比值偏差。
    # 逐记录口径(先比再平均)在真实 cv≈0.215 下给 0.76、聚合给 0.58 —— 差 30%,
    # 但该比值的 bootstrap CI 本就是 [0.13, 1.14],这点差别在噪声之内,不值得为它换估计量。
    # 注意:聚合是**按梯度尺度加权**的平均(跨记录尺度跨度 1.8×),口径须随结论一起写明。
    return ip_sum / ip_n / (n2_sum / n2_n), cs_sum / ip_n, ip_n, used


def report(path, half_batch_signal=None):
    recs = [json.loads(l) for l in io.open(path) if l.strip()]
    out = estimates(recs)
    if out is None:
        have = sum(1 for r in recs if r.get("nrm_canon"))
        print(f"  ❌ 无可用记录(总 {len(recs)} 条,其中含 nrm_canon 的 {have} 条)")
        print(f"     若为 0:说明跑的是**加范数记录之前**的代码,须重跑,不要拿旧数据凑。")
        return None
    sig, rho, npair, nrec = out
    print(f"  记录={nrec}  跨 prompt 对={npair}")
    print(f"  余弦法      ρ = E[cos]            = {rho:.5f}")
    print(f"  无偏(cos+范数) ‖E[g]‖²/E‖g‖²    = {sig:.5f}")
    print(f"  比值 无偏/余弦 = {sig / rho:.2f}×  "
          + ("(>1: 范数大者更对齐 ⇒ 余弦法低估)" if sig > rho else "(<1: 纯尺度异质 ⇒ 余弦法高估)"))
    if half_batch_signal is not None:
        print("  ⛔ 与半批法的直接比较**已废除**(2026-09-01,EXPERIMENT_LOG §4.92):")
        print("     (1) 单位不同:本脚本给**样本级** ‖G‖²/E‖g‖²,半批法给**批级** ‖G‖²/‖ḡ‖²,")
        print("         二者差一个 trΣ/B 项;换算式 batch = s/(s+(1−s)/B)。")
        print("     (2) **梯度不同**:本探针对 `pos_loss` 求逐样本梯度(L1972),")
        print("         而半批法测的是真正进 optimizer 的 `r1·pos+(1−r1)·neg`(L2037);")
        print("         实测 neg 与 pos 等量级、r1 以 0.5 为中心 ⇒ 两者本就不是同一个量。")
        print("     ⇒ 不做该比较。Gate B 的判决只依赖半批法本身(0066),不受影响。")
    return sig


def _synth(mode, nrec=400, ns=6, d=800, seed=5):
    rng = np.random.default_rng(seed)
    G = np.zeros(d); G[0] = 1.0
    recs = []
    for _ in range(nrec):
        if mode == "pure_scale":
            w = rng.lognormal(0, 0.8, ns); a = 0.15 * w; s = w
        elif mode == "align_with_norm":
            u = rng.lognormal(0, 0.8, ns); a = 0.15 * u; s = np.ones(ns)
        else:
            a = 0.15 * np.ones(ns); s = np.ones(ns)
        g = a[:, None] * G + s[:, None] * rng.normal(scale=1 / np.sqrt(d), size=(ns, d))
        nr = np.linalg.norm(g, axis=1)
        gh = g / nr[:, None]
        cs = [float(gh[i] @ gh[j]) for i in range(ns) for j in range(i + 1, ns)]
        recs.append({"n_samp": ns, "cos_canon": cs, "nrm_canon": [float(x) for x in nr]})
    # 真值:‖E[g]‖²/E‖g‖²(用同一分布大样本算)
    if mode == "pure_scale":
        w = rng.lognormal(0, 0.8, 40000); a = 0.15 * w; s = w
    elif mode == "align_with_norm":
        u = rng.lognormal(0, 0.8, 40000); a = 0.15 * u; s = np.ones(40000)
    else:
        a = 0.15 * np.ones(40000); s = np.ones(40000)
    gg = a[:, None] * G + s[:, None] * rng.normal(scale=1 / np.sqrt(d), size=(40000, d))
    truth = float(np.linalg.norm(gg.mean(0)) ** 2 / (np.linalg.norm(gg, axis=1) ** 2).mean())
    return recs, truth


def selftest():
    print("=" * 70)
    print("对账脚本自检(合成已知真值;三种范数结构)")
    print("=" * 70)
    ok = True
    print(f"  {'情形':<16}{'真值':>9}{'余弦法':>10}{'偏差':>8}{'无偏重建':>11}{'偏差':>8}")
    for mode, lbl in [("homog", "同质(对照)"), ("pure_scale", "纯尺度异质"),
                      ("align_with_norm", "大范数更对齐")]:
        recs, truth = _synth(mode)
        sig, rho, _, _ = estimates(recs)
        hit = abs(sig / truth - 1) < 0.15
        ok &= hit
        print(f"  {lbl:<16}{truth:>9.4f}{rho:>10.4f}{rho/truth-1:>+7.0%}"
              f"{sig:>11.4f}{sig/truth-1:>+7.0%}{'' if hit else '  ❌'}")
    print("\n  判据:无偏重建三种情形均在真值 ±15% 内;余弦法应在异质情形显著偏离(双向)。")
    recs, truth = _synth("pure_scale"); _, rho, _, _ = estimates(recs)
    caught = abs(rho / truth - 1) > 0.3
    ok &= caught
    print(f"  反向:纯尺度异质下余弦法偏差 {rho/truth-1:+.0%} "
          + ("✅ 确实被检出" if caught else "❌ 未检出,自检无意义"))
    print("\n  --- 反向:无 nrm_canon 的旧数据必须拒绝,不得静默降级 ---")
    old = [{"n_samp": 6, "cos_canon": [0.0] * 15}]
    print("  " + ("✅ 正确拒绝" if estimates(old) is None else "❌ 竟然接受了"))
    ok &= estimates(old) is None
    print("\n" + ("  ✅ 自检全部通过" if ok else "  ❌ 自检未通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None)
    ap.add_argument("--half-batch-signal", type=float, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    p = a.path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "outputs", "gateQ2n_probe", "gateQ2a_probe.jsonl")
    if not os.path.exists(p):
        print(f"  (尚无数据: {p})"); sys.exit(0)
    report(p, a.half_batch_signal)
