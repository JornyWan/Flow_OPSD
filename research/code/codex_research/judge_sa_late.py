#!/usr/bin/env python
"""§4.894 判定器(2026-10-11 17:25 写定;split λ=0.5 ck150+ 零数据)。首要目标 = 40 步每个 ckpt 都胜 canonical。
一、早读(只是指示,不是判定):筛查 run 本身 UPDATES=300,但同硬件 canonical 只到 ck100。ck150–300 先对 H200 np4 的现成 300 步对照读:
    s0830:sah3s30(H100 np3)vs o3canon300 / o3msp300;s0905:saa3s05(A100 np3)vs canon300s2 / msp300s2。
    跨硬件偏差(ck100 实测 canonical |Δ| ≤ 0.70、MSP ≤ 1.06 ×MDE)⇒ 指示阈值抬到 **≥3.5×MDE 且 |t|>2**(2.5 + 1.0 偏差余量)。
    每格标 + / − / ·;另报 ck100 的硬件偏差估计 = 同 ck100 下(对 H200np4 canonical)−(对同硬件 canonical)。
二、正式判定(同硬件,10 格 flow-40 × ck100–300 × 两集,对 canonical):W 全胜 | W− 无显著负且 ≥6 | T 无显著负且 <6 | L 任一显著负;显著 = ≥2.5 且 |t|>2。
    s0830:sah3s30 vs g3hcanon(H100 np3,排在 sah3s30 之后);s0905:暂无同硬件 300 步 canonical(⏳)。
三、副读数:等效提前步数 Δ(c)(§4.892,对 H200np4 canonical 轨迹,跨硬件只作形状);稳定性:metrics.jsonl 中 r = od_gn_x / od_gn_q,
    ep100–300 内 r > 0.75(门控阈值)的最长连续步数 Lr 与 od_cos_xx 中位数(r 是 OPT_DIAG=1 下"抵消失效"的代理,门控 run 8 步、不门控 18 步)。"""
import sys, os, json, importlib.util
SP = "/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec = importlib.util.spec_from_file_location("judge", SP + "/judge.py"); J = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = [sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
sys.argv = _argv
CKS = (100, 150, 200, 250, 300); SETS = ("drawbench", "unseen1k")
def d(a, b, ck, p, n):
    fa = f"matched_pi_{a}_{ck}_default_{p}_flow{n}"; fb = f"matched_pi_{b}_{ck}_default_{p}_flow{n}"
    if not (os.path.exists(f"outputs/{fa}.json") and os.path.exists(f"outputs/{fb}.json")): return None
    r = J.paired(J.load(fa, "open3"), J.load(fb, "open3"))
    return r[1] / J.MDE, r[3]
def mark(v, thr):
    if v is None: return "   ⏳         "
    s = "+" if v[0] >= thr and v[1] > 2 else ("−" if v[0] <= -thr and v[1] < -2 else "·")
    return "%+6.2f(t%+5.1f)%s" % (v[0], v[1], s)
RUNS = {"s0830": ("sah3s30", "sa_h3_s30", "o3canon300", "o3msp300", "baseH100", "g3hcanon"),
        "s0905": ("saa3s05", "sa_a3_s05", "canon300s2", "msp300s2", "bases2", None)}
for s, (a, run, xc, xm, hc, hc300) in RUNS.items():
    print(f"\n===== {s}:{a}")
    print(" 一、早读(对 H200np4,阈 3.5):ck | vs canonical dr / un | vs MSP dr / un")
    for n in (40, 4):
        for ck in CKS:
            print(f"   f{n:<2d} ck{ck} | " + " / ".join(mark(d(a, xc, ck, p, n), 3.5) for p in SETS) + " | " + " / ".join(mark(d(a, xm, ck, p, n), 3.5) for p in SETS))
    bias = [d(hc, xc, 100, p, 40) for p in SETS]
    print("   ck100 硬件偏差估计(同硬件 canonical − H200np4 canonical,f40):" + " / ".join("%+.2f" % v[0] if v else "⏳" for v in bias))
    if hc300:
        L = [d(a, hc300, ck, p, 40) for ck in CKS for p in SETS]
        if any(v is None for v in L): v = "⏳ 缺格"
        else:
            k = sum(x[0] >= 2.5 and x[1] > 2 for x in L)
            v = f"L 有输({k}/10)" if any(x[0] <= -2.5 and x[1] < -2 for x in L) else ("W 全胜" if k == 10 else (f"W− ({k}/10)" if k >= 6 else f"T ({k}/10)"))
        print(f" 二、正式判定(同硬件 {hc300},flow-40 10 格):**{v}**  " + "  ".join(f"ck{ck} " + "/".join(mark(d(a, hc300, ck, p, 40), 2.5) for p in SETS) for ck in CKS))
    else:
        print(" 二、正式判定:⏳ 无同硬件 300 步 canonical")
    # 三、Δ(c) 与稳定性
    for p in SETS:
        if all(d(a, xc, ck, p, 40) for ck in CKS):
            lc = [0.0] + [J.paired(J.load(f"matched_pi_{xc}_{k}_default_{p}_flow40", "open3"), J.load(f"matched_pi_{xc}_100_default_{p}_flow40", "open3"))[1] / J.MDE for k in CKS[1:]]
            out = []
            for i, k in enumerate(CKS):
                y = d(a, xc, k, p, 40)[0] + lc[i]; z = None; t = ""
                for j in range(4):
                    if lc[j] <= y <= lc[j + 1]: z = CKS[j] + 50 * (y - lc[j]) / (lc[j + 1] - lc[j]); break
                if z is None:
                    t = "*"; z = (300 + 50 * (y - lc[-1]) / (lc[-1] - lc[-2])) if y > lc[-1] else (100 + 50 * (y - lc[0]) / (lc[1] - lc[0]))
                out.append("%+.0f%s" % (z - k, t))
            print(f" 三、Δ(c) f40 {p[:2]}(对 {xc},跨硬件只看形状):" + "/".join(out))
        else:
            print(f" 三、Δ(c) f40 {p[:2]}:⏳")
    M = [json.loads(l) for l in open(f"outputs/{run}/metrics.jsonl") if l.strip()]
    M = [m for m in M if "od_gn_x" in m and "od_gn_q" in m]
    st = [(m.get("epoch", m.get("step", i)), m["od_gn_x"] / max(m["od_gn_q"], 1e-12), m.get("od_cos_xx")) for i, m in enumerate(M)]
    late = [x for x in st if x[0] >= 100]
    best = cur = 0
    for _, r, _ in late:
        cur = cur + 1 if r > 0.75 else 0; best = max(best, cur)
    cx = sorted(c for _, _, c in late if c is not None)
    print(f" 三、稳定性:已记 {len(st)} 步,ep≥100 有 {len(late)} 步;Lr(r>0.75 最长连续)= {best if late else '⏳'};"
          f"od_cos_xx 中位 {cx[len(cx)//2]:.3f}" if cx else f" 三、稳定性:已记 {len(st)} 步,ep≥100 有 {len(late)} 步 ⏳")
# §4.894:晋级后的 H200×4 300 步(h2sa30 / h2sa05)—— 同硬件正式判定,对照 = 现成 o3canon300 / canon300s2(MSP:o3msp300 / msp300s2)
print("\n===== H200×4 300 步(晋级后才有)")
for s, (a, c, m) in {"s0830": ("h2sa30", "o3canon300", "o3msp300"), "s0905": ("h2sa05", "canon300s2", "msp300s2")}.items():
    for nm, b in (("vs canonical", c), ("vs MSP", m)):
        L = [d(a, b, ck, p, 40) for ck in CKS for p in SETS]
        if any(v is None for v in L): v = "⏳ 缺格"
        else:
            k = sum(x[0] >= 2.5 and x[1] > 2 for x in L)
            v = f"L 有输({k}/10)" if any(x[0] <= -2.5 and x[1] < -2 for x in L) else ("W 全胜" if k == 10 else (f"W− ({k}/10)" if k >= 6 else f"T ({k}/10)"))
        print(f" {s} {a} {nm} flow-40:**{v}**  " + "  ".join(f"ck{ck} " + "/".join(mark(d(a, b, ck, p, 40), 2.5) for p in SETS) for ck in CKS))
