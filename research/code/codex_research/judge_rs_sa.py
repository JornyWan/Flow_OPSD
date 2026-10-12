#!/usr/bin/env python
"""§4.895 判定器(2026-10-11 19:20 写定,单 reward split 零数据)。用户目标:单 reward 100 步胜过 canonical(主看 flow-40)。
沿用 §4.709 的 R 轴(尺度无关,判在训练所用 reward 的配对 t 上;不报 ×MDE):
  R1 正向 t > 7.5 且胜率 > 55% | R2 无效 |t| < 5.5 | R3 负向 t < −7.5 | R4 不判 5.5 ≤ |t| ≤ 7.5(或 t > 7.5 但胜率 ≤ 55%)
主判定:ck100 flow-40 两集都 R1 ⇒ "胜 canonical";任一 R3 ⇒ 负(先归因);其余 ⇒ 未胜。
对照 = 现成单 reward canonical / MSP(s0830,np4,同节点类型:pickscore/hpsv2 = H100 NVL,clipscore/imagereward = H200 NVL)。
副读数:flow-4;held-out imagereward(训练 reward 本身是 imagereward 时不报)。"""
import sys, os, importlib.util
SP = "/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec = importlib.util.spec_from_file_location("judge", SP + "/judge.py"); J = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = [sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
sys.argv = _argv
SETS = ("drawbench", "unseen1k")
def d(a, b, ck, p, n, axis):
    fa = f"matched_pi_{a}_{ck}_default_{p}_flow{n}"; fb = f"matched_pi_{b}_{ck}_default_{p}_flow{n}"
    if not (os.path.exists(f"outputs/{fa}.json") and os.path.exists(f"outputs/{fb}.json")): return None
    r = J.paired(J.load(fa, axis), J.load(fb, axis))
    return r[1], r[3], r[4]
def R(v):
    if v is None: return "⏳"
    _, t, w = v
    if t > 7.5 and w > 55: return "R1"
    if t < -7.5: return "R3"
    if abs(t) < 5.5: return "R2"
    return "R4"
f = lambda v: "%+.4f(t%+5.1f,%2.0f%%)%s" % (v[0], v[1], v[2], R(v)) if v else "⏳"
for rw in ("pickscore", "hpsv2", "clipscore", "imagereward"):
    a, c, m = f"rs{rw}sa", f"rs{rw}canon", f"rs{rw}msp"
    print(f"\n===== {rw}:{a} vs canonical {c} / MSP {m}(轴 = {rw})")
    for ck in (50, 100):
        for n in (40, 4):
            Lc = [d(a, c, ck, p, n, rw) for p in SETS]; Lm = [d(a, m, ck, p, n, rw) for p in SETS]
            ir = "" if rw == "imagereward" else " | held-out IR vs canon " + " / ".join("%+.3f" % v[0] if v else "⏳" for v in (d(a, c, ck, p, n, "imagereward") for p in SETS))
            print(f"  ck{ck} f{n:<2d} vs canon {f(Lc[0])} / {f(Lc[1])} | vs MSP {f(Lm[0])} / {f(Lm[1])}{ir}")
    L = [R(d(a, c, 100, p, 40, rw)) for p in SETS]
    v = "⏳" if "⏳" in L else ("负 ⇒ 先归因" if "R3" in L else ("胜 canonical" if L == ["R1", "R1"] else "未胜"))
    print(f"  主判定(ck100 flow-40 vs canonical):{L} ⇒ **{v}**")
