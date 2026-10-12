#!/usr/bin/env python
"""§4.859 判定器(数据前写定)。L1 上的 2×2:qlook1(00)/ qlook1fz(10)/ qlook1sfx(01)/ qlook1fzsfx(11);ck100,flow-40 主轴。
C = 11 − 10 ≥ +2.5(t>2,两集)| W = 11 − canonical ≥ +2.5(两集两对照)
K1 C 且 W | K2 C 且 |11 − canon| < 2.5 | K3 C 且 11 − canon ≤ −2.5(t<−2)| K4 |11 − 10| < 2.5 | K5 11 − 10 ≤ −2.5(t<−2)| K6 其余。"""
import sys, os, importlib.util
SP = "/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec = importlib.util.spec_from_file_location("judge", SP + "/judge.py"); J = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = [sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
sys.argv = _argv
ck = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SETS = ("drawbench", "unseen1k"); CTL = ("baseH100", "baserepH100")
def d(a, b, p, n):
    fa = f"matched_pi_{a}_{ck}_default_{p}_flow{n}"; fb = f"matched_pi_{b}_{ck}_default_{p}_flow{n}"
    if not (os.path.exists(f"outputs/{fa}.json") and os.path.exists(f"outputs/{fb}.json")): return None
    r = J.paired(J.load(fa, "open3"), J.load(fb, "open3")); return r[1] / J.MDE, r[3]
f = lambda v: "%+7.2f(t=%+6.1f)" % v if v else "   ⏳   "
for n in (40, 4):
    print(f"flow-{n} @ck{ck}:")
    for a, b in [("qlook1fzsfx", "qlook1fz"), ("qlook1fzsfx", "qlook1sfx")] + [("qlook1fzsfx", c) for c in CTL]:
        print(f"   {a} − {b:12s} " + "  ".join(f"{p[:4]} {f(d(a, b, p, n))}" for p in SETS))
    I = []
    for p in SETS:
        x, y = d("qlook1fzsfx", "qlook1fz", p, n), d("qlook1sfx", "qlook1", p, n)
        I.append("%+.2f" % (x[0] - y[0]) if x and y else "⏳")
    print("   2×2 交互 (11−10)−(01−00): " + " / ".join(I))
C_ = [d("qlook1fzsfx", "qlook1fz", p, 40) for p in SETS]
W_ = [d("qlook1fzsfx", c, p, 40) for c in CTL for p in SETS]
if any(v is None for v in C_ + W_): sys.exit("主判定:⏳ 缺格")
up = lambda L: all(x >= 2.5 and t > 2 for x, t in L); dn = lambda L: all(x <= -2.5 and t < -2 for x, t in L)
nl = lambda L: all(abs(x) < 2.5 for x, t in L)
if up(C_) and up(W_): v = "K1 叠加后赢 OPSD"
elif up(C_) and nl(W_): v = "K2 叠加追平 OPSD"
elif up(C_) and dn(W_): v = "K3 互补但仍差"
elif nl(C_): v = "K4 冻结后锚不再有用"
elif dn(C_): v = "K5 冻结后锚只剩代价"
else: v = "K6 未判"
print(f"主判定(§4.859,flow-40,ck{ck}):**{v}**  —— 底模门另查")
