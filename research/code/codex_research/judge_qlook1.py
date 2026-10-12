#!/usr/bin/env python
"""§4.844 第二阶段判定器(数据前写好)。Δ_k = qlook1@ck − control_k@ck,k ∈ {baseH100, baserepH100}。
主轴 flow-40,两集;§4.615 阈值(≥2.5×MDE 且 |t|>2 显著;<1.0 無效應)。
T1 两对照都 Δ≥+2.5 且 t>2(两集)| T2 两对照都 |Δ|<1.0 | T3 两对照都 Δ≤−2.5 且 t<−2 | T4 其余"""
import sys, os, importlib.util
SP = "/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec = importlib.util.spec_from_file_location("judge", SP + "/judge.py"); J = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = [sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
sys.argv = _argv
ck = int(sys.argv[1]) if len(sys.argv) > 1 else 100
def d(a, b, p, n):
    fa = f"matched_pi_{a}_default_{p}_flow{n}"; fb = f"matched_pi_{b}_default_{p}_flow{n}"
    if not (os.path.exists(f"outputs/{fa}.json") and os.path.exists(f"outputs/{fb}.json")): return None
    r = J.paired(J.load(fa, "open3"), J.load(fb, "open3")); return r[1] / J.MDE, r[3]
print("qlook1 @ck%d(open3 ×MDE)" % ck)
res = {}
for ctl in ("baseH100", "baserepH100"):
    for n in (40, 4):
        for p in ("drawbench", "unseen1k"):
            v = d(f"qlook1_{ck}", f"{ctl}_{ck}", p, n); res[(ctl, n, p)] = v
            print("  vs %-12s flow-%-2d %-9s %s" % (ctl, n, p, "%+7.2f (t=%+6.2f)" % v if v else "⏳"))
print("  对照间噪声(baseH100 − baserepH100,flow-40): %s / %s" % tuple(
    "%+.2f" % d(f"baseH100_{ck}", f"baserepH100_{ck}", p, 40)[0] if d(f"baseH100_{ck}", f"baserepH100_{ck}", p, 40) else "⏳"
    for p in ("drawbench", "unseen1k")))
cells = [res[(c, 40, p)] for c in ("baseH100", "baserepH100") for p in ("drawbench", "unseen1k")]
if any(v is None for v in cells): print("主判定:⏳ 缺格,不判"); sys.exit(0)
if all(v[0] >= 2.5 and v[1] > 2 for v in cells): verdict = "T1(兑现)"
elif all(abs(v[0]) < 1.0 for v in cells): verdict = "T2(没兑现)"
elif all(v[0] <= -2.5 and v[1] < -2 for v in cells): verdict = "T3(更差)"
else: verdict = "T4(未判)"
print("主判定(§4.844 §三,flow-40):**%s**" % verdict)
