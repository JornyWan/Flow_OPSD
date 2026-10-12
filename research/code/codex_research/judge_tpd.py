#!/usr/bin/env python
"""§4.868 判定器(数据前写定)。ck100;主 tpdF4 − mkF4hr2(flow-40,两集):P+ 都 ≥+2.5 且 t>2 | P0 都 |·|<2.5 | P− 都 ≤−2.5 且 t<−2 | 未判。"""
import sys, os, importlib.util
SP = "/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec = importlib.util.spec_from_file_location("judge", SP + "/judge.py"); J = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = [sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
sys.argv = _argv
ck = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SETS = ("drawbench", "unseen1k")
def d(a, b, p, n):
    fa = f"matched_pi_{a}_{ck}_default_{p}_flow{n}"; fb = f"matched_pi_{b}_{ck}_default_{p}_flow{n}"
    if not (os.path.exists(f"outputs/{fa}.json") and os.path.exists(f"outputs/{fb}.json")): return None
    r = J.paired(J.load(fa, "open3"), J.load(fb, "open3")); return r[1] / J.MDE, r[3]
f = lambda v: "%+7.2f(t=%+6.1f)" % v if v else "   ⏳   "
def cls(L):
    if any(v is None for v in L): return "⏳"
    if all(v[0] >= 2.5 and v[1] > 2 for v in L): return "P+"
    if all(abs(v[0]) < 2.5 for v in L): return "P0"
    if all(v[0] <= -2.5 and v[1] < -2 for v in L): return "P−"
    return "未判"
for n in (40, 4):
    print(f"flow-{n} @ck{ck}:")
    for b in ("mkF4hr2", "baseH100", "baserepH100"):
        print(f"   tpdF4 − {b:12s} " + "  ".join(f"{p[:4]} {f(d('tpdF4', b, p, n))}" for p in SETS))
    print(f"   主判定(tpdF4 − mkF4hr2,flow-{n}):**{cls([d('tpdF4', 'mkF4hr2', p, n) for p in SETS])}**")
