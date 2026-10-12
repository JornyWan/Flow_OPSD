#!/usr/bin/env python
"""§4.847 判据的机械实现(判据数据前已登记;本脚本写于数据之后,只照抄规则)。
A = qlook1fz − qlook1,B = qlook1fz − canonical(baseH100、baserepH100 都要满足);flow-40,两集,ck100;
"≥+2.5" 与 "≤−2.5" 另要求同号 |t|>2(§4.615)。
Z1+ A≥+2.5 且 B≥+2.5 | Z1 A≥+2.5 且 |B|<2.5 | Z3 A≥+2.5 且 B≤−2.5 | Z2 |A|<2.5 且 B≤−2.5 | Z4 其余。
Z2 按 §4.849 修订读法:不是 old-Jacobian 漂移;剩"目标没兑现"与"共享参数改其他步"两种。"""
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
    for a, b in [("qlook1fz", "qlook1")] + [("qlook1fz", c) for c in CTL]:
        print(f"   {a} − {b:12s} " + "  ".join(f"{p[:4]} {f(d(a, b, p, n))}" for p in SETS))
A = [d("qlook1fz", "qlook1", p, 40) for p in SETS]
B = [d("qlook1fz", c, p, 40) for c in CTL for p in SETS]
if any(v is None for v in A + B): sys.exit("主判定:⏳ 缺格")
up = lambda L: all(x >= 2.5 and t > 2 for x, t in L); dn = lambda L: all(x <= -2.5 and t < -2 for x, t in L)
null = lambda L: all(abs(x) < 2.5 for x, t in L)
if up(A) and up(B): v = "Z1+"
elif up(A) and null(B): v = "Z1"
elif up(A) and dn(B): v = "Z3"
elif null(A) and dn(B): v = "Z2"
else: v = "Z4 不判"
print(f"主判定(§4.847,flow-40,ck{ck}):**{v}**")
