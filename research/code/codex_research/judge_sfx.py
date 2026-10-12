#!/usr/bin/env python
"""§4.855 (a) 判定器(数据前写定)。2×2:{canonical, L1} × {无锚, 有锚},ck100,open3 ×MDE。
F = qlook1sfx − qlook1 ≥ +2.5(flow-40,两集,t>2)| W = qlook1sfx − canon_k ≥ +2.5(flow-40,两集,两条对照)
A1 F 且 W | A2 F 且 |qlook1sfx − canon_k| < 2.5(两集两对照)| A3 F 且 qlook1sfx − canon_k ≤ −2.5(两集两对照,t<−2)
A4 非 F:qlook1sfx − qlook1 < +2.5(两集都)| A5 其余。交互 I = (qlook1sfx − qlook1) − (canonsfx − canon_k)。
§4.857 补:同时报 R11−R01 = qlook1sfx − canonsfx;该锚是 A 类(query target 传播到前瞻态),失败不反证 B 类或更新后约束。"""
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
print(f"§4.855 (a) @ck{ck}(open3 ×MDE)")
for n in (40, 4):
    print(f" flow-{n}:")
    for a, b in [("qlook1sfx", "qlook1"), ("qlook1sfx", "canonsfx")] + [("qlook1sfx", c) for c in CTL] + [("canonsfx", c) for c in CTL] + [("qlook1", c) for c in CTL]:
        print(f"   {a:10s} − {b:12s} " + "  ".join(f"{p[:4]} {f(d(a, b, p, n))}" for p in SETS))
    for c in CTL:
        I = []
        for p in SETS:
            x, y = d("qlook1sfx", "qlook1", p, n), d("canonsfx", c, p, n)
            I.append("%+.2f" % (x[0] - y[0]) if x and y else "⏳")
        print(f"   交互 I(对照 {c}): " + " / ".join(I))
F_ = [d("qlook1sfx", "qlook1", p, 40) for p in SETS]
Wc = [d("qlook1sfx", c, p, 40) for c in CTL for p in SETS]
if any(v is None for v in F_ + Wc):
    print("主判定:⏳ 缺格,不判"); sys.exit(0)
F = all(x >= 2.5 and t > 2 for x, t in F_)
if F and all(x >= 2.5 and t > 2 for x, t in Wc): v = "A1 锚修好了 L1,且 40 步赢 OPSD"
elif F and all(abs(x) < 2.5 for x, t in Wc): v = "A2 修好了失稳,只追平 OPSD"
elif F and all(x <= -2.5 and t < -2 for x, t in Wc): v = "A3 部分修复,仍差于 OPSD"
elif all(x < 2.5 for x, t in F_): v = "A4 锚没修好 L1"
else: v = "A5 未判"
print(f"主判定(§4.855 (a),flow-40):**{v}**  —— 底模门另查(open3 四格须高于未训练底模)")
