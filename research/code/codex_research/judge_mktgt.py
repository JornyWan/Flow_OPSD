#!/usr/bin/env python
"""§4.861 判定器(数据前写定)。2×2:R00 canonical(baseH100/baserepH100)| R01 mkconsF4 | R10 mkdirF4 | R11 mkF4h。ck100。
T1 R10−R00 ≥ +2.5(flow-40)| T2 R01−R00 ≥ +2.5 且 R11−R10 ≥ +2.5(flow-4)| T3 R11−R10 ≤ −2.5(flow-40)。
每项:是 = 两集(含两对照)都满足且 |t|>2;否 = 都 |·|<1;未判 = 其余。"""
import sys, os, importlib.util
SP = "/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec = importlib.util.spec_from_file_location("judge", SP + "/judge.py"); J = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = [sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
sys.argv = _argv
ck = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SETS = ("drawbench", "unseen1k"); CTL = ("baseH100", "baserepH100")
R = {"R01": "mkconsF4", "R10": "mkdirF4", "R11": "mkF4h"}
# §4.863:mkF4h 在 ck80 被抢占取消(干净数据只到 ck80);ck100 的 R11 改用从头重跑的 mkF4hr2(同配置、同硬件、同 seed)。
if ck == 100: R["R11"] = "mkF4hr2"
def d(a, b, p, n):
    fa = f"matched_pi_{a}_{ck}_default_{p}_flow{n}"; fb = f"matched_pi_{b}_{ck}_default_{p}_flow{n}"
    if not (os.path.exists(f"outputs/{fa}.json") and os.path.exists(f"outputs/{fb}.json")): return None
    r = J.paired(J.load(fa, "open3"), J.load(fb, "open3")); return r[1] / J.MDE, r[3]
f = lambda v: "%+7.2f(t=%+6.1f)" % v if v else "   ⏳   "
def vs00(a, n): return [d(a, c, p, n) for c in CTL for p in SETS]
def cls_ge(L, thr):
    if any(v is None for v in L): return "⏳"
    if all(v[0] >= thr and v[1] > 2 for v in L): return "是"
    if all(abs(v[0]) < 1 for v in L): return "否"
    return "未判"
def cls_le(L, thr):
    if any(v is None for v in L): return "⏳"
    if all(v[0] <= thr and v[1] < -2 for v in L): return "是"
    if all(abs(v[0]) < 1 for v in L): return "否"
    return "未判"
for n in (40, 4):
    print(f"flow-{n} @ck{ck}(×MDE,两列 = drawbench / unseen1k)")
    for k, a in R.items():
        for c in CTL:
            print(f"   {k} {a:9s} − {c:12s} " + "  ".join(f(d(a, c, p, n)) for p in SETS))
    for a, b in ((R["R11"], "mkdirF4"), (R["R11"], "mkconsF4"), ("mkdirF4", "mkconsF4")):
        print(f"   {a} − {b:9s}           " + "  ".join(f(d(a, b, p, n)) for p in SETS))
T1 = cls_ge(vs00("mkdirF4", 40), 2.5)
T2a = cls_ge(vs00("mkconsF4", 4), 2.5); T2b = cls_ge([d(R["R11"], "mkdirF4", p, 4) for p in SETS], 2.5)
T2 = "是" if (T2a == "是" and T2b == "是") else ("⏳" if "⏳" in (T2a, T2b) else ("否" if "否" in (T2a, T2b) else "未判"))
T3 = cls_le([d(R["R11"], "mkdirF4", p, 40) for p in SETS], -2.5)
print(f"T1(D 单独带来 40 步增益)= {T1} | T2(C 承担少步增益)= {T2}(R01−R00 {T2a},R11−R10 {T2b})| T3(C 伤 40 步)= {T3}")
dm = [d("mkdirF4", R["R11"], p, 40) for p in SETS]
if T1 == "是" and all(v and v[0] >= -1 for v in dm):
    print("决策:dir 版成为 40 步稳健性新候选 ⇒ 接 300 步两 seed(§4.861 §四)")
elif T1 == "否":
    print("决策:方向搬运不携带增益;MSP 的价值在 C ⇒ 稳健性靠稳住 C")
else:
    print("决策:⏳ 或未判 —— 不触发决策规则")
