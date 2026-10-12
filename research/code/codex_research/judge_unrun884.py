#!/usr/bin/env python
"""§4.884 判定器(数据前写定):三条从未跑过的 idea,A100 np3、seed 0830、100 次更新,与同硬件对照配对。
  vadv     (连续已验证增益加权,canonical 目标不变)  主对照 baseAA(canonical A100 np3),副 baseA3(另一条 canonical 重跑)
  hybridtr (高噪态 K={2,5} 用选择式目标、投影回信任域) 主对照 hybgrad(同状态集、全梯度)⇒ 只换 oracle;副 baseAA
  hybridw  (高噪态用 reward 加权平均目标,τ=1)          同上
每臂、每 ck(50/100)、每 NFE(40/4)两集:P 两格都显著正 | P− ≥1 格显著正且无显著负 | 0 无显著 | N ≥1 格显著负。
主判定 = ck100 的 flow-40 与 flow-4 各自的格;另报底模门(该臂相对未训练底模必须为正)与 baseA3−baseAA 的同协议噪声。"""
import sys, os, importlib.util
SP = "/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec = importlib.util.spec_from_file_location("judge", SP + "/judge.py"); J = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = [sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
sys.argv = _argv
TAG = {}                     # baseAA = 新 watcher 的 baseAAq 标签(旧 baseAA_*_a40 只有 drawbench/flow-40 一格)
fn = lambda t, ck: TAG.get(t, t + "_{ck}").format(ck=ck)
def d(a, b, ck, p, n):
    fa = f"matched_pi_{fn(a, ck)}_default_{p}_flow{n}"; fb = f"matched_pi_{fn(b, ck)}_default_{p}_flow{n}"
    if not (os.path.exists(f"outputs/{fa}.json") and os.path.exists(f"outputs/{fb}.json")): return None
    r = J.paired(J.load(fa, "open3"), J.load(fb, "open3")); return r[1] / J.MDE, r[3]
def base(a, ck, p, n):
    fa = f"matched_pi_{fn(a, ck)}_default_{p}_flow{n}"; fb = f"matched_dflt_basemodel_none_{p}_flow{n}"
    if not (os.path.exists(f"outputs/{fa}.json") and os.path.exists(f"outputs/{fb}.json")): return None
    r = J.paired(J.load(fa, "open3"), J.load(fb, "open3")); return r[1] / J.MDE, r[3]
f = lambda v: "%+6.2f(t%+5.1f)" % v if v else "   ⏳        "
sp = lambda v: v[0] >= 2.5 and v[1] > 2
sn = lambda v: v[0] <= -2.5 and v[1] < -2
def cell(L):
    if any(v is None for v in L): return "⏳"
    if any(sn(v) for v in L): return "N"
    if all(sp(v) for v in L): return "P"
    if any(sp(v) for v in L): return "P−"
    return "0"
ARMS = {"vadv": ("baseH100", "baserepH100"), "hybridtr36": ("mkF4hr2", "baseH100"), "hybridw36": ("mkF4hr2", "baseH100")}   # §4.884 补:改到 H100 np3
print("同协议噪声 baserepH100 − baseH100:", "  ".join(f"ck{ck} f{n} " + "/".join(f(d('baserepH100','baseH100',ck,p,n)) for p in ('drawbench','unseen1k')) for ck in (50,100) for n in (40,4)))
for a, (c1, c2) in ARMS.items():
    print(f"\n===== {a}(主对照 {c1},副 {c2})")
    for ck in (50, 100):
        for n in (40, 4):
            L1 = [d(a, c1, ck, p, n) for p in ("drawbench", "unseen1k")]; L2 = [d(a, c2, ck, p, n) for p in ("drawbench", "unseen1k")]
            B = [base(a, ck, p, n) for p in ("drawbench", "unseen1k")]
            gate = "⏳" if any(v is None for v in B) else ("过" if all(v[0] > 0 for v in B) else "**否决**")
            print(f"  ck{ck} f{n:<2d} vs {c1}: {f(L1[0])} / {f(L1[1])} ⇒ {cell(L1):2s} | vs {c2}: {f(L2[0])} / {f(L2[1])} ⇒ {cell(L2):2s} | 底模门 {gate}")
    print(f"  主判定(§4.884)ck100:flow-40 **{cell([d(a, c1, 100, p, 40) for p in ('drawbench','unseen1k')])}**,flow-4 **{cell([d(a, c1, 100, p, 4) for p in ('drawbench','unseen1k')])}**")
