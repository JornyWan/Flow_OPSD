#!/usr/bin/env python
"""B 轴(RB-LT)判定器 —— **在数据到齐之前**写好,避免临场改判据。

登记来源:§4.780 §八(B0/B1–B4)+ §4.792 §二(主对照换成 poseulerrot)
          + §4.776(底模门)+ §4.709(单 reward 判在 t 上,阈 7.5/5.5)+ §4.801(B0 的符号一致性归因)。

主轴 = **pickscore**(该配方自己的训练 reward,§4.709 的 R 轴口径);open3 复合并列报,不判。
flow-4 两集必须同号。阈值用 R 轴:t>7.5 正 / |t|<5.5 无效 / t<-7.5 负 / 5.5-7.5 不判。

三条比较:
  rblt − poseulerrot  **主判定**。两臂同 σ 轮转、同 d 构造、同 positive-only、同 Euler
                      ⇒ 只差"目标怎么构"(residual-bridge vs canonical 的 y+)。
  rblt − poseuler     次要。还多差"训练状态分布"(轮转 σ vs 固定 σ*)⇒ **报但不判**。
  rbltd0 − 未训练底模 **B0 门**。d≡0 本该等于没训练。
                      任一集 |t|>5.5 ⇒ 门触发;但按 §4.801 必须同时看**符号是否一致**:
                        符号不一致 ⇒ Adam 放大数值残差(良性,已预登记)
                        一致向好   ⇒ 信号泄漏(实现缺陷)
  另:§4.776 底模门 —— 两臂的 open3 复合分都必须高于未训练底模,否则该格口径否决。
⚠️ 缺格一律"未判",绝不外推。
用法: judge_axis_B.py [ck]   默认 100
"""
import sys, os, importlib.util, statistics as st
SP = "/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec = importlib.util.spec_from_file_location("judge", SP + "/judge.py")
J = importlib.util.module_from_spec(spec); _argv = sys.argv; sys.argv = [sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
CK = int(_argv[1]) if len(_argv) > 1 else 100
T = {"rblt": "rblt_%d" % CK, "rot": "rblposeulerrot_%d" % CK,
     "pos": "rblposeuler_%d" % CK, "d0": "rbltd0_%d" % CK}
def path(tag, p, n): return "matched_pi_%s_default_%s_flow%d" % (tag, p, n)
def have(f): return os.path.exists("outputs/%s.json" % f)
def cell(a, b, p, n, ax):
    fa, fb = path(T[a], p, n), path(T[b], p, n)
    if not (have(fa) and have(fb)): return None
    A, B = J.load(fa, ax), J.load(fb, ax); r = J.paired(A, B)
    return dict(t=r[3], d=r[1], win=r[4], absA=st.mean(A.values()), absB=st.mean(B.values()))
def rband(t):
    if t > 7.5: return "+ 正向(t>7.5)"
    if t < -7.5: return "- 负向(t<-7.5)"
    if abs(t) < 5.5: return "0 无效(|t|<5.5)"
    return "? 不判(5.5-7.5)"
print("B 轴判定 @ck%d   主轴=pickscore(§4.709),open3 并列报不判" % CK)
res = {}
for name, (a, b) in (("主判定  rblt − poseulerrot", ("rblt", "rot")),
                     ("次要    rblt − poseuler", ("rblt", "pos"))):
    print("\n── %s ──" % name)
    got = {}
    for p in ("drawbench", "unseen1k"):
        for n in (4, 40):
            rp = cell(a, b, p, n, "pickscore"); r3 = cell(a, b, p, n, "open3")
            if rp is None: print("   %-10s flow%-3d ⏳ 缺格 ⇒ 未判" % (p, n)); continue
            got[(p, n)] = rp
            print("   %-10s flow%-3d pickscore Δ=%+.4f t=%+7.2f %-16s | open3 %+6.2f×MDE t=%+6.2f"
                  % (p, n, rp["d"], rp["t"], rband(rp["t"]), r3["d"]/J.MDE, r3["t"]))
    res[a+b] = got
# ── B0 门 ──
print("\n── B0 门  rbltd0 − 未训练底模(open3)──")
b0 = {}
for p in ("drawbench", "unseen1k"):
    for n in (4, 40):
        fa, fb = path(T["d0"], p, n), "matched_dflt_basemodel_none_%s_flow%d" % (p, n)
        if not (have(fa) and have(fb)): print("   %-10s flow%-3d ⏳ 缺格" % (p, n)); continue
        A, B = J.load(fa, "open3"), J.load(fb, "open3"); r = J.paired(A, B)
        b0[(p, n)] = (r[1]/J.MDE, r[3])
        print("   %-10s flow%-3d %+7.2f×MDE  t=%+7.2f" % (p, n, r[1]/J.MDE, r[3]))
# ── 底模门(§4.776)──
print("\n── 底模门(§4.776):两臂复合分都须高于未训练底模 ──")
bg = True
for p in ("drawbench", "unseen1k"):
    for n in (4, 40):
        fb = "matched_dflt_basemodel_none_%s_flow%d" % (p, n)
        if not have(fb): continue
        base = st.mean(J.load(fb, "open3").values())
        for k in ("rblt", "rot"):
            fa = path(T[k], p, n)
            if not have(fa): continue
            v = st.mean(J.load(fa, "open3").values())
            if v < base:
                bg = False; print("   ❌ %s %s flow%d: %.4f < 底模 %.4f" % (k, p, n, v, base))
print("   %s" % ("✓ 通过(或缺格未查全)" if bg else "❌ 口径否决"))
# ── 终判 ──
print()
need = [("drawbench", 4), ("unseen1k", 4)]
if not all(k in b0 for k in need):
    print("B0 门:⏳ 缺格 ⇒ 无法判门 ⇒ **主判定未判**"); sys.exit(0)
fired = any(abs(b0[k][1]) > 5.5 for k in b0)
signs = [b0[k][0] > 0 for k in b0]
consistent = all(signs) or not any(signs)
print("B0 门:%s" % ("**触发**(任一格 |t|>5.5)" if fired else "未触发"))
print("  符号一致性:%s ⇒ %s" % ("一致" if consistent else "**不一致**",
      "疑似信号泄漏,先查实现" if (fired and consistent) else "Adam 放大数值残差(§4.801 预登记的良性解释)"))
g = res["rbltrot"]
if not all(k in g for k in need):
    print("主判定:⏳ 缺格 ⇒ **未判**"); sys.exit(0)
ts = [g[k]["t"] for k in need]
same = (ts[0] > 0) == (ts[1] > 0)
if not same: print("主判定:flow-4 两集**不同号** ⇒ **未判**(t=%+.2f / %+.2f)" % tuple(ts))
elif all(t > 7.5 for t in ts):   print("主判定:**B1 正向** t=%+.2f / %+.2f" % tuple(ts))
elif all(t < -7.5 for t in ts):  print("主判定:**B3 负向** t=%+.2f / %+.2f" % tuple(ts))
elif all(abs(t) < 5.5 for t in ts): print("主判定:**B2 无效** t=%+.2f / %+.2f" % tuple(ts))
else: print("主判定:**B4 不判**(落在 5.5-7.5 或两集带不同)t=%+.2f / %+.2f" % tuple(ts))
if fired: print("⚠️ B0 已触发 ⇒ 上述主判定的正向**不能**归给 reward 修正,须先按符号一致性归因")
