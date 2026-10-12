#!/usr/bin/env python
"""A 轴(退火臂)判定器 —— **在数据到齐之前**写好,避免临场改判据。

登记来源:§4.773(格子 A0/A1–A5)+ §4.782(A0 的更正)+ §4.615(分带与 t 门)。

三条比较:
  Δ_ar = anneal200 − resumectl200   **主判定**。两臂共担 EMA 重启 + 同源 ckpt + 同套 RNG 处理,
                                     只差"后 1/3 是否关掉传播" ⇒ 单变量。
  Δ_rm = resumectl200 − msp300      **A0 门**,量的是"续训(= EMA 重启)本身"的伪效应。
                                     §4.782 查明 save_ckpt 把 EMA 权重写进 lora/,resume 载为活权重
                                     ⇒ 这不是零,它本身是一个**与本方法无关**的独立发现。
  Δ_ac = anneal200 − canon300       "可部署"口径。**A0 触发时此项不判**(被 EMA 重启混淆)。

判据(§4.615:量级 ≥2.5×MDE **且** |t|>2 才算显著;1.0–2.5 需第二 seed;<1.0 记无效应):
  A0 触发(|Δ_rm| ≥ 2.5 且 |t|>2) ⇒ Δ_ac **不判**,只报;主判定照做但措辞限定在"两条同样续训的臂之间"。
  主判定(Δ_ar,flow-4 两集必须同号):
    A1'  Δ_ar ≥ +2.5   ⇒ 关掉传播**确实恢复**
    A4   |Δ_ar| < 2.5  ⇒ 无恢复(ck300 的赤字在第 200 步就已定下)
    A5   Δ_ar ≤ −2.5   ⇒ 关掉更差(传播到最后仍在帮忙)
  A0 未触发时,再按 §4.773 原文用 Δ_ac 把 A1' 细分为 A1/A2/A3。
⚠️ 缺格一律打印"未判",**绝不**用现有格外推。
用法: judge_axis_A.py [ck]   默认 300
"""
import sys, os, importlib.util, statistics as st
SP = "/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec = importlib.util.spec_from_file_location("judge", SP + "/judge.py")
J = importlib.util.module_from_spec(spec); sys.argv_backup = sys.argv; sys.argv = [sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
CK = int(sys.argv_backup[1]) if len(sys.argv_backup) > 1 else 300
TAG = {"anneal": "anneal200_%d" % CK, "resumectl": "resumectl200_%d" % CK,
       "msp": "o3msp300_%d" % CK, "canon": "o3canon300_%d" % CK}
def cell(a, b, p, n):
    fa = "matched_pi_%s_default_%s_flow%d" % (TAG[a], p, n)
    fb = "matched_pi_%s_default_%s_flow%d" % (TAG[b], p, n)
    if not (os.path.exists("outputs/%s.json" % fa) and os.path.exists("outputs/%s.json" % fb)): return None
    A, B = J.load(fa, "open3"), J.load(fb, "open3"); r = J.paired(A, B)
    return dict(mde=r[1]/J.MDE, t=r[3], win=r[4], absA=st.mean(A.values()), absB=st.mean(B.values()))
def band(m, t):
    """§4.615 分带。返回 (显著?, 文字)"""
    if abs(m) >= 2.5 and abs(t) > 2: return True,  "显著(|Δ|≥2.5 且 |t|>2)"
    if abs(m) >= 1.0:                return False, "未判(1.0≤|Δ|<2.5,需第二 seed 同号)"
    return False, "无效应(|Δ|<1.0)"
print("A 轴判定 @ck%d  (MDE=%.4f)" % (CK, J.MDE))
res = {}
for name, (a, b) in (("Δ_ar 主判定  anneal−resumectl", ("anneal","resumectl")),
                     ("Δ_rm A0门    resumectl−msp300", ("resumectl","msp")),
                     ("Δ_ac 可部署  anneal−canon300", ("anneal","canon"))):
    print("\n── %s ──" % name)
    got = {}
    for p in ("drawbench", "unseen1k"):
        for n in (4, 40):
            r = cell(a, b, p, n)
            if r is None:
                print("   %-10s flow%-3d  ⏳ 缺格 ⇒ 未判" % (p, n)); continue
            sig, txt = band(r["mde"], r["t"])
            got[(p, n)] = r
            print("   %-10s flow%-3d %+8.2f×MDE  t=%+7.2f  win=%5.1f%%  %s"
                  % (p, n, r["mde"], r["t"], r["win"], txt))
    res[a + "-" + b] = got
print()
ar, rm, ac = res["anneal-resumectl"], res["resumectl-msp"], res["anneal-canon"]
need4 = [("drawbench",4),("unseen1k",4)]
# A0
if all(k in rm for k in need4):
    a0 = any(band(rm[k]["mde"], rm[k]["t"])[0] for k in rm)
    print("A0 门:%s" % ("**触发** ⇒ 续训(EMA 重启)自身有显著伪效应 ⇒ Δ_ac 不判,主判定措辞限定在"
                        "'两条同样续训的臂之间'" if a0 else "未触发 ⇒ 续训本身无显著伪效应,Δ_ac 可判"))
else:
    a0 = None; print("A0 门:⏳ 缺格 ⇒ **无法判门**,主判定与 Δ_ac 一律记未判")
# 主判定
if a0 is not None and all(k in ar for k in need4):
    ms = [ar[k]["mde"] for k in need4]
    same = (ms[0] > 0) == (ms[1] > 0)
    sigs = [band(ar[k]["mde"], ar[k]["t"])[0] for k in need4]
    if not same: print("主判定:flow-4 两集**不同号** ⇒ **未判**")
    elif all(sigs) and ms[0] >= 2.5: print("主判定:**A1'(关掉传播确实恢复)** Δ_ar = %+.2f / %+.2f" % tuple(ms))
    elif all(sigs) and ms[0] <= -2.5: print("主判定:**A5(关掉更差)** Δ_ar = %+.2f / %+.2f" % tuple(ms))
    elif not any(abs(m) >= 2.5 for m in ms): print("主判定:**A4(无恢复)** Δ_ar = %+.2f / %+.2f" % tuple(ms))
    else: print("主判定:**未判**(一集显著一集不显著)Δ_ar = %+.2f / %+.2f" % tuple(ms))
else:
    print("主判定:⏳ 缺格或 A0 未知 ⇒ **未判**")
