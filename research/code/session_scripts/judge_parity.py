#!/usr/bin/env python
"""§4.723 的 P 轴判定:我们的 canonical@300 对官方 OPSD 联合@300 行。
容差来自 §4.693 实测的管线一致度,**不得事后调**。
用法: judge_parity.py <tag>   例: judge_parity.py o3canon300_300
"""
import json, os, sys
O = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related"
OFFICIAL = dict(pickscore=25.51, clipscore=0.333, hpsv2=0.389, aesthetic=6.03, imagereward=1.51)
REL = 0.016                      # §4.693: base model 上最大相对差 1.6%(CLIP)
ABS_FLOOR = dict(imagereward=0.032)   # ImgR 接近 0 时相对量发散 ⇒ 取绝对下限
NAMES = [("pickscore","Pick",2),("clipscore","CLIP",3),("hpsv2","HPSv2.1",3),
         ("aesthetic","Aes",2),("imagereward","ImgR",2)]

def load(tag):
    f = os.path.join(O, "outputs", "matched_pi_%s_default_drawbench_flow40.json" % tag)
    if not os.path.exists(f): return None, f
    d = json.load(open(f)); e = d["results"][0] if "results" in d else d
    return {k: v["mean"] for k, v in e["score_stats"].items()}, f

tag = sys.argv[1] if len(sys.argv) > 1 else "o3canon300_300"
got, f = load(tag)
if got is None:
    print("  ⏳ 尚未落地: %s" % f); sys.exit(0)

print("  ═══ P 轴判定 · %s vs 官方 OPSD 联合@300 ═══\n" % tag)
print("  %-9s %9s %9s %9s %8s %7s %s" % ("列","官方","我们","偏差","tol","|d|/tol","在容差内"))
W = 0; D = 0.0; signs = []
for k, lab, nd in NAMES:
    o, g = OFFICIAL[k], got.get(k)
    if g is None: print("  %-9s %9.*f %9s" % (lab, nd, o, "缺")); continue
    dv = g - o
    tol = max(REL*abs(o), ABS_FLOOR.get(k, 0.0))
    r = abs(dv)/tol
    inside = r <= 1.0
    W += inside; D = max(D, r)
    if not inside: signs.append(1 if dv > 0 else -1)
    print("  %-9s %9.*f %9.*f %+9.*f %8.4f %7.2f %s"
          % (lab, nd, o, nd, g, nd, dv, tol, r, "✓" if inside else "✗"))

same_sign = len(set(signs)) <= 1
if   W == 5:                      v = "P1 复现成立"
elif W in (3,4) and D <= 2.0:     v = "P2 大体复现"
elif same_sign:                   v = "P3 系统性偏差(超容差列同号)—— 必须先归因"
else:                             v = "P4 复现不成立(超容差列异号)"
print("\n  W=%d/5  D=%.2f  超容差列符号=%s" % (W, D, signs if signs else "—"))
print("  ⇒ **%s**" % v)
print("  ⇒ 继承官方 14 行 baseline: %s" % ("有依据,按现写法保留" if v.startswith(("P1","P2"))
      else "失去依据,须降级为 context only 并写明偏差"))
