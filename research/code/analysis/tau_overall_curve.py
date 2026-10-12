#!/usr/bin/env python
"""用**整体分数**画 τ 曲线,回答"τ 是否改变收敛速度或最终上限"。

逐轴看容易被单轴噪声带偏;整体分数(相对官方 Student 的 6 轴比值均值)有 2σ=0.63% 的判定阈值,
是更干净的比较量。对每个 τ 家族给出:
  上限 = step>=600 的所有点的均值(带 n 和 sem)
  速度 = 首次达到"自身最终值 95%(以 step61 为起点归一)"的 step
"""
import importlib.util, glob, os, re, json, math, statistics as st, sys

O = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related"
src = open(f"{O}/overall_score.py").read().split("ours = collect(N)")[0]
M = type(importlib)("M"); exec(compile(src, "M", "exec"), M.__dict__)

ours = M.collect(M.N)
ref = ours["Student"]
AX = M.TRAINED
SIG = M.retest_sigma(M.collect(M.R), ours)
SE = math.sqrt(sum((SIG.get(a, 0) / ref[a]) ** 2 for a in AX)) / len(AX)

FAM = {"τ=0.2": "tau02", "τ=0.5": "caopd", "τ=1.0": "tau10"}
print(f"整体分数(6 轴 Ratio, 官方 Student=1.000)   单点 1σ={SE*100:.2f}%  2σ={SE*200:.2f}%\n")
curves = {}
for label, fam in FAM.items():
    pts = []
    for m, v in ours.items():
        mm = re.fullmatch(rf"{fam}_?(\d+)", m)
        if mm and all(a in v for a in AX):
            pts.append((int(mm.group(1)), st.mean(v[a] / ref[a] for a in AX)))
    pts.sort()
    if not pts:
        print(f"{label:8s} 尚无评测数据(该 arm 还没跑完/评完)"); continue
    curves[label] = pts
    print(f"{label:8s} ({len(pts)} 点)  " + "  ".join(f"{s}:{r:.4f}" for s, r in pts))

print()
for label, pts in curves.items():
    tail = [r for s, r in pts if s >= 600]
    if len(tail) >= 2:
        ceil_m, ceil_se = st.mean(tail), st.pstdev(tail) / math.sqrt(len(tail))
    else:
        ceil_m, ceil_se = (tail[0], float("nan")) if tail else (float("nan"),) * 2
    lo = pts[0][1]
    tgt = lo + 0.95 * (ceil_m - lo)
    spd = next((s for s, r in pts if r >= tgt), None)
    print(f"{label:8s} 上限={ceil_m:.4f}±{ceil_se:.4f}(n={len(tail)})   "
          f"达到自身 95% 的 step={spd}")

ls = list(curves)
if len(ls) >= 2:
    print("\n两两比较上限(判据: 差值是否超过单点 2σ):")
    for i in range(len(ls)):
        for j in range(i + 1, len(ls)):
            a, b = ls[i], ls[j]
            ta = [r for s, r in curves[a] if s >= 600]; tb = [r for s, r in curves[b] if s >= 600]
            if not ta or not tb: continue
            d = st.mean(ta) - st.mean(tb)
            print(f"  {a} - {b} = {d*100:+.2f}%   " +
                  ("打平(噪声内)" if abs(d) < SE * 2 else "★ 超出噪声"))
