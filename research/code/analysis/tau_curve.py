#!/usr/bin/env python
"""τ 扫描的曲线对比。

CA-OPD 的核心主张是"τ 交换收敛速度与最终天花板"(τ 小 -> 更快但更低)。
本工具直接检验它,而不只是列数字:
  - 天花板  = 末段(>=600 步)各轴的均值
  - 收敛速度 = 首次达到"该 arm 自身最终值 95%"所需步数
判据:若三条 τ 曲线的天花板与速度差异都落在采样噪声内,则该主张不成立,应如实写。
"""
import json, os, sys, math, statistics

O = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related"
N, R = f"{O}/outputs/eval", f"{O}/CA-OPD/results/eval"
AX = ["geneval", "ocr", "pickscore", "hpsv2", "aesthetic"]
DS = {"geneval": "geneval", "ocr": "ocr", "pickscore": "drawbench",
      "hpsv2": "drawbench", "aesthetic": "drawbench"}
ARMS = [("τ=0.2", "tau02_{}"), ("τ=0.5", "caopd{}"), ("τ=1.0", "tau10_{}"), ("τ=2.0", "tau20_{}")]
STEPS = [61, 121, 181, 211, 241, 271, 301, 331, 361, 391, 451, 481, 541, 601, 631, 721, 781, 841, 901, 991]

def val(name, ax):
    for b in (N, R):
        f = os.path.join(b, f"{name}_{DS[ax]}", "average_scores.json")
        try:
            d = json.load(open(f))
            if ax in d: return d[ax]
        except Exception: pass
    return None

def curve(pat):
    out = {}
    for s in STEPS:
        row = {ax: val(pat.format(s), ax) for ax in AX}
        # caopd 的 721 点在原记录里叫 caopd721,regret 类似;直接用 pat 即可
        if any(v is not None for v in row.values()): out[s] = row
    return out

data = {lab: curve(pat) for lab, pat in ARMS}
data = {k: v for k, v in data.items() if v}

print("=" * 78)
print("τ 扫描曲线对比  (n=256, gs=4.5;· 表示该点未评测)")
print("=" * 78)
for ax in AX:
    print(f"\n--- {ax} ---")
    hdr = f"{'step':>6}" + "".join(f"{lab:>11}" for lab in data)
    print(hdr); print("-" * len(hdr))
    for s in STEPS:
        cells = [data[lab].get(s, {}).get(ax) for lab in data]
        if all(c is None for c in cells): continue
        print(f"{s:>6}" + "".join(f"{c:>11.3f}" if c is not None else f"{'·':>11}" for c in cells))

print("\n" + "=" * 78)
print("主张检验")
print("=" * 78)
for ax in AX:
    print(f"\n{ax}:")
    for lab in data:
        pts = sorted((s, r[ax]) for s, r in data[lab].items() if r.get(ax) is not None)
        if len(pts) < 3: print(f"  {lab:>7}: 点太少"); continue
        late = [v for s, v in pts if s >= 600]
        ceil = statistics.mean(late) if late else pts[-1][1]
        thr = ceil * 0.95
        hit = next((s for s, v in pts if v >= thr), None)
        sd = statistics.pstdev(late) if len(late) > 1 else float('nan')
        print(f"  {lab:>7}: 天花板(>=600步均值) {ceil:.4f} ±{sd:.4f}   达到95%所需步数 {hit}")
print("\n判读: 若各 τ 的天花板差异 < 末段波动(±)，且 95% 步数无单调趋势，")
print("      则'τ 交换速度与天花板'不成立 —— 这是需要如实报告的结果，不是失败。")
