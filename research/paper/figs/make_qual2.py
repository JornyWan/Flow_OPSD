#!/usr/bin/env python
"""Fig.6 v2:同 prompt / 同 seed=42 / 同 sampler,只换 checkpoint 与预算。
**prompt 选取规则(事先固定,无人工筛选)**:canonical@ck100 在 NFE=40 下逐 prompt 的
复合分均值最高的前 K 条 —— 即 canonical 本来就做得好的题,使 4 步退化成为唯一变量。"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

QD = "paper_figs/qual2"
ROWS = [("canonical\n40 NFE", "canonical_nfe40", "#333333"),
        ("canonical\n4 NFE",  "canonical_nfe4",  "#333333"),
        ("+ MSP (C)\n4 NFE",  "oursC_nfe4",      "#d62728"),
        ("+ B$+$C\n4 NFE",    "oursBC_nfe4",     "#d62728")]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
OFF = int(sys.argv[2]) if len(sys.argv) > 2 else 0
OUT = sys.argv[3] if len(sys.argv) > 3 else "paper_figs/fig6_qualitative"

pr = [l.rstrip("\n") for l in open(os.path.join(QD, "prompts_top.txt"))]
fig, axes = plt.subplots(len(ROWS), N, figsize=(2.0*N, 2.28*len(ROWS)))
for r, (lab, d, col) in enumerate(ROWS):
    for c in range(N):
        ax = axes[r][c]; ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_linewidth(.5); s.set_color("#cccccc")
        p = os.path.join(QD, d, "%04d.png" % (c + OFF))
        if os.path.exists(p): ax.imshow(Image.open(p))
        else: ax.text(.5, .5, "missing", ha="center", va="center", fontsize=7)
        if c == 0:
            ax.set_ylabel(lab, fontsize=8.5, rotation=90, labelpad=8, color=col,
                          fontweight="bold" if col == "#d62728" else "normal")
        if r == 0:
            t = pr[c + OFF]
            ax.set_title("\n".join([t[:32], t[32:64], t[64:92]]).strip(), fontsize=5.6, pad=3)
fig.subplots_adjust(wspace=.02, hspace=.02, left=.052, right=.997, top=.885, bottom=.004)
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=145, bbox_inches="tight")
print("  ✅ %s.{pdf,png}  (prompts %d..%d,规则见 docstring)" % (OUT, OFF, OFF+N-1))
