#!/usr/bin/env python
"""少步定性图(CFG=1.5, NFE=4):canonical vs MSP,**§4.666 规则选出的 12 条全部画出**。
§4.686:定量保持现行协议(CFG=1.0),定性改用 CFG=1.5 —— CFG=1.0 会放大高频纹理约 1.5×。
⚠️ 不在 12 条内再按肉眼二次挑选:那会是一层未登记的筛选。规则写进 caption。"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
P = [l.strip() for l in open("paper_figs/qual3/prompts_gain.txt") if l.strip()]
ROWS = [("OPSD", "paper_figs/qualcfg/canon_cfg15_n4", "#1f77b4"),
        ("+MSP (ours)", "paper_figs/qualcfg/msp_cfg15_n4", "#d62728")]
NC = 6                                   # 每块 6 列,12 条拆成两块(4 行)
BLOCKS = [range(0, NC), range(NC, len(P))]
fig, axes = plt.subplots(2 * len(BLOCKS), NC, figsize=(1.74 * NC, 1.98 * 2 * len(BLOCKS)))
def short(s, n=52):
    return s if len(s) <= n else s[:n - 1].rsplit(" ", 1)[0] + "…"
for bi, blk in enumerate(BLOCKS):
    for r, (rlab, d, c) in enumerate(ROWS):
        for i, p in enumerate(blk):
            ax = axes[bi * 2 + r][i]; ax.set_xticks([]); ax.set_yticks([])
            f = "%s/%04d.png" % (d, p)
            if os.path.exists(f): ax.imshow(np.asarray(Image.open(f)))
            else: ax.text(.5, .5, "missing", ha="center", va="center", fontsize=7)
            for s in ax.spines.values(): s.set_linewidth(.5); s.set_color("#bbb")
            if i == 0:
                ax.text(.035, .955, rlab, transform=ax.transAxes, fontsize=7.5, color="white",
                        va="top", weight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc=c, ec="none", alpha=.9))
            if r == len(ROWS) - 1:          # prompt 放在每块的下面一行底部
                ax.set_xlabel(short(P[p]), fontsize=5.6, labelpad=2, wrap=True)
fig.suptitle("Same prompt, same seed, NFE$=$4 (CFG $=1.5$): OPSD vs. $+$MSP", fontsize=9.5, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig("paper_figs/fig6_qualitative_cfg.pdf", bbox_inches="tight")
fig.savefig("paper_figs/fig6_qualitative_cfg.png", dpi=140, bbox_inches="tight")
print("  ✅ paper_figs/fig6_qualitative_cfg.{pdf,png}  (%d prompts × %d arms)" % (len(P), len(ROWS)))
