#!/usr/bin/env python
"""Fig.6 定性对比:同 prompt / 同 seed(42)/ 同 NFE,只换 checkpoint。
**按固定规则取前 N 个 prompt,不做任何挑选**(index 0..N-1)。"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

QD = "paper_figs/qual"
ROWS = [("canonical, 40 NFE", "canonical_nfe40"),
        ("ours, 40 NFE",      "ours_nfe40"),
        ("canonical, 4 NFE",  "canonical_nfe4"),
        ("ours, 4 NFE",       "ours_nfe4")]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6

pr = [json.loads(l)["prompt"] for l in open(os.path.join(QD, "ours_nfe4", "prompts.jsonl"))]
fig, axes = plt.subplots(len(ROWS), N, figsize=(1.9*N, 2.15*len(ROWS)))
for r, (lab, d) in enumerate(ROWS):
    for c in range(N):
        ax = axes[r][c]; ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_linewidth(.4); s.set_color("#bbbbbb")
        p = os.path.join(QD, d, "%04d.png" % c)
        if os.path.exists(p): ax.imshow(Image.open(p))
        else: ax.text(.5, .5, "missing", ha="center", va="center", fontsize=7)
        if c == 0:
            ax.set_ylabel(lab, fontsize=8, rotation=90, labelpad=6,
                          fontweight="bold" if lab.startswith("ours") else "normal",
                          color="#d62728" if lab.startswith("ours") else "#333333")
        if r == 0:
            ax.set_title("\n".join([pr[c][:34], pr[c][34:68]]).strip(), fontsize=5.5, pad=3)
fig.subplots_adjust(wspace=.03, hspace=.03, left=.055, right=.995, top=.90, bottom=.01)
fig.savefig("paper_figs/fig6_qualitative.pdf", bbox_inches="tight")
fig.savefig("paper_figs/fig6_qualitative.png", dpi=140, bbox_inches="tight")
print("  ✅ fig6_qualitative.{pdf,png}  (前 %d 个 prompt,无挑选)" % N)
