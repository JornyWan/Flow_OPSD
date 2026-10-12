#!/usr/bin/env python
"""512 vs 1024 对比 + 放大裁切,验证网格来源。"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
COLS = [("base @512\n(no reward training)", "paper_figs/qual3/base_nfe4",      "#666666"),
        ("OPSD @512",                        "paper_figs/qual3/canonical_nfe4", "#1f77b4"),
        ("+MSP @512",                        "paper_figs/qual3/oursC_nfe4",     "#d62728"),
        ("OPSD @1024",                       "paper_figs/qual1k/canonical_nfe4","#1f77b4"),
        ("+MSP @1024",                       "paper_figs/qual1k/oursC_nfe4",    "#d62728")]
PROMPTS = [4, 3]          # 两条 prompt
ZOOM = 0.14               # 裁切边长占比(按各自分辨率等比,保证视野相同)
fig, axes = plt.subplots(2 * len(PROMPTS), len(COLS), figsize=(11.4, 4.7 * len(PROMPTS)))
for pi, p in enumerate(PROMPTS):
    for ci, (lab, d, c) in enumerate(COLS):
        f = "%s/%04d.png" % (d, p)
        axf, axz = axes[2 * pi][ci], axes[2 * pi + 1][ci]
        for a in (axf, axz): a.set_xticks([]); a.set_yticks([])
        if not os.path.exists(f):
            axf.text(.5, .5, "missing", ha="center", va="center", fontsize=7); continue
        im = np.asarray(Image.open(f)); N = im.shape[0]
        axf.imshow(im)
        s = int(N * ZOOM); y0 = int(N * 0.42); x0 = int(N * 0.42)   # 同一相对位置
        axf.add_patch(plt.Rectangle((x0, y0), s, s, fill=False, ec="#ffdd00", lw=1.2))
        axz.imshow(im[y0:y0 + s, x0:x0 + s], interpolation="nearest")
        if pi == 0:
            axf.set_title(lab, fontsize=8.5, color=c)
        if ci == 0:
            axf.set_ylabel("full", fontsize=7.5); axz.set_ylabel("%d%% crop" % int(ZOOM * 100), fontsize=7.5)
fig.suptitle("Same prompt, same seed, NFE$=$4.  Crops are the same relative region, shown at native pixels.",
             fontsize=9, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig("paper_figs/rescmp.png", dpi=150, bbox_inches="tight")
print("  ✅ paper_figs/rescmp.png")
