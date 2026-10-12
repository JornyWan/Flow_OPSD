#!/usr/bin/env python
"""伪影对比:base / 我们的 OPSD / 我们的 MSP / 官方三个 release。
裁切位置 = base 图里**方差最低**的 64×64 区块(纹理在平坦区最显眼),各臂用同一位置。
列头标注实测 8px 网格强度(1.0 = 无)。"""
import os, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

COLS = [("base\n(no reward training)",  "paper_figs/qual3/base_nfe4",                 "#666666"),
        ("our OPSD repro\n(open3)",      "paper_figs/qual3/canonical_nfe4",            "#1f77b4"),
        ("our +MSP\n(open3)",            "paper_figs/qual3/oursC_nfe4",                "#d62728"),
        ("official OPSD\nsd35-m-hpsv2",  "paper_figs/qualofc/opsdOfficialHps_nfe4",    "#2ca02c"),
        ("official OPSD\nsd35-m-pickscore","paper_figs/qualofc/opsdOfficialPick_nfe4", "#2ca02c"),
        ("official OPD\nStudent",        "paper_figs/qualofc/opdOfficialStudent_nfe4", "#9467bd")]
PROMPTS = [4, 3, 9]
CROP = 72

def grid8(f):
    im = np.asarray(Image.open(f).convert("L"), dtype=np.float32); im -= im.mean(); N = im.shape[0]
    r = []
    for spec in (np.abs(np.fft.rfft(im, axis=1)).mean(0), np.abs(np.fft.rfft(im, axis=0)).mean(1)):
        k = int(round(N / 8)); lo, hi = max(1, k - 14), min(len(spec) - 1, k + 15)
        base = np.median(np.r_[spec[lo:k - 2], spec[k + 3:hi]]); r.append(spec[k] / (base + 1e-9))
    return float(np.mean(r))

def flattest(f, s=CROP):
    """返回 base 图里方差最低的 s×s 区块左上角(步长 24,避开边缘)。"""
    im = np.asarray(Image.open(f).convert("L"), dtype=np.float32)
    N = im.shape[0]; best = (1e18, N // 3, N // 3)
    for y in range(40, N - s - 40, 24):
        for x in range(40, N - s - 40, 24):
            v = im[y:y + s, x:x + s].var()
            if v < best[0]: best = (v, y, x)
    return best[1], best[2]

# 列头的网格强度:对 12 张取均值
GRID = {}
for lab, d, c in COLS:
    fs = sorted(glob.glob(d + "/*.png"))
    GRID[d] = np.mean([grid8(f) for f in fs]) if fs else float("nan")

fig, axes = plt.subplots(2 * len(PROMPTS), len(COLS), figsize=(2.05 * len(COLS), 4.25 * len(PROMPTS)))
for pi, p in enumerate(PROMPTS):
    bf = "%s/%04d.png" % (COLS[0][1], p)
    y0, x0 = flattest(bf) if os.path.exists(bf) else (200, 200)
    for ci, (lab, d, c) in enumerate(COLS):
        f = "%s/%04d.png" % (d, p)
        axf, axz = axes[2 * pi][ci], axes[2 * pi + 1][ci]
        for a in (axf, axz): a.set_xticks([]); a.set_yticks([])
        if not os.path.exists(f):
            axf.text(.5, .5, "missing", ha="center", va="center", fontsize=7); continue
        im = np.asarray(Image.open(f))
        axf.imshow(im)
        axf.add_patch(plt.Rectangle((x0, y0), CROP, CROP, fill=False, ec="#ffdd00", lw=1.3))
        axz.imshow(im[y0:y0 + CROP, x0:x0 + CROP], interpolation="nearest")
        if pi == 0:
            axf.set_title("%s\n8px grid = %.2f" % (lab, GRID[d]), fontsize=7.6, color=c, pad=4)
        if ci == 0:
            axz.set_ylabel("flattest %dpx crop" % CROP, fontsize=6.8)
fig.suptitle("All at 512, same prompt, same seed, NFE$=$4.  Yellow box = lowest-variance region of the "
             "base image; crops shown at native pixels (no interpolation).", fontsize=9, y=0.997)
fig.tight_layout(rect=[0, 0, 1, 0.986])
fig.savefig("paper_figs/artifact_cmp.png", dpi=150, bbox_inches="tight")
print("  ✅ paper_figs/artifact_cmp.png")
for lab, d, c in COLS: print("    %-34s 8px grid = %.2f" % (lab.replace("\n", " "), GRID[d]))
