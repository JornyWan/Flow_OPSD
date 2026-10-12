#!/usr/bin/env python
"""Teaser(Fig.1):(a) 质量-NFE 曲线 (b) NFE=4 定性对比。
骨架照 DiffusionOPD 原文 fig:teaser —— 第一页把主张视觉化。
数据全部来自已落盘的 outputs/matched_pi_*.json 与 paper_figs/qual2/;缺的点跳过,不插值。"""
import sys, os, glob, importlib.util
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

SP = "/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec = importlib.util.spec_from_file_location("judge", SP + "/judge.py")
J = importlib.util.module_from_spec(spec); sys.argv = [sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
NFE = [2, 4, 6, 10, 20, 40]

def level(tag, adapter, pset, nfe):
    f = "matched_%s_%s_%s_flow%d" % (tag, adapter, pset, nfe)
    if not os.path.exists("outputs/%s.json" % f): return None
    A = J.load(f, "open3"); return sum(A.values()) / len(A)

LINES = [("SD3.5-M (no reward training)", "dflt_basemodel",  "none",    "#9e9e9e", "o", "--"),
         ("+ OPSD (prior state of the art)", "pi_baseline_100", "default", "#1f77b4", "s", "-"),
         ("+ OPSD + MSP (ours)",          "pi_multikF4_100", "default", "#d62728", "^", "-")]
# 注:(B+C) 只有 NFE=4/40 两点,放 teaser 会误导(<4 点不画实线),故只画主方法 C

fig = plt.figure(figsize=(12.2, 4.3))
gs = GridSpec(2, 5, figure=fig, width_ratios=[1.55, 1, 1, 1, 1], wspace=0.06, hspace=0.06)

# ── (a) 质量 vs NFE ─────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[:, 0])
for lab, tag, ad, c, m, ls in LINES:
    xs, ys = [], []
    for n in NFE:
        v = level(tag, ad, "drawbench", n)
        if v is None: print("  [skip] %s NFE=%d" % (tag, n), file=sys.stderr); continue
        xs.append(n); ys.append(v)
    if xs: ax.plot(xs, ys, color=c, marker=m, ls=ls, lw=1.9, ms=4.5, label=lab)
ref = level("pi_baseline_100", "default", "drawbench", 40)
if ref is not None:
    ax.axhline(ref, color="#1f77b4", ls=":", lw=1.1)
    ax.annotate("prior SOTA @ 40 steps", (2.05, ref), fontsize=6.5, color="#1f77b4", va="bottom")
ax.set_xscale("log", base=2); ax.set_xticks(NFE); ax.set_xticklabels([str(n) for n in NFE])
ax.set_xlabel("NFE (sampling steps)", fontsize=8.5)
ax.set_ylabel("reported composite  (pick/26 + clip + hpsv2)", fontsize=8.5)
ax.tick_params(labelsize=7.5); ax.grid(alpha=.25, lw=.5)
ax.legend(fontsize=6.6, loc="lower right", framealpha=.92)
ax.set_title("(a) quality vs. budget", fontsize=8.5, loc="left")
# 把主张标到曲线上:NFE=4 处两条线的差
y_c = level("pi_baseline_100", "default", "drawbench", 4)
y_o = level("pi_multikF4_100", "default", "drawbench", 4)
if y_c is not None and y_o is not None:
    ax.annotate("", xy=(4, y_o), xytext=(4, y_c),
                arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1.1))
    ax.annotate(r"$+10.7\times$MDE" "\n" "(83% win)", xy=(4.45, (y_o + y_c) / 2 - 0.02),
                fontsize=6.8, color="#d62728", va="center", ha="left")

# ── (b) NFE=4 定性对比 ─────────────────────────────────────────────────────
# §4.748:原先用 qual2/ 按"主体清晰"挑的 4 条,两行肉眼几乎无差别 —— teaser 是第一印象,
# 等于开篇说"没差别"。改用 §4.666 规则选出的 12 条(CFG=1.5,与 Fig.6 同工作点),
# 并取覆盖**三类可核验差异**的四条:计数 / 空间关系 / 文字。全部 12 条在 Fig.6,读者可自核。
PICK = [2, 7, 4, 6]          # two dogs(计数)· racket underneath(空间)· zebra underneath(空间)· 文字
ROWS = [("OPSD", "canon_cfg15_n4"), ("+ MSP (ours)", "msp_cfg15_n4")]
for r, (rlab, sub) in enumerate(ROWS):
    for c, p in enumerate(PICK):
        f = "paper_figs/qualcfg/%s/%04d.png" % (sub, p)
        axi = fig.add_subplot(gs[r, c + 1]); axi.set_xticks([]); axi.set_yticks([])
        if os.path.exists(f): axi.imshow(mpimg.imread(f))
        else: axi.text(.5, .5, "missing", ha="center", va="center", fontsize=6)
        for sp in axi.spines.values(): sp.set_linewidth(.5); sp.set_color("#bbbbbb")
        if c == 0:                      # 行标签放图内左上角(published 常见做法)
            col = "#d62728" if "ours" in rlab else "#1f77b4"
            axi.text(.035, .955, rlab, transform=axi.transAxes, fontsize=7.8, color="white",
                     va="top", ha="left", weight="bold",
                     bbox=dict(boxstyle="round,pad=0.25", fc=col, ec="none", alpha=.88))
        if r == 0 and c == 0:
            axi.set_title("(b) same prompt, same seed, NFE$=$4 (CFG$=$1.5)", fontsize=8.5, loc="left")
fig.savefig("paper_figs/fig0_teaser.pdf", bbox_inches="tight")
fig.savefig("paper_figs/fig0_teaser.png", dpi=150, bbox_inches="tight")
print("  ✅ paper_figs/fig0_teaser.pdf")
