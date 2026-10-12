#!/usr/bin/env python
"""Fig.2(质量-NFE 曲线)与 Fig.3(优势 vs NFE)。数据全部来自 outputs/matched_pi_*.json。
用法: python paper_figs/make_figs.py            # 从 opd_related/ 下跑
只画**已落盘**的点;缺的点跳过并在 stderr 报告,绝不插值或补数。"""
import sys, os, json, math, importlib.util
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = "/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec = importlib.util.spec_from_file_location("judge", SP + "/judge.py")
J = importlib.util.module_from_spec(spec); sys.argv = [sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
MDE = J.MDE
NFE = [2, 4, 6, 10, 20, 40]

def fname(tag, adapter, pset, nfe):
    return "matched_%s_%s_%s_flow%d" % (tag, adapter, pset, nfe)

def level(tag, adapter, pset, nfe):
    f = fname(tag, adapter, pset, nfe)
    if not os.path.exists("outputs/%s.json" % f): return None
    A = J.load(f, "open3"); return sum(A.values()) / len(A)

def paired(tag_a, tag_b, pset, nfe):
    fa, fb = fname(tag_a, "default", pset, nfe), fname(tag_b, "default", pset, nfe)
    if not (os.path.exists("outputs/%s.json" % fa) and os.path.exists("outputs/%s.json" % fb)): return None
    return J.paired(J.load(fa, "open3"), J.load(fb, "open3"))   # n, mean, se, t, win%

# ── Fig.2:质量 vs NFE ───────────────────────────────────────────────────────
LINES = [("SD3.5-M (no reward training)", "dflt_basemodel",  "none",    "#888888", "o", "--"),
         ("+ OPSD (canonical)",           "pi_baseline_100", "default", "#1f77b4", "s", "-"),
         ("+ OPSD + ours (multikF4)",     "pi_multikF4_100", "default", "#d62728", "^", "-")]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
for ax, pset in zip(axes, ("drawbench", "unseen1k")):
    ref = level("pi_baserep_50", "default", pset, 40)      # canonical@40 = 等质量参考线
    for lab, tag, ad, c, m, ls in LINES:
        xs, ys = [], []
        for n in NFE:
            v = level(tag, ad, pset, n)
            if v is None: print("  [skip] %s %s NFE=%d" % (tag, pset, n), file=sys.stderr); continue
            xs.append(n); ys.append(v)
        if xs: ax.plot(xs, ys, color=c, marker=m, ls=ls, lw=1.8, ms=5, label=lab)
    if ref is not None:
        ax.axhline(ref, color="#1f77b4", ls=":", lw=1.2)
        ax.annotate("canonical @ 40 NFE", (2.05, ref), fontsize=7, color="#1f77b4",
                    va="bottom", ha="left")
    ax.set_xscale("log", base=2); ax.set_xticks(NFE); ax.set_xticklabels([str(n) for n in NFE])
    ax.set_xlabel("NFE (sampling steps)"); ax.set_title(pset, fontsize=10)
    ax.grid(alpha=.25, lw=.5)
    # 放大插图:4–40 NFE 才是有意义的工作区(2 步两个模型都已崩坏)
    axi = ax.inset_axes([0.40, 0.10, 0.57, 0.44])
    for lab, tag, ad, c, m, ls in LINES:
        xs, ys = [], []
        for n in NFE:
            if n < 4: continue
            v = level(tag, ad, pset, n)
            if v is None: continue
            xs.append(n); ys.append(v)
        if xs: axi.plot(xs, ys, color=c, marker=m, ls=ls, lw=1.5, ms=4)
    if ref is not None: axi.axhline(ref, color="#1f77b4", ls=":", lw=1.0)
    axi.set_xscale("log", base=2); axi.set_xticks([4,6,10,20,40])
    axi.set_xticklabels(["4","6","10","20","40"], fontsize=6)
    axi.tick_params(labelsize=6); axi.grid(alpha=.2, lw=.4)
    _v=[level(t,a2,pset,n) for _,t,a2,_,_,_ in LINES[1:] for n in NFE if n>=4]
    _v=[x for x in _v if x is not None]
    if _v: axi.set_ylim(min(_v)-0.004, max(_v)+0.004)
    axi.set_title("zoom: NFE$\\geq$4", fontsize=6)
axes[0].set_ylabel("reported composite  (pick/26 + clip + hpsv2)")
axes[0].legend(fontsize=7.5, loc="upper left", framealpha=.9)
fig.tight_layout(); fig.savefig("paper_figs/fig2_quality_vs_nfe.pdf"); fig.savefig("paper_figs/fig2_quality_vs_nfe.png", dpi=160)
print("  ✅ fig2_quality_vs_nfe.{pdf,png}")

# ── Fig.3:优势(×MDE)vs NFE,阴影=噪声地板 ──────────────────────────────
FLOOR = 1.23   # §4.614:canonical 两两差的最大 |Δ|,单位 ×MDE
fig, ax = plt.subplots(figsize=(6.2, 4.2))
# 族形状:同 ck50、同对照 baserep_50 —— 只换传播的状态集(§4.597 的部署网格原则)
# ck100、同对照 baseline_100 —— 与论文其余部分同一 checkpoint
SERIES = [(r"$\mathcal{K}=\{3,6\}$ (ours, reaches $\sigma=0.858$) - drawbench", "pi_multikF4_100", "pi_baseline_100", "drawbench", "#d62728", "^", "-"),
          (r"$\mathcal{K}=\{3,6\}$ (ours, reaches $\sigma=0.858$) - unseen1k",  "pi_multikF4_100", "pi_baseline_100", "unseen1k",  "#2ca02c", "o", "-"),
          (r"$\mathcal{K}=\{5,6,7\}$ (3 states, tops out at $\sigma=0.707$) - drawbench", "pi_multik4_100", "pi_baseline_100", "drawbench", "#d62728", "^", ":"),
          (r"$\mathcal{K}=\{5,6,7\}$ (3 states, tops out at $\sigma=0.707$) - unseen1k",  "pi_multik4_100", "pi_baseline_100", "unseen1k",  "#2ca02c", "o", ":")]
for lab, arm, ctl, pset, c, m, ls in SERIES:
    xs, ys, es = [], [], []
    for n in NFE:
        r = paired(arm, ctl, pset, n)
        if r is None: continue
        xs.append(n); ys.append(r[1] / MDE); es.append(r[2] / MDE)
    if xs: ax.errorbar(xs, ys, yerr=es, color=c, marker=m, ls=ls, lw=1.8 if ls == "-" else 1.2,
                       ms=5 if ls == "-" else 3.5, capsize=2, alpha=1.0 if ls == "-" else .55, label=lab)
ax.axhspan(-FLOOR, FLOOR, color="#999999", alpha=.18, lw=0)
ax.annotate("canonical-vs-canonical floor", (2.05, FLOOR), fontsize=7, color="#555555", va="bottom")
ax.axhline(0, color="k", lw=.6)
ax.set_xscale("log", base=2); ax.set_xticks(NFE); ax.set_xticklabels([str(n) for n in NFE])
ax.set_xlabel("NFE (sampling steps)"); ax.set_ylabel(r"advantage over canonical  ($\times$MDE)")
ax.legend(fontsize=6.5, loc="upper right"); ax.grid(alpha=.25, lw=.5)
fig.tight_layout(); fig.savefig("paper_figs/fig3_advantage_vs_nfe.pdf"); fig.savefig("paper_figs/fig3_advantage_vs_nfe.png", dpi=160)
print("  ✅ fig3_advantage_vs_nfe.{pdf,png}")

# ── Fig.4:flow-4 优势 vs 训练 checkpoint(效应不依赖早停点)───────────────
CKS = [20, 30, 50, 80, 100]
fig, ax = plt.subplots(figsize=(5.2, 3.8))
for pset, c, m in (("drawbench", "#d62728", "^"), ("unseen1k", "#2ca02c", "o")):
    xs, ys, es = [], [], []
    for ck in CKS:
        fa = "matched_pi_multik4_%d_default_%s_flow4" % (ck, pset)
        fb = "matched_pi_baserep_%d_default_%s_flow4" % (ck, pset)
        if not (os.path.exists("outputs/%s.json" % fa) and os.path.exists("outputs/%s.json" % fb)):
            print("  [skip] C ck%d %s" % (ck, pset), file=sys.stderr); continue
        n, mu, se, t, w = J.paired(J.load(fa, "open3"), J.load(fb, "open3"))
        xs.append(ck); ys.append(mu / MDE); es.append(se / MDE)
    if xs: ax.errorbar(xs, ys, yerr=es, color=c, marker=m, lw=1.8, ms=5, capsize=2, label=pset)
ax.axhspan(-FLOOR, FLOOR, color="#999999", alpha=.18, lw=0)
ax.axhline(0, color="k", lw=.6)
ax.set_xlabel("optimizer updates (checkpoint)")
ax.set_ylabel(r"advantage at NFE=4  ($\times$MDE)")
ax.set_xticks(CKS); ax.set_ylim(bottom=-2)
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.25, lw=.5)
fig.tight_layout(); fig.savefig("paper_figs/fig4_advantage_vs_ckpt.pdf"); fig.savefig("paper_figs/fig4_advantage_vs_ckpt.png", dpi=160)
print("  ✅ fig4_advantage_vs_ckpt.{pdf,png}")

# ── Fig.5:逐轴 t 值 vs NFE(满步是置换,少步是全轴同向)────────────────
AXES5 = [("pickscore", "#1f77b4", "o"), ("clipscore", "#ff7f0e", "s"),
         ("hpsv2", "#2ca02c", "^"), ("imagereward", "#d62728", "D")]
fig, axs = plt.subplots(1, 2, figsize=(10.0, 3.9), sharey=True)
for ax, pset in zip(axs, ("drawbench", "unseen1k")):
    for a, c, m in AXES5:
        xs, ys = [], []
        for n in NFE:
            fa = "matched_pi_multikF4_100_default_%s_flow%d" % (pset, n)
            fb = "matched_pi_baseline_100_default_%s_flow%d" % (pset, n)
            if not (os.path.exists("outputs/%s.json" % fa) and os.path.exists("outputs/%s.json" % fb)): continue
            xs.append(n); ys.append(J.paired(J.load(fa, a), J.load(fb, a))[3])
        if xs: ax.plot(xs, ys, color=c, marker=m, lw=1.6, ms=4.5,
                       label=a + (" (held-out)" if a == "imagereward" else ""))
    ax.axhline(0, color="k", lw=.8); ax.axhspan(-2, 2, color="#999999", alpha=.15, lw=0)
    ax.set_xscale("log", base=2); ax.set_xticks(NFE); ax.set_xticklabels([str(n) for n in NFE])
    ax.set_xlabel("NFE (sampling steps)"); ax.set_title(pset, fontsize=10); ax.grid(alpha=.25, lw=.5)
axs[0].set_ylabel("paired $t$  (ours $-$ canonical)")
axs[0].legend(fontsize=7.5, loc="upper right")
fig.tight_layout(); fig.savefig("paper_figs/fig5_peraxis_vs_nfe.pdf"); fig.savefig("paper_figs/fig5_peraxis_vs_nfe.png", dpi=160)
print("  ✅ fig5_peraxis_vs_nfe.{pdf,png}")
