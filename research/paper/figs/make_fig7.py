#!/usr/bin/env python
"""Fig.7(新框架的核心图):三种"reward 目标该施加在哪里"的答案 vs NFE。
A=换一个地方(查询σ↑,免费) B=换整条轨迹(6 步 rollout) C=多加几个地方(ours)。
只画已落盘的点。"""
import sys, os, importlib.util
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SP = "/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec = importlib.util.spec_from_file_location("judge", SP + "/judge.py")
J = importlib.util.module_from_spec(spec); sys.argv = [sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
MDE, NFE, FLOOR = J.MDE, [2, 4, 6, 10, 20, 40], 1.23
SER = [("A: move the query state ($\\sigma$ 0.278$\\to$0.465), free", "qstrain7_100", "basenp3_100", "#1f77b4", "o"),
       ("B: shorten the training rollout (10$\\to$6), 0.90$\\times$", "base6_100", "baseline_100", "#2ca02c", "s"),
       ("C: propagate to extra states (ours), 1.20$\\times$", "multikF4_100", "baseline_100", "#d62728", "^")]
def pair(a, b, p, n, ax):
    fa = "matched_pi_%s_default_%s_flow%d" % (a, p, n); fb = "matched_pi_%s_default_%s_flow%d" % (b, p, n)
    if not (os.path.exists("outputs/%s.json" % fa) and os.path.exists("outputs/%s.json" % fb)): return None
    return J.paired(J.load(fa, ax), J.load(fb, ax))
fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), sharex=True)
for r, (ax_metric, ylab) in enumerate([("open3", r"reported composite  ($\times$MDE)"),
                                       ("imagereward", r"held-out ImageReward  (paired $t$)")]):
    for c, pset in enumerate(("drawbench", "unseen1k")):
        ax = axes[r][c]
        for lab, a, b, col, m in SER:
            xs, ys = [], []
            for n in NFE:
                rr = pair(a, b, pset, n, ax_metric)
                if rr is None: continue
                xs.append(n); ys.append(rr[1] / MDE if ax_metric == "open3" else rr[3])
            # 点数 <4 时用虚线 + 标注,避免"两点连成直线"被误读为真实形状
            if xs: ax.plot(xs, ys, color=col, marker=m, lw=1.8, ms=5,
                           ls="-" if len(xs) >= 4 else "--", alpha=1.0 if len(xs) >= 4 else .55,
                           label=(lab + ("" if len(xs) >= 4 else "  [%d pts]" % len(xs))) if (r == 0 and c == 0) else None)
        if ax_metric == "open3": ax.axhspan(-FLOOR, FLOOR, color="#999", alpha=.16, lw=0)
        else: ax.axhspan(-2, 2, color="#999", alpha=.16, lw=0)
        ax.axhline(0, color="k", lw=.7)
        ax.set_xscale("log", base=2); ax.set_xticks(NFE); ax.set_xticklabels([str(n) for n in NFE])
        ax.grid(alpha=.25, lw=.5)
        if r == 1: ax.set_xlabel("NFE (sampling steps)")
        if c == 0: ax.set_ylabel(ylab, fontsize=9)
        if r == 0: ax.set_title(pset, fontsize=10)
axes[0][0].legend(fontsize=7.5, loc="upper right")
fig.suptitle("Where should the on-policy reward target be applied?", fontsize=11.5, y=.985)
fig.tight_layout(rect=[0, 0, 1, .97])
fig.savefig("paper_figs/fig7_where_to_apply.pdf"); fig.savefig("paper_figs/fig7_where_to_apply.png", dpi=150)
print("  ✅ fig7_where_to_apply.{pdf,png}")
