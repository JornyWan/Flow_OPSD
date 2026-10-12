#!/usr/bin/env python
"""Fig.10:少步缺口(同一模型的 40 步 − 4 步)随训练预算的变化。
§4.728 的论断:训练预算**加剧**这个缺口,传播**缓解**它。只画已落盘的点。"""
import sys, os, importlib.util, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SP = "/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec = importlib.util.spec_from_file_location("judge", SP + "/judge.py")
J = importlib.util.module_from_spec(spec); sys.argv = [sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass

def deficit(tag, p):
    f4 = "matched_pi_%s_default_%s_flow4" % (tag, p); f40 = "matched_pi_%s_default_%s_flow40" % (tag, p)
    if not (os.path.exists("outputs/%s.json" % f4) and os.path.exists("outputs/%s.json" % f40)): return None
    a4, a40 = J.load(f4, "open3"), J.load(f40, "open3")
    ks = sorted(set(a4) & set(a40))
    return st.mean(a40[k] - a4[k] for k in ks) / J.MDE

# (ck, tag) —— ck100 用独立的 u100 臂;ck300 来自 300 步那次运行
CANON = [(20, "baseline_20"), (30, "baseline_30"), (50, "baseline_50"),
         (80, "baseline_80"), (100, "baseline_100"), (300, "o3canon300_300")]
MSP   = [(20, "multikF4_20"), (30, "multikF4_30"), (50, "multikF4_50"),
         (80, "multikF4_80"), (100, "multikF4_100"), (300, "o3msp300_300")]
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), sharey=True)
for ax, pset in zip(axes, ("drawbench", "unseen1k")):
    for rows, lab, col, mk, ls in ((CANON, "OPSD (canonical)", "#1f77b4", "o", "-"),
                                   (MSP, "$+$MSP (ours)", "#d62728", "^", "-")):
        xs, ys = [], []
        for ck, tag in rows:
            v = deficit(tag, pset)
            if v is None: continue
            xs.append(ck); ys.append(v)
        if xs: ax.plot(xs, ys, color=col, marker=mk, ls=ls, lw=1.8, ms=5, label=lab)
    ax.set_xscale("log"); ax.set_xticks([20, 30, 50, 80, 100, 300])
    ax.set_xticklabels(["20", "30", "50", "80", "100", "300"])
    ax.grid(alpha=.25, lw=.5); ax.set_xlabel("optimizer updates")
    ax.set_title(pset, fontsize=10)
axes[0].set_ylabel(r"few-step deficit  ($\times$MDE)", fontsize=9)
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("The four-step deficit grows with training budget", fontsize=11, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("paper_figs/fig10_deficit_vs_budget.pdf", bbox_inches="tight")
fig.savefig("paper_figs/fig10_deficit_vs_budget.png", dpi=150, bbox_inches="tight")
print("  ✅ paper_figs/fig10_deficit_vs_budget.{pdf,png}")
