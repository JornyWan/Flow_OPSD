#!/usr/bin/env python
"""Fig.11(§4.764 → §4.846 改):MSP − canonical 随 optimizer updates 的变化,**两对** 300-update 运行
(seed 1 = o3msp300/o3canon300,seed 2 = msp300s2/canon300s2),各自同 seed 配对,ck100/150/200/250/300。"""
import sys, os, importlib.util
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SP = "/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec = importlib.util.spec_from_file_location("judge", SP + "/judge.py")
J = importlib.util.module_from_spec(spec); sys.argv = [sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass

CKS = [100, 150, 200, 250, 300]
PAIRS = {1: ("o3msp300", "o3canon300"), 2: ("msp300s2", "canon300s2")}
def gain(ck, n, p="drawbench", seed=1):
    fa = "matched_pi_%s_%d_default_%s_flow%d" % (PAIRS[seed][0], ck, p, n)
    fb = "matched_pi_%s_%d_default_%s_flow%d" % (PAIRS[seed][1], ck, p, n)
    if not (os.path.exists("outputs/%s.json" % fa) and os.path.exists("outputs/%s.json" % fb)): return None
    return J.paired(J.load(fa, "open3"), J.load(fb, "open3"))[1] / J.MDE

fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.axhspan(-2.5, 2.5, color="#999", alpha=.15, lw=0)
ax.axhline(0, color="k", lw=.8)
for seed, ls in ((1, "-"), (2, "--")):
    for n, col, mk in ((4, "#d62728", "^"), (40, "#1f77b4", "s")):
        xs, ys = [], []
        for ck in CKS:
            v = gain(ck, n, seed=seed)
            if v is None: continue
            xs.append(ck); ys.append(v)
        if xs:
            ax.plot(xs, ys, color=col, marker=mk, ls=ls, lw=1.9, ms=6, label="NFE$=$%d, seed %d" % (n, seed))
            print("  seed %d NFE=%d: %s" % (seed, n, " ".join("%d:%+.2f" % xy for xy in zip(xs, ys))))
ax.set_xticks(CKS); ax.set_xlabel("optimizer updates")
ax.set_ylabel(r"propagation gain  ($\times$MDE)", fontsize=9)
ax.grid(alpha=.25, lw=.5); ax.legend(fontsize=7.5, loc="lower left", ncol=2)
ax.set_title("Propagation gain versus training length, two seeds", fontsize=10.5)
fig.tight_layout()
fig.savefig("paper_figs/fig11_gain_vs_updates.pdf", bbox_inches="tight")
fig.savefig("paper_figs/fig11_gain_vs_updates.png", dpi=150, bbox_inches="tight")
print("  ✅ fig11_gain_vs_updates.{pdf,png}")
