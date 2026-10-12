#!/usr/bin/env python
"""Fig.1 方法示意。σ 值取自真实 scheduler(EXPERIMENT_LOG §4.638 复现),非示意值。"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SIG = [1.00000, 0.96013, 0.91335, 0.85769, 0.79037, 0.70728,
       0.60215, 0.46488, 0.27805, 0.00893]          # 10-step SD3.5-M, shift=3.0
KQ, EXTRA = 8, [3, 6]                                # query state / ours: RPI_MULTIK_STATES="3,6"
C_BASE, C_Q, C_X = "#9aa5b1", "#1f77b4", "#d62728"

fig, ax = plt.subplots(figsize=(9.2, 3.5))
xs = list(range(len(SIG)))
ax.plot(xs, [0]*len(SIG), color=C_BASE, lw=1.4, zorder=1)
for i, s in enumerate(SIG):
    is_q, is_x = (i == KQ), (i in EXTRA)
    c = C_Q if is_q else (C_X if is_x else C_BASE)
    ax.scatter([i], [0], s=190 if (is_q or is_x) else 80, color=c, zorder=3,
               edgecolors="white", linewidths=1.2)
    ax.text(i, -0.115, r"$\sigma{=}%.3f$" % s, ha="center", va="top",
            fontsize=6.8, color="#333333" if (is_q or is_x) else "#888888")
    ax.text(i, 0.055, r"$z_{%d}$" % i, ha="center", va="bottom", fontsize=7.5,
            color=c, fontweight="bold" if (is_q or is_x) else "normal")

# 查询态:构造 y+
ax.add_patch(FancyBboxPatch((KQ-1.42, 0.30), 2.84, 0.30, boxstyle="round,pad=0.03",
                            fc="#eaf2fb", ec=C_Q, lw=1.1, zorder=2))
ax.text(KQ, 0.45, r"build $y^{+}$  (1 reward gradient)", ha="center", va="center",
        fontsize=8.2, color=C_Q, fontweight="bold")
ax.add_patch(FancyArrowPatch((KQ+0.30, 0.28), (KQ+0.30, 0.055), arrowstyle="-|>", mutation_scale=11,
                             color=C_Q, lw=1.3, zorder=4))

# OPSD:只在查询态施加损失
ax.text(KQ, -0.30, "OPSD: loss here only", ha="center", va="top", fontsize=8,
        color=C_Q, style="italic")

# ours:把同一个 y+ 铺到额外态
for k in EXTRA:
    ax.add_patch(FancyArrowPatch((KQ-0.15, 0.30), (k, 0.085), arrowstyle="-|>",
                                 mutation_scale=11, color=C_X, lw=1.5,
                                 connectionstyle="arc3,rad=0.28", zorder=4))
ax.text(4.5, 0.70, r"\textbf{ours}: reuse the \emph{same} $y^{+}$ as the regression target"
                   "\nat higher-noise states  (no extra reward eval, no extra VAE decode)"
        if False else
        "ours: reuse the same $y^{+}$ as the regression target\n"
        "at higher-noise states  —  no extra reward eval, no extra VAE decode",
        ha="center", va="bottom", fontsize=8.6, color=C_X, fontweight="bold")

ax.annotate("", xy=(9.35, 0), xytext=(-0.35, 0),
            arrowprops=dict(arrowstyle="-|>", color=C_BASE, lw=1.4))
ax.text(9.45, 0, "clean", fontsize=7.5, va="center", color="#777777")
ax.text(-0.45, 0, "noise", fontsize=7.5, va="center", ha="right", color="#777777")
# ── 部署网格(NFE=4):设计原则的**可视化依据**(§4.597)──────────────────────
# 4 步 schedule 的 σ = {1.000, 0.858, 0.602, 0.009};它的两个**内部**态恰好是
# 10 步网格上的 k=3 与 k=6,也就是我们传播的目标;而 OPSD 的 σ=0.278 **不在**这张网格上。
SIG4 = [1.00000, 0.85769, 0.60215, 0.00893]
YD = -0.78                                   # 部署网格所在的行
X4 = [min(range(len(SIG)), key=lambda i: abs(SIG[i] - s)) for s in SIG4]   # 对齐到同一横轴
ax.plot([X4[0], X4[-1]], [YD]*2, color=C_BASE, lw=1.4, zorder=1)
for xi, s in zip(X4, SIG4):
    interior = 0 < SIG4.index(s) < len(SIG4) - 1
    ax.scatter([xi], [YD], s=190 if interior else 80,
               color=C_X if interior else C_BASE, zorder=3, edgecolors="white", linewidths=1.2)
    ax.text(xi, YD - 0.085, r"$\sigma{=}%.3f$" % s, ha="center", va="top", fontsize=6.8,
            color="#333333" if interior else "#888888")
# 两个内部态与训练网格上对应点的对齐虚线
for xi in X4[1:-1]:
    ax.plot([xi, xi], [YD + 0.06, -0.055], color=C_X, lw=0.9, ls=(0, (2, 2)), zorder=2, alpha=.8)
# 查询态落在网格之外
ax.plot([KQ, KQ], [YD + 0.06, -0.055], color=C_Q, lw=0.9, ls=(0, (2, 2)), zorder=2, alpha=.5)
ax.text(KQ, YD - 0.085, "not on this grid", ha="center", va="top", fontsize=6.8,
        color=C_Q, style="italic")
ax.text(-0.45, YD, "deployment\ngrid (NFE$=$4)", fontsize=7.5, va="center", ha="right",
        color="#555555", linespacing=1.25)
ax.text(4.3, YD + 0.14, "the two states we propagate to are exactly the interior states of the "
                        "$4$-step grid;\nthe query state OPSD uses is not one of them",
        ha="center", va="bottom", fontsize=7.8, color="#444444")

ax.set_xlim(-2.3, 10.3); ax.set_ylim(-1.05, 1.15); ax.axis("off")
fig.tight_layout()
fig.savefig("paper_figs/fig1_method.pdf", bbox_inches="tight")
fig.savefig("paper_figs/fig1_method.png", dpi=170, bbox_inches="tight")
print("  ✅ fig1_method.{pdf,png}")
