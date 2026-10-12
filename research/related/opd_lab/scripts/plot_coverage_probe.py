"""Standalone figures for phase17 candidate availability, not trained-policy gains."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from opdlab.io import ROOT, read_json
from review_coverage_probe import pools


def main():
    out = ROOT / "experiments/local_expert_v17"
    report = read_json(out / "results.json")
    names = list(pools())
    labels = ["i3 / a1\n2 candidates", "i1 / a1\n2", "i5 / a1\n2", "i3 / a.5,1\n4", "i1,3,5 / a1\n6", "Expanded\n12"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax = axes[0, 0]
    safe = [report["pool_scores"][n]["safe_available"]["sum"] for n in names]
    unknown = [report["pool_scores"][n]["unresolved_without_safe"]["sum"] for n in names]
    bars = ax.bar(range(6), safe, color="#2a9d8f", label="Known safe available")
    ax.bar_label(bars, padding=2)
    ax.bar(range(6), unknown, bottom=safe, color="#bbbbbb", label="Unresolved, no known safe")
    ax.set(xticks=range(6), xticklabels=labels, ylim=(0,13), title="State coverage (12 collection seeds)")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    rows = [report["candidate_scores"][s] for s in pools()["expanded"]]
    for offset, metric, color, label in ((-.25,"safe_repair","#2a9d8f","Safe repair"),(0,"known_harm","#e76f51","Known harm"),(.25,"comparison_unknown","#bbbbbb","Pair unknown")):
        vals = [r[metric]["sum"] for r in rows]
        ax.bar(np.arange(12)+offset, vals, .23, color=color, label=label)
    ax.set(xticks=range(12), xticklabels=pools()["expanded"], ylim=(0,13), title="Every candidate: repair, harm and uncertainty")
    ax.tick_params(axis="x", labelrotation=65, labelsize=8)
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    ids = sorted(report["per_prompt_safe_states"]["expanded"])
    matrix = np.array([[report["per_prompt_safe_states"][n][p] for n in names] for p in ids])
    ax.imshow(matrix, vmin=0, vmax=2, cmap="Greens", aspect="auto")
    for i in range(6):
        for j in range(6):
            ax.text(j,i,str(matrix[i,j]),ha="center",va="center",color="white" if matrix[i,j]==2 else "black")
    ax.set(xticks=range(6), xticklabels=labels, yticks=range(6), yticklabels=[p.replace("jp1-train-", "") for p in ids], title="Safe target states per prompt (0, 1 or 2)")
    ax = axes[1, 1]
    base = report["candidate_scores"]["noop"]["complete_scene"]["sum"]
    vals = [base] + [report["pool_scores"][n]["safe_selected_complete"]["sum"] for n in names]
    bars = ax.bar(range(7), vals, color=["#7d8597"]+["#457b9d"]*6)
    ax.bar_label(bars, labels=[f"{int(v)}/12" for v in vals], padding=3)
    ax.set(xticks=range(7), xticklabels=["Noop"]+labels, ylim=(0,13), title="Complete endpoints after oracle safe-target choice")
    fig.suptitle("Phase17 | Candidate coverage on new collection seeds; 0 training updates\nSix selected prompts; provisional assistant labels. More candidates cost more compute; oracle choice is not a deployed method.", fontsize=12)
    for ext in ("png","pdf"):
        fig.savefig(out / f"candidate_coverage.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
