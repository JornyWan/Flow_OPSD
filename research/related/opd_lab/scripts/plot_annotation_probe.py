"""Plot phase15 learning, full-rollout transfer and collateral effects."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opdlab.annotation_probe import ARMS
from opdlab.io import ROOT, read_json, sha256, write_json
from review_prospective import lines


def main():
    campaign, run = ROOT / "experiments/local_expert_v15", ROOT / "runs/le15_annotation_001"
    report, metrics = read_json(campaign / "results.json"), lines(run / "metrics.jsonl")
    models = ("joint128", *ARMS)
    names = ("Initial", "Label-selected", "Random matched", "All teacher")
    colors = ("#8b929a", "#218774", "#d98b28", "#8d65b8")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained")
    ax = axes[0, 0]
    for arm, name, color in zip(ARMS, names[1:], colors[1:]):
        medians = []
        for step in (0, 32, 128):
            vals = [r["label_target_alignment"]["relative_target_mse"] for r in metrics
                    if r["arm"] == arm and r["step"] == step and "label_target_alignment" in r]
            medians.append(np.median(vals))
        ax.plot((0, 32, 128), medians, "o-", label=name, color=color)
    ax.axhline(.5, color="black", linestyle=":", linewidth=1)
    ax.set(title="A  Alignment to the 6 labelled safe actions", xlabel="Optimizer updates", ylabel="Median relative target MSE")
    ax.set_xticks((0, 32, 128))
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    cohorts = ("positive_fresh_seed", "preservation_fresh_seed", "unfitted_prompt_fresh_seed", "fitting_seed_diagnostic")
    sizes = (12, 12, 8, 6)
    x = np.arange(4)
    for i, (model, name, color) in enumerate(zip(models, names, colors)):
        vals = [report["scores"][c][model]["complete_scene"]["sum"] / n for c, n in zip(cohorts, sizes)]
        bars = ax.bar(x + (i - 1.5) * .19, vals, .19, label=name, color=color)
        for bar, val, n in zip(bars, vals, sizes):
            ax.text(bar.get_x() + bar.get_width()/2, val + .015, f"{round(val*n)}", ha="center", fontsize=8)
    ax.set_xticks(x, ("Positive\n12 pairs / 6 prompts", "Preservation\n12 pairs / 6 prompts", "Unfitted\n8 pairs / 8 prompts", "Fitting seed\n6 pairs / 6 prompts"), fontsize=9)
    ax.set(title="B  Complete full generations (count above bars)", ylabel="Complete fraction", ylim=(0, 1.12))
    ax.legend(fontsize=8, ncols=2)
    ax = axes[1, 0]
    x = np.arange(3)
    for j, (metric, name, color) in enumerate((("safe_repair", "Known safe repairs", "#218774"),
                                            ("known_harm", "Known harms", "#c65454"),
                                            ("comparison_unknown", "Unknown comparisons", "#8b929a"))):
        vals = [report["scores"]["all_fresh"][a][metric]["sum"] for a in ARMS]
        bars = ax.bar(x + (j - 1) * .25, vals, .25, label=name, color=color)
        ax.bar_label(bars, fmt="%g", fontsize=9, padding=2)
    ax.set_xticks(x, names[1:])
    ax.set(title="C  All 32 fresh pairs vs the initial model", ylabel="Pair count (harm and unknown may overlap)")
    ax.legend(fontsize=9)
    ax.margins(y=.2)
    ax = axes[1, 1]
    rows = []
    for cohort, label in zip(cohorts[:3], ("Positive fresh (6 prompts)", "Preservation fresh (6 prompts)", "Unfitted fresh (8 prompts)")):
        for other, name in (("joint128", "initial"), ("random_matched", "random")):
            stat = report["label_vs_other_complete"][cohort][other]
            rows.append((f"{label}: vs {name}", stat))
    for i, (label, stat) in enumerate(rows):
        lo, hi = stat["ci95"]
        ax.errorbar(stat["mean"], i, xerr=[[stat["mean"] - lo], [hi - stat["mean"]]], fmt="o", color="#218774", capsize=3)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
    ax.axvline(0, color="black", linewidth=1)
    ax.invert_yaxis()
    ax.set(title="D  Label-selected complete-generation differences", xlabel="Paired difference; 95% prompt-bootstrap intervals")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Phase15: selected-cohort label-assisted mechanism pilot\nOriginal text + object + full-visibility task; flow10 only; provisional assistant review", fontsize=14)
    fig.savefig(campaign / "annotation_probe.png", dpi=170)
    fig.savefig(campaign / "annotation_probe.pdf")
    write_json(campaign / "figure_sources.json", {"results_sha256": sha256(campaign / "results.json"),
               "metrics_sha256": sha256(run / "metrics.jsonl"), "script_sha256": sha256(__file__),
               "figures": {p.name: sha256(p) for p in (campaign / "annotation_probe.png", campaign / "annotation_probe.pdf")}})


if __name__ == "__main__":
    main()
