"""Plot all-cohort paired deltas after the frozen review is joined."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opdlab.io import ROOT, read_json, sha256, write_json


def main():
    campaign = ROOT / "experiments/local_expert_v11"
    result = read_json(campaign / "results.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    models = ["sparse32", "sparse320", "dense32"]
    for ax, cohort, title in zip(axes, ["train_new_seed", "validation"],
                                 ["5 fitted prompts, 2 new seeds each", "32 existing validation prompts"]):
        stats = [result["paired_summary"][cohort][m]["complete_scene_delta"] for m in models]
        means = [s["mean"] * 100 for s in stats]
        errors = [[(s["mean"] - s["ci95"][0]) * 100 for s in stats], [(s["ci95"][1] - s["mean"]) * 100 for s in stats]]
        ax.bar(models, means, yerr=errors, capsize=5, color=["#4b91c4", "#236499", "#dc9348"])
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Complete-scene change vs joint128 (percentage points)")
    fig.suptitle("Transfer of a selected-case correction: ordinary flow10 generation", fontsize=12)
    fig.text(.5, .02, "Model-blinded implementing-assistant labels; unknown is not success.\n"
             "95% paired prompt-bootstrap intervals; development evidence, no model selected here.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.09, right=.98, bottom=.19, top=.85, wspace=.18)
    fig.savefig(campaign / "transfer.png", dpi=160)
    plt.close(fig)
    write_json(campaign / "figure_sources.json", {"script_sha256": sha256(__file__), "results_sha256": sha256(campaign / "results.json"),
               "figure_sha256": sha256(campaign / "transfer.png")})


if __name__ == "__main__":
    main()
