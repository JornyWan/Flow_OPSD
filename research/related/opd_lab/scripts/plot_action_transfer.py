"""Standalone phase16 figures from the frozen, joined assistant observations."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from opdlab.action_transfer import OLD, FRESH
from opdlab.io import ROOT, read_json


def main():
    out = ROOT / "experiments/local_expert_v16"
    result = read_json(out / "results.json")
    names = {"joint128": "Initial full", "label_selected": "Learned full", "student_single": "Learned one step",
             "selected_teacher": "Selected teacher", "ocr@1": "OCR one step", "geneval@1": "GenEval one step"}
    modes = ["joint128", "label_selected", "student_single"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, cohort, extra, denominator, title in (
        (axes[0, 0], OLD, ["selected_teacher"], 6, "Fitting seeds: complete scene"),
        (axes[0, 1], FRESH, ["ocr@1", "geneval@1"], 12, "Previously exposed fresh seeds: complete scene")):
        ms = modes + extra
        vals = [result["scores"][cohort][m]["complete_scene"]["sum"] for m in ms]
        bars = ax.bar(range(len(ms)), vals, color=["#7d8597", "#0077b6", "#48a9a6", "#e9c46a", "#f4a261"][:len(ms)])
        ax.bar_label(bars, labels=[f"{int(v)}/{denominator}" for v in vals], padding=3)
        ax.set(xticks=range(len(ms)), xticklabels=[names[m] for m in ms], ylim=(0, denominator + 1), title=title)
        ax.tick_params(axis="x", labelrotation=17)
        ax.set_ylabel("Complete images (unknown is not complete)")
    ax = axes[1, 0]
    ms = ["label_selected", "student_single", "ocr@1", "geneval@1"]
    for offset, metric, label, color in ((-.25, "safe_repair", "Known safe repair", "#2a9d8f"),
                                        (0, "known_harm", "Known harm", "#e76f51"),
                                        (.25, "comparison_unknown", "Pair has unknown", "#a8a8a8")):
        vals = [result["scores"][FRESH][m][metric]["sum"] for m in ms]
        bars = ax.bar(np.arange(4) + offset, vals, .24, label=label, color=color)
        ax.bar_label(bars, padding=2)
    ax.set(xticks=np.arange(4), xticklabels=[names[m] for m in ms], ylim=(0, 13), title="Fresh seeds: changes relative to initial full")
    ax.tick_params(axis="x", labelrotation=17)
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    cases = sorted(result["cases"], key=lambda r: (r["query"]["id"], r["query"]["cohort"] != OLD, r["query"]["seed"]))
    cols = ["any_teacher_safe", "teacher_safe_local_determinate_not_safe", "local_safe_full_not_safe", "full_complete_local_incomplete"]
    matrix = np.array([[int(c["diagnostic_flags"][k]) for k in cols] for c in cases])
    ax.imshow(matrix, vmin=0, vmax=1, cmap="Greens", aspect="auto", interpolation="none")
    ax.set(xticks=range(4), xticklabels=["Any safe\nteacher", "Teacher safe,\nlocal fails*", "Local safe,\nfull not safe", "Full complete,\nlocal incomplete"],
           yticks=range(len(cases)), yticklabels=[f"Case {r['case']:02d} {'old' if r['query']['cohort'] == OLD else 'fresh'}" for r in cases],
           title="Per-state diagnostic signals (green = observed)")
    ax.tick_params(labelsize=8)
    fig.suptitle("Phase16 | Shared initial prefix and suffix; one substituted action at flow10 index3\nSix selected prompts; provisional assistant labels; 0 updates. *All local comparison labels known.", fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"action_transfer.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
