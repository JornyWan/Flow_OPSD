"""Export development outcomes and controls; no extrapolation to trained gains."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from opdlab.io import ROOT, read_json, sha256, write_json


def main():
    campaign = ROOT / "experiments/local_expert_v14"
    r = read_json(campaign / "results.json")
    d = read_json(campaign / "array_diagnostic.json")
    names = ["v1", "v1_discrete_event", "v2", "phase13_posthoc_dominance", "primary", "forward", "reverse", "pair_consensus", "primary"]
    labels = ["v1", "+ discrete", "evidence v2", "v13 posthoc", "frozen pair", "A→B*", "B→A*", "pair agree*", "+ literal*"]
    sources = [r] * 5 + [d] * 4
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    bottoms = np.zeros(len(names))
    for cls, label, color in (("safe_repair", "Safe repair", "#238b45"), ("unsafe", "Known harm", "#cb181d"),
                              ("non_repair", "No encoded repair", "#9e9ac8"), ("unknown", "Unresolved", "#bdbdbd")):
        values = [s["policies"][n]["accepted_classes"].get(cls, 0) for s, n in zip(sources, names)]
        axes[0].bar(np.arange(len(names)), values, bottom=bottoms, label=label, color=color)
        bottoms += values
    axes[0].set_xticks(np.arange(len(names)), labels, rotation=25, ha="right")
    axes[0].set_ylabel("Selected states / 64 (frozen review)")
    axes[0].set_title("Accepted actions: repair vs harm / no repair / unknown")
    axes[0].legend(fontsize=8)
    o, s = d["order_control"], d["self_control"]
    num = [o["shared_determinate_fields"], o["shared_determinate_agreements"], s["determinate_fields"], s["repair_fields"] + s["damage_fields"]]
    den = [o["fields"], o["shared_determinate_fields"], s["fields"], s["fields"]]
    values = [n / d if d else 0 for n, d in zip(num, den)]
    axes[1].barh(range(4), values, color=["#6baed6", "#2171b5", "#6baed6", "#cb181d"])
    axes[1].set_yticks(range(4), ["Order: known coverage", "Order: agreement if known", "Self: known coverage", "Self: false changes"])
    axes[1].set_xlim(0, 1.2)
    axes[1].set_xlabel("Fraction of indicated field denominator")
    for i, (v, n, d) in enumerate(zip(values, num, den)):
        axes[1].text(v + .015, i, f"{n}/{d}", va="center", fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_title("Array-compatible diagnostic* controls")
    fig.suptitle("Phase14 • *posthoc array-container parsing • exposed development pool • no training", fontsize=11)
    fig.tight_layout()
    output = campaign / "paired_audit.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    write_json(campaign / "figure_sources.json", {"results_sha256": sha256(campaign / "results.json"),
               "array_diagnostic_sha256": sha256(campaign / "array_diagnostic.json"),
               "script_sha256": sha256(__file__), "figure_sha256": sha256(output)})


if __name__ == "__main__":
    main()
