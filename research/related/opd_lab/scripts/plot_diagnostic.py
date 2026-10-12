"""Export fixed-action effects across intervention positions from an audited report."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from opdlab.io import read_json, sha256, write_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    path, out = Path(args.summary).resolve(), Path(args.out).resolve()
    summary = read_json(path)
    out.mkdir(parents=True, exist_ok=False)
    positions = sorted(summary["by_position"], key=int)
    arms = list(summary["fixed_arms_all_positions"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    colors = ["#1764ab", "#00a6a6", "#df6b00", "#8d4aaf"]
    for ax, metric, title in zip(axes, ["utility_delta", "collateral_damage_rate"],
                                  ["Terminal utility change vs. no intervention", "Fraction damaging an already-satisfied clause"]):
        for i, arm in enumerate(arms):
            values = [summary["by_position"][j]["fixed_arms"][arm][metric] for j in positions]
            means = np.array([v["mean"] for v in values])
            intervals = np.array([v["ci95"] for v in values])
            x = np.arange(len(positions)) + (i - (len(arms) - 1) / 2) * .11
            ax.errorbar(x, means, yerr=[means - intervals[:, 0], intervals[:, 1] - means],
                        fmt="o", capsize=3, markersize=4, label=arm, color=colors[i % len(colors)])
        ax.axhline(0, color="grey", linewidth=.8)
        ax.set_xticks(range(len(positions)), [f"index {j}\nsigma={summary['by_position'][j]['sigma']:.3f}" for j in positions])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=.2)
        ax.set_xlabel("Intervention location on the frozen flow-10 trajectory")
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"{summary['prompts']} development prompts, {summary['queries']} frozen queries; prompt bootstrap 95% intervals", fontsize=11)
    fig.savefig(out / "fixed_actions.png", dpi=200)
    fig.savefig(out / "fixed_actions.pdf")
    plt.close(fig)
    write_json(out / "provenance.json", {"summary": str(path), "sha256": sha256(path),
               "interpretation": "Fixed-action local effects only; no trained-policy or cross-sampler claim."})


if __name__ == "__main__":
    main()
