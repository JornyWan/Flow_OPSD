"""Export paired evidence and a post-hoc scalar-rescaling diagnostic from frozen tensors."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from opdlab.io import ROOT, read_json, sha256, write_json


def main():
    campaign, run = ROOT / "experiments/local_expert_v19", ROOT / "runs/le19_opportunity_001"
    report = read_json(campaign / "results.json")
    geometry = []
    for case in report["cases"]:
        row = case["query"]
        q = torch.load(run / row["saved_query"], map_location="cpu", weights_only=False)
        actions = {k: a.double() for k, a in q["actions"].items()}
        displacement = actions["student_single"] - actions["noop"]
        for mode in ("ocr@1", "geneval@1"):
            target = actions[mode] - actions["noop"]
            denom, scale = target.square().mean(), displacement.square().mean()
            dot = (target * displacement).mean()
            beta = max(0., float(dot / scale)) if scale > 0 else 0.
            relative = float((beta * displacement - target).square().mean() / denom)
            cosine = float(dot / (denom * scale).sqrt()) if scale > 0 else None
            geometry.append({"case": case["case"], "id": row["id"], "seed": row["seed"], "teacher": mode,
                "relative_mse_beta1": float((displacement - target).square().mean() / denom),
                "optimal_nonnegative_scalar": beta, "minimum_relative_mse_by_scalar": relative,
                "residual_cosine": cosine, "teacher_safe_complete": case["comparisons"][mode]["safe_repair"]
                and all(v is True for v in case["observations"][mode]["clauses"].values())
                and case["observations"][mode]["full_visibility"] is True})
    write_json(campaign / "scalar_geometry.json", {"analysis": "Post-hoc exact FP64 tensor geometry for all24 teacher comparisons.",
        "results_sha256": sha256(campaign / "results.json"), "rows": geometry,
        "formula": "min over beta>=0 of ||beta*(student-noop)-(teacher-noop)||^2 / ||teacher-noop||^2",
        "limit": "Uses teacher actions and per-state oracle scalars. It bounds action MSE only; no endpoint was rendered with rescaling and no quality claim follows."})
    modes = ["noop", "student_single", "ocr@1", "geneval@1"]
    names = ["Initial", "Learned action", "OCR action", "GenEval action"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, metric, title in zip(axes, ("complete_scene", "known_harm"), ("Complete scenes (12 states)", "Known harms versus initial (12 states)")):
        values = [report["scores"][m][metric]["sum"] for m in modes]
        bars = ax.bar(names, values, color=["#89939d", "#277da1", "#f9c74f", "#43aa8b"])
        ax.bar_label(bars)
        ax.set(title=title, ylim=(0, 12.5))
        ax.tick_params(axis="x", labelrotation=15)
    axes[0].axhline(report["signals"]["safe_complete_or_noop"]["sum"], color="#9d4edd", linestyle="--", label="Safe teacher oracle + keep initial: 7")
    axes[0].legend(fontsize=8)
    fig.suptitle("Teacher opportunity on the same Phase18 states\nExposed development data; provisional assistant labels; no new training", fontsize=11)
    for ext in ("png", "pdf"):
        fig.savefig(campaign / f"teacher_opportunity.{ext}", dpi=170)
    plt.close(fig)
    selected = [c for c in report["cases"] if c["signals"]["teacher_safe_complete_local_determinate_incomplete"]]
    fig, axes = plt.subplots(len(selected), 4, figsize=(12, 3.4 * len(selected)), squeeze=False, constrained_layout=True)
    joined = {(e["case"], e["row"]["mode"]): e for e in report["joined_review"]}
    for i, case in enumerate(selected):
        for j, mode in enumerate(modes):
            entry = joined[case["case"], mode]
            axes[i, j].imshow(plt.imread(run / entry["row"]["image"]))
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            axes[i, j].set_title(names[j], fontsize=10)
            if j == 0:
                axes[i, j].set_ylabel(f"Case {case['case']}: {case['query']['id'].removeprefix('jp1-train-')}\nseed {case['query']['seed']}", fontsize=9)
    fig.suptitle("All 2 determinate cases where a teacher safely completes and the learned action does not\nSame initial prefix and suffix; selected by the predeclared diagnostic flag", fontsize=11)
    for ext in ("png", "pdf"):
        fig.savefig(campaign / f"teacher_student_gap.{ext}", dpi=170)
    plt.close(fig)
    print([{k: r[k] for k in ("case", "teacher", "relative_mse_beta1", "minimum_relative_mse_by_scalar", "optimal_nonnegative_scalar", "residual_cosine")}
           for r in geometry if r["teacher_safe_complete"]])


if __name__ == "__main__":
    main()
