"""Join completed phase7 measurements with already frozen visual observations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from opdlab.io import ROOT, read_json, sha256, write_json


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def main():
    campaign = ROOT / "experiments/local_expert_v7"
    attempt = read_json(campaign / "attempt_001.json")
    fit, capability = Path(attempt["fit_run"]), Path(attempt["capability_run"])
    review = ROOT / "runs/le7_review_001"
    frozen = read_json(review / "observations_frozen.json")
    for name in ("observations", "key"):
        if sha256(review / f"{name}.json") != frozen[f"{name}_sha256"]:
            raise ValueError("Review or key changed after freeze")
    for run in (fit, capability):
        result = read_json(run / "result.json")
        if result["status"] != "complete":
            raise ValueError("Incomplete worker")
        for name, expected in result["output_hashes"].items():
            if sha256(run / name) != expected:
                raise ValueError("Completed output changed")
        for filename in (("references.jsonl", "endpoints.jsonl") if run == fit else ("capability.jsonl",)):
            for row in rows(run / filename):
                if sha256(run / row["image"]) != row["image_sha256"]:
                    raise ValueError("Image evidence changed")
    metrics, endpoints = rows(fit / "metrics.jsonl"), rows(fit / "endpoints.jsonl")
    if len(metrics) != 60 or len(endpoints) != 100 or len(rows(fit / "updates.jsonl")) != 768:
        raise ValueError("Incomplete fitting matrix")
    if read_json(fit / "result.json")["counts"]["backward_queries"] != 1280:
        raise ValueError("Unexpected fitting compute")
    specs = {s["id"]: s for s in read_json(fit / "phase7_plan.json")["selected"]}
    labels = {r["case"]: r for r in read_json(review / "observations.json")}
    if set(labels) != set(range(1, 21)):
        raise ValueError("Missing/duplicate visual review coverage")
    joined = []
    for item in read_json(review / "key.json"):
        row, obs = item["row"], labels[item["case"]]
        spec = specs[row["id"]]
        expected = set(spec["old_observations"]["target"]["clauses"])
        if set(obs["clauses"]) != expected or any(v not in (True, False, None) for v in obs["clauses"].values()):
            raise ValueError("Review clause contract differs")
        needed = set(spec["repaired"] + spec["preserved"])
        retained = all(obs["clauses"][c] is True for c in needed)
        if spec["old_observations"]["target"]["full_visibility"]:
            retained = retained and obs["full_visibility"] is True
        joined.append({"case": item["case"], "id": row["id"], "arm": row["arm"],
                       "fit_kind": "joint" if row["arm"] == "joint" else "individual",
                       "mode": row["mode"], "repair_retained": retained,
                       "complete_visibility_subset": spec["complete_visibility_subset"],
                       "observation": obs, "relative_target_mse": row["relative_target_mse"],
                       "proxy_utility": row["utility"]})
    summaries = {}
    for kind in ("individual", "joint"):
        final = [r for r in metrics if r["optimizer_step"] == 128 and (r["arm"] == "joint") == (kind == "joint")]
        summary = {"queries": len(final), "mse_threshold_pass_count": sum(r["relative_target_mse"] <= .05 for r in final),
                   "mean_relative_target_mse": float(np.mean([r["relative_target_mse"] for r in final])),
                   "mean_gamma": float(np.mean([r["gamma"] for r in final])), "per_query": final}
        for mode in ("single", "full"):
            selected = [r for r in joined if r["fit_kind"] == kind and r["mode"] == mode]
            subset = [r for r in selected if r["complete_visibility_subset"]]
            summary[mode] = {"retained": sum(r["repair_retained"] for r in selected), "total": len(selected),
                             "complete_visibility_subset_retained": sum(r["repair_retained"] for r in subset),
                             "complete_visibility_subset_total": len(subset)}
        summary["local_engineering_gate_passed"] = (summary["mse_threshold_pass_count"] >= 4 and
                                                    summary["single"]["retained"] >= 4)
        local = {r["id"]: r for r in joined if r["fit_kind"] == kind and r["mode"] == "single"}
        summary["same_query_mse_and_local_repair_pass_count"] = sum(
            r["relative_target_mse"] <= .05 and local[r["id"]]["repair_retained"] for r in final)
        summaries[kind] = summary
    cap = rows(capability / "capability.jsonl")
    conditions = sorted({r["condition"] for r in cap})
    if len(cap) != 320 or len({(r["condition"], r["id"]) for r in cap}) != 320:
        raise ValueError("Incomplete capability matrix")
    base = {r["id"]: r for r in cap if r["condition"] == "base_cfg4.5"}
    capability_summary = {}
    for condition in conditions:
        group = sorted([r for r in cap if r["condition"] == condition], key=lambda r: r["id"])
        differences = np.array([r["utility"] - base[r["id"]]["utility"] for r in group])
        bootstrap = np.random.default_rng(42).choice(differences, (10000, 64), replace=True).mean(1)
        capability_summary[condition] = {
            "n": len(group), "proxy_utility": float(np.mean([r["utility"] for r in group])),
            "text_exact_proxy": float(np.mean([r["satisfied"][0] for r in group])),
            "nontext_clause_success_proxy": float(np.mean([np.mean(r["satisfied"][1:]) for r in group])),
            "joint_success_proxy": float(np.mean([r["joint_success"] for r in group])),
            "paired_delta_utility_vs_base_cfg4.5": float(differences.mean()),
            "paired_prompt_bootstrap_ci95": np.quantile(bootstrap, [.025, .975]).tolist()}
    native = read_json(campaign / "native_history.json")
    native_eval = [{"step": r["_step"], "ocr_reward": r["eval_monitor_proxy_ocr"]}
                   for r in native["rows"] if "eval_monitor_proxy_ocr" in r]
    native_summary = {"scope": "native SOPD-OCR subset; EMA eval, unpaired noise, n=16; last eval24 not32",
                      "optimizer_steps": len(rows(Path(attempt["native_run"]) / "artifacts/progress.jsonl")),
                      "evaluation": native_eval,
                      "first_online_loss": native["rows"][0]["loss"],
                      "last_online_loss": native["rows"][-1]["loss"],
                      "online_loss_queries_change_each_update": True,
                      "wandb_source_sha256": sha256(native["source"])}
    copy = rows(capability / "teacher_copy.jsonl")
    replay = rows(fit / "references.jsonl")
    result = {"scope": "selected-positive learnability, not generalization or policy efficacy",
              "teacher_copy": {"total": len(copy), "exact": sum(r["exact"] for r in copy)},
              "reference_replay": {"total": len(replay), "exact": sum(r["old_image_exact_match"] for r in replay)},
              "native": native_summary, "capability_proxy": capability_summary, "fitting": summaries,
              "visual_review": joined, "observations_sha256": frozen["observations_sha256"]}
    write_json(campaign / "results.json", result)
    write_json(campaign / "calibration_observations.json", read_json(review / "observations.json"))
    write_json(campaign / "calibration_observations_frozen.json", frozen)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for kind, style in (("individual", "--"), ("joint", "-")):
        for ident, spec in specs.items():
            group = sorted([r for r in metrics if r["id"] == ident and
                            (r["arm"] == "joint") == (kind == "joint")], key=lambda r: r["optimizer_step"])
            x = [r["optimizer_step"] for r in group]
            axes[0].plot(x, [r["relative_target_mse"] for r in group], style, label=f"{kind} case{spec['case']}")
            axes[1].plot(x, [r["gamma"] for r in group], style)
    axes[0].set_yscale("log")
    axes[0].axhline(.05, color="black", lw=.7, label="5% engineering threshold")
    axes[0].set_title("Fixed target MSE / initial MSE")
    axes[1].set_title("Target-direction projection gamma")
    axes[2].plot([r["step"] for r in native_eval], [r["ocr_reward"] for r in native_eval], "o-")
    axes[2].set_title("Native SOPD-OCR EMA proxy (n=16)")
    axes[2].text(.03, .05, "Unpaired noise; subset diagnostic", transform=axes[2].transAxes, fontsize=9)
    for ax in axes:
        ax.set_xlabel("Optimizer updates")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(campaign / "learnability.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"native": native_summary, "fitting": {k: {a: b for a, b in v.items() if a != "per_query"}
                                                            for k, v in summaries.items()},
                      "capability": capability_summary}, indent=2))


if __name__ == "__main__":
    main()
