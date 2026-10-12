"""Verify and summarize the complete predeclared frozen-query probe matrix."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from opdlab.evaluate import compare, read_evaluation
from opdlab.fitting_probe import ARMS, tensor_digest
from opdlab.io import read_json, sha256, write_json
from opdlab.report import prompt_summary
from opdlab.runtime import verify_run


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def mean(values):
    return float(np.mean(values)) if values else None


def summarize(run, evaluation):
    verify_run(run)
    result = read_json(run / "result.json")
    if result["status"] != "complete" or read_json(evaluation / "result.json")["status"] != "complete":
        raise ValueError("Wait for all four branches and both evaluation cells")
    for name in ("queries", "postfit", "metrics"):
        if sha256(run / f"{name}.jsonl") != result[f"{name}_sha256"]:
            raise ValueError(f"{name} changed after completion")
    plan = read_json(run / "probe_plan.json")
    if sha256(run / "probe_plan.json") != result["plan_sha256"]:
        raise ValueError("Plan changed")
    queries, metrics, post = (lines(run / f"{name}.jsonl") for name in ("queries", "metrics", "postfit"))
    qmap = {r["id"]: r for r in queries}
    pmap = {(r["arm"], r["id"]): r for r in post}
    if len(qmap) != len(queries) or len(queries) != 64 or len(pmap) != len(post) or len(post) != 256:
        raise ValueError("Query/endpoint rows are missing or duplicated")
    for q, spec in zip(queries, plan["queries"]):
        if any(q[k] != spec[k] for k in ("id", "seed", "query_index")):
            raise ValueError("Query differs from frozen plan")
        if [c["name"] for c in q["candidates"]] != ["noop", "ocr@0.5", "ocr@1", "geneval@0.5", "geneval@1"]:
            raise ValueError("Candidate set differs")
    image_count = 0
    for q in queries:
        for c in q["candidates"]:
            if not c["valid"] or sha256(run / c["image"]) != c["image_sha256"]:
                raise ValueError("Invalid or changed candidate image")
            image_count += 1
        selected = next(c for c in q["candidates"] if c["name"] == q["selected"])
        if q["accepted"] and (not selected["admissible"] or selected["utility"] <= q["candidates"][0]["utility"]):
            raise ValueError("Accepted query violates protected selector")
    for row in post:
        if row["arm"] not in ARMS or row["id"] not in qmap:
            raise ValueError("Unknown endpoint row")
        q = qmap[row["id"]]
        if any(q[k] != row[k] for k in ("seed", "query_index", "accepted")):
            raise ValueError("Postfit query identity changed")
        for endpoint in ("single", "full"):
            c = row[endpoint]
            if not c["valid"] or sha256(run / c["image"]) != c["image_sha256"]:
                raise ValueError("Invalid or changed endpoint image")
            image_count += 1
    expected_metrics = [("initial", 0), ("shared", 1), ("keep0_restored", 1),
                        ("keep0", 2), ("keep0", 3), ("keep0", 4), ("keep1_restored", 1),
                        ("keep1", 2), ("keep1", 3), ("keep1", 4)]
    if [(m["branch"], m["optimizer_step"]) for m in metrics] != expected_metrics:
        raise ValueError("Actual fitting trajectory differs from the design")
    if len({m["student_sha256"] for m in metrics if m["optimizer_step"] == 1}) != 1:
        raise ValueError("Branches did not restore a common student prefix")
    behavior = torch.load(run / "frozen_behavior.pt", weights_only=True)
    if {m["behavior_sha256"] for m in metrics} != {tensor_digest(behavior)} or result["behavior_sha256"] != tensor_digest(behavior):
        raise ValueError("Behavior snapshot mismatch")
    trajectory = []
    # Candidate distances are measured directly against the frozen noop, so
    # this remains valid even if initial student/behavior inference differs numerically.
    target_noop_norms = {q["id"]: .5 * q["dt"] ** 2 * next(c["distance"] for c in q["candidates"]
                         if c["name"] == q["selected"]) for q in queries if q["accepted"]}
    geometry = []
    for m in metrics:
        if len(m["rows"]) != 64 or {r["id"] for r in m["rows"]} != set(qmap):
            raise ValueError("Fitting audit coverage mismatch")
        for row in m["rows"]:
            if row["accepted"] != qmap[row["id"]]["accepted"]:
                raise ValueError("Acceptance changed during fitting")
        trajectory.append({"branch": m["branch"], "step": m["optimizer_step"],
                           "accepted_target_loss": mean([r["target_loss"] for r in m["rows"] if r["accepted"]]),
                           "rejected_noop_drift": mean([r["noop_drift"] for r in m["rows"] if not r["accepted"]]),
                           "fitting": m["fitting"]})
        # Post-hoc descriptive geometry from three already logged squared norms.
        # t=||target-noop||^2, d=||prediction-noop||^2, e=||prediction-target||^2
        # (all contain the SAME .5*dt^2 factor within a query).
        g = []
        for row in m["rows"]:
            if row["accepted"] and target_noop_norms[row["id"]] > 1e-12:
                t, d, e = target_noop_norms[row["id"]], row["noop_drift"], row["target_loss"]
                projection = (d + t - e) / (2 * t)
                cosine = (d + t - e) / (2 * np.sqrt(d * t)) if d > 1e-15 else None
                g.append({"id": row["id"], "target_norm": t, "projection_fraction": projection,
                          "cosine": float(cosine) if cosine is not None else None})
        geometry.append({"branch": m["branch"], "step": m["optimizer_step"], "queries": len(g),
                         "mean_projection_fraction": mean([r["projection_fraction"] for r in g]),
                         "target_loss_weighted_projection_fraction": sum(r["target_norm"] * r["projection_fraction"] for r in g) / sum(r["target_norm"] for r in g) if g else None,
                         "mean_cosine": mean([r["cosine"] for r in g if r["cosine"] is not None]), "rows": g})
    from safetensors.torch import load_file
    for arm, (step, keep) in ARMS.items():
        target = run / "checkpoints" / arm
        meta = read_json(target / "metadata.json")
        state = torch.load(target / "training.pt", weights_only=False, map_location="cpu")
        raw = load_file(str(target / "raw_lora/adapter_model.safetensors"))
        if (not meta["complete"] or meta["optimizer_step"] != step or state["optimizer_step"] != step or
                meta["rejected_noop_coefficient"] != keep or sha256(target / "training.pt") != meta["training_state_sha256"] or
                sha256(run / "shared_step1.pt") != meta["shared_step1_sha256"] or
                tensor_digest(raw) != meta["student_sha256"] or tensor_digest(state["student"]) != meta["student_sha256"]):
            raise ValueError("Raw export, optimizer budget or full-state provenance mismatch")
        if any(float(s["step"]) != step for s in state["optimizer"]["state"].values()):
            raise ValueError("Adam step disagrees with checkpoint label")
        del state, raw
    target_gain = {}
    for group in ("all", "accepted", "rejected"):
        ids = [i for i, q in qmap.items() if group == "all" or q["accepted"] == (group == "accepted")]
        target_gain[group] = prompt_summary([(i, next(c["utility"] for c in qmap[i]["candidates"]
                                         if c["name"] == qmap[i]["selected"]) - qmap[i]["candidates"][0]["utility"]) for i in ids])
    endpoints = {}
    for arm in ARMS:
        endpoints[arm] = {}
        for group in ("all", "accepted", "rejected"):
            ids = [i for i, q in qmap.items() if group == "all" or q["accepted"] == (group == "accepted")]
            entry = {}
            for endpoint in ("single", "full"):
                gains = [(i, pmap[arm, i][endpoint]["utility"] - qmap[i]["candidates"][0]["utility"]) for i in ids]
                existing = [i for i in ids if any(qmap[i]["candidates"][0]["satisfied"])]
                damaged = [i for i in existing if any(a and not b for a, b in zip(
                           qmap[i]["candidates"][0]["satisfied"], pmap[arm, i][endpoint]["satisfied"]))]
                entry[endpoint] = {"utility_gain": prompt_summary(gains),
                                   "positive": sum(v > 0 for _, v in gains), "negative": sum(v < 0 for _, v in gains),
                                   "queries_with_existing_clause": len(existing), "queries_losing_existing_clause": len(damaged)}
            endpoints[arm][group] = entry
    fit_rows = {(m["branch"], m["optimizer_step"]): {r["id"]: r for r in m["rows"]} for m in metrics}
    discordance = {"scope": "Post-hoc descriptive paired audit; not an additional hypothesis test.", "accepted": {}}
    for keep in (0, 1):
        a, b = [f"fit{step}" + ("_keep" if keep else "") for step in (2, 4)]
        improved = [i for i, q in qmap.items() if q["accepted"] and
                    fit_rows[f"keep{keep}", 4][i]["target_loss"] < fit_rows[f"keep{keep}", 2][i]["target_loss"]]
        worse_value = [i for i in improved if pmap[b, i]["single"]["utility"] < pmap[a, i]["single"]["utility"]]
        discordance["accepted"][f"keep{keep}"] = {"loss_improved_ids_2_to_4": improved,
                                                  "local_utility_dropped_despite_loss_improvement_ids": worse_value}
    rejected = [i for i, q in qmap.items() if not q["accepted"]]
    discordance["rejected_drift_reduced_by_keep_at_4_ids"] = [i for i in rejected if
                   fit_rows["keep1", 4][i]["noop_drift"] < fit_rows["keep0", 4][i]["noop_drift"]]
    contrasts = [(name, "base") for name in ARMS] + [("fit2_keep", "fit2"), ("fit4_keep", "fit4"),
                 ("fit4", "fit2"), ("fit4_keep", "fit2_keep")]
    validation = {}
    for sampler, steps in plan["evaluation"]["cells"]:
        cell = evaluation / f"{sampler}{steps}"
        for arm in ("base", *ARMS):
            protocol, rows = read_evaluation(cell / arm)
            if (len(rows) != 32 or protocol["split"] != "validation" or protocol["replicate_seeds"] != [42]
                    or protocol["sampler"] != sampler or protocol["steps"] != steps):
                raise ValueError("Validation protocol differs from predeclared probe")
            image_count += len(rows)
        validation[f"{sampler}{steps}"] = {f"{a}_minus_{b}": compare(cell / b, cell / a) for a, b in contrasts}
    if image_count != 1152 or result["counts"]["backward_queries"] != 448:
        raise ValueError("Image or backward-operation budget mismatch")
    return {"scope": "Fixed-query mechanism diagnostic, one training seed; not on-policy efficacy or sampler robustness proof.",
            "training_run": str(run), "evaluation_run": str(evaluation),
            "integrity": {"verified_images": image_count, "shared_prefix": True, "behavior_frozen": True,
                          "raw_exports_match_full_states": True, "unique_optimizer_updates": 7,
                          "backward_queries": 448, "test_used": False},
            "queries": 64, "accepted": sum(q["accepted"] for q in queries),
            "selected_actions": dict(Counter(q["selected"] for q in queries)),
            "target_gain": target_gain, "fit_trajectory": trajectory, "train_endpoints": endpoints,
            "fit_geometry_posthoc": {"interpretation": "Descriptive decomposition added after observing partial fit logs; no new optimization or hypothesis test. Projection fraction is along the selected target-minus-noop direction; aggregate weighting uses .5*dt^2 target norm.", "trajectory": geometry},
            "fit_endpoint_discordance_posthoc": discordance,
            "validation": validation, "counts": result["counts"], "branch_counts": result["branch_counts"],
            "training_worker_gpu_hours": result["allocated_gpu_hours"]}


def render(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    trajectory = summary["fit_trajectory"]
    for keep, label in ((0, "No preservation"), (1, "Preservation = 1")):
        rows = [r for r in trajectory if r["branch"] in ("initial", "shared", f"keep{keep}")]
        for ax, key in zip(axes, ("accepted_target_loss", "rejected_noop_drift")):
            ax.plot([r["step"] for r in rows], [r[key] for r in rows], marker="o", label=label)
            ax.set_xlabel("Full-batch fitting updates"); ax.set_xticks(range(5)); ax.grid(alpha=.2)
    axes[0].set_ylabel("Accepted: mean transition loss"); axes[1].set_ylabel("Rejected: mean noop drift")
    axes[0].legend(); fig.savefig(out / "fitting.png", dpi=180); fig.savefig(out / "fitting.pdf"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained", sharey=True)
    for ax, (cell, values) in zip(axes, summary["validation"].items()):
        for i, arm in enumerate(ARMS):
            row = values[f"{arm}_minus_base"]["utility"]
            value, (low, high) = row["paired_delta"], row["ci95"]
            ax.errorbar(value, i, xerr=[[value-low], [high-value]], fmt="o", capsize=3)
        ax.axvline(0, color="black", linewidth=.8); ax.set_title(cell)
        ax.set_xlabel("Utility delta vs base (95% prompt bootstrap)")
        ax.set_yticks(range(4), list(ARMS)); ax.grid(axis="x", alpha=.2)
    fig.savefig(out / "validation.png", dpi=180); fig.savefig(out / "validation.pdf"); plt.close(fig)
    rows = ["# Phase 3 frozen-query fitting probe", "", summary["scope"], "",
            f"Accepted {summary['accepted']}/64 queries. Verified all {summary['integrity']['verified_images']} saved images, shared first update, frozen behavior and four raw checkpoint exports.", "",
            "## Training-query mechanism measurements", "",
            "| Arm | Accepted loss / initial | Rejected noop drift | Accepted single-step ΔU | Rejected full-policy ΔU | All full-policy ΔU |", "|---|---:|---:|---:|---:|---:|"]
    initial = trajectory[0]["accepted_target_loss"]
    for arm, (step, keep) in ARMS.items():
        t = next(r for r in trajectory if r["branch"] == f"keep{keep}" and r["step"] == step)
        e = summary["train_endpoints"][arm]
        rows.append(f"| {arm} | {t['accepted_target_loss']/initial:.6f} | {t['rejected_noop_drift']:.8g} | {e['accepted']['single']['utility_gain']['mean']:+.6f} | {e['rejected']['full']['utility_gain']['mean']:+.6f} | {e['all']['full']['utility_gain']['mean']:+.6f} |")
    rows += ["", "![Fitting and drift](fitting.png)", "", "## Ordinary validation generation", "",
             "32 prompts, seed 42, no expert/search at inference. Intervals reflect prompt variation, not training-seed variation; no multiple-comparison correction.", "",
             "| Cell | Contrast | Δ utility [95% CI] | Δ joint success [95% CI] |", "|---|---|---:|---:|"]
    for cell, values in summary["validation"].items():
        for contrast, value in values.items():
            def number(metric):
                r = value[metric]
                return f"{r['paired_delta']:+.6f} [{r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}]"
            rows.append(f"| {cell} | {contrast} | {number('utility')} | {number('joint_success')} |")
    rows += ["", "![Validation deltas](validation.png)", "", "All fixed-query train endpoint statistics, component validation results and operation counts are in `summary.json`. The separate training-image coverage audit does not replace the unchanged v1 metrics. Test remains unused."]
    rows += ["", "## Post-hoc fitting geometry", "",
             "Derived from the three logged squared norms, without new GPU work. For accepted queries, projection fraction = (noop drift + target-to-noop squared norm − current target loss) / (2 × target-to-noop squared norm), with the same .5*dt² factor in every norm. This measures how much of the selected correction was realized along its own direction, and is descriptive post-hoc analysis.", "",
             "| Arm | Mean projection fraction | Target-loss-weighted fraction | Mean cosine with selected correction |", "|---|---:|---:|---:|"]
    for arm, (step, keep) in ARMS.items():
        g = next(r for r in summary["fit_geometry_posthoc"]["trajectory"] if r["branch"] == f"keep{keep}" and r["step"] == step)
        rows.append(f"| {arm} | {g['mean_projection_fraction']:.6f} | {g['target_loss_weighted_projection_fraction']:.6f} | {g['mean_cosine']:.6f} |")
    (out / "report.md").write_text("\n".join(rows) + "\n")


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--evaluation", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run, evaluation, out = (Path(v).resolve() for v in (args.run, args.evaluation, args.out))
    out.mkdir(parents=True, exist_ok=False)
    summary = summarize(run, evaluation)
    write_json(out / "summary.json", summary)
    render(summary, out)
    print({"report": str(out), "accepted": summary["accepted"], "integrity": summary["integrity"]})


if __name__ == "__main__":
    main()
