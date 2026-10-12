"""Audit every action tensor/image and report the predeclared phase4 contrasts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from opdlab.action_probe import ARMS, BETAS, project_action, ray_action, verify_plan
from opdlab.fitting_probe import tensor_digest
from opdlab.io import read_json, sha256, write_json
from opdlab.report import prompt_summary
from opdlab.runtime import verify_run


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]


def summarize(run):
    verify_run(run)
    plan, result = verify_plan(run), read_json(run / "result.json")
    if result["status"] != "complete" or result["optimizer_updates"] != 0:
        raise ValueError("Incomplete or non-diagnostic run")
    if result["plan_sha256"] != sha256(run / "action_plan.json"):
        raise ValueError("Completion plan mismatch")
    for name, checksum in result["artifacts_sha256"].items():
        if sha256(run / name) != checksum:
            raise ValueError(f"Changed output: {name}")
    parent, ids = Path(plan["parent_run"]), plan["query_ids"]
    names = [f"ray@{b:g}" for b in BETAS] + [f"{a}/{k}" for a in ARMS for k in ("actual", "projected")]
    rows, geometry = lines(run / "actions.jsonl"), lines(run / "geometry.jsonl")
    rowmap = {(r["id"], r["action"]): r for r in rows}
    expected = {(i, n) for i in ids for n in names}
    geom = {(g["id"], g["arm"]): g for g in geometry}
    if set(rowmap) != expected or len(rows) != len(expected) or len(ids) != 22:
        raise ValueError("Incomplete/duplicated action grid")
    if set(geom) != {(i, a) for i in ids for a in ARMS} or len(geometry) != 88:
        raise ValueError("Incomplete/duplicated geometry grid")
    queries = {q["record"]["id"]: q for q in torch.load(parent / "frozen_queries.pt", map_location="cpu", weights_only=False)}
    frozen = torch.load(parent / "frozen_behavior.pt", map_location="cpu", weights_only=True)
    if result["behavior_sha256"] != tensor_digest(frozen) or any("teacher" in k and v for k, v in result["counts"].items()):
        raise ValueError("Behavior changed or teachers queried")
    actions = torch.load(run / "actions.pt", map_location="cpu", weights_only=True)
    if set(actions) != expected or result["queries"] != len(ids) or result["images"] != len(rows):
        raise ValueError("Action tensor/completion coverage mismatch")
    old = {r["id"]: r for r in lines(parent / "queries.jsonl")}
    post = {(r["id"], r["arm"]): r for r in lines(parent / "postfit.jsonl")}
    refs, comparison_flag_errors = [], []
    # This immutable first attempt compared tuple-valued Verdict.as_dict() to
    # list-valued JSON references. Preserve its flags; recompute exact serialized
    # values separately. Permit this correction ONLY for the audited source hash.
    legacy_flag_bug = sha256(run / "framework/python/opdlab/action_probe.py") == "1a7eedf61557e3ffb61594ecea4e6a18d1f5315979d4d99f405aacf4822f4d1d"
    fields = ("valid", "satisfied", "progress", "utility", "joint_success")
    for i in ids:
        q = queries[i]
        for n in names:
            row, action = rowmap[i, n], actions[i, n]
            if (not row["valid"] or row["seed"] != q["seed"] or row["query_index"] != q["index"] or
                    len(row["satisfied"]) != len(q["record"]["constraints"]) or
                    not np.isfinite(row["utility"]) or not torch.isfinite(action).all() or
                    sha256(run / row["image"]) != row["image_sha256"]):
                raise ValueError("Invalid image, tensor, score or query identity")
            reference = None
            if n == "ray@0":
                reference = old[i]["candidates"][0]
            elif n == "ray@1":
                reference = next(c for c in old[i]["candidates"] if c["name"] == old[i]["selected"])
            elif n.endswith("/actual"):
                reference = post[i, n.split("/")[0]]["single"]
            if reference is not None:
                repro = {"id": i, "action": n, "same_image": row["image_sha256"] == reference["image_sha256"],
                         "same_verdict": all(row[f] == reference[f] for f in fields),
                         "parent_image": reference["image"], "parent_sha256": reference["image_sha256"]}
                logged = row.get("reproduction")
                if logged != repro:
                    if not (legacy_flag_bug and logged and logged["same_verdict"] is False and
                            repro["same_verdict"] is True and {**logged, "same_verdict": True} == repro):
                        raise ValueError("Incorrect reproduction record")
                    comparison_flag_errors.append({"id": i, "action": n, "logged_same_verdict": False,
                                                   "audited_same_verdict": True, "reason": "tuple/list comparison bug in attested runtime"})
                refs.append(repro)
        base, target = actions[i, "ray@0"], actions[i, "ray@1"]
        if not torch.equal(target, q["target"]):
            raise ValueError("Saved selected target changed")
        for b in BETAS:
            if not torch.equal(actions[i, f"ray@{b:g}"], ray_action(base, target, b)):
                raise ValueError("Ray tensor mismatch")
        for a in ARMS:
            projected, stats = project_action(base, target, actions[i, f"{a}/actual"])
            if not torch.equal(projected, actions[i, f"{a}/projected"]):
                raise ValueError("Projection tensor mismatch")
            for k, v in stats.items():
                logged = geom[i, a][k]
                if v is None:
                    if logged is not None:
                        raise ValueError("Undefined geometry changed")
                elif not np.isclose(v, logged, rtol=1e-9, atol=1e-12):
                    raise ValueError("Projection geometry mismatch")
    logged_refs = read_json(run / "reproduction.json")
    row_refs = [r["reproduction"] for r in rows if "reproduction" in r]
    if sorted(row_refs, key=lambda r: (r["id"], r["action"])) != sorted(logged_refs, key=lambda r: (r["id"], r["action"])):
        raise ValueError("Reproduction summary differs from rows")
    if len(refs) != 132 or any(result[k] != v for k, v in {
            "reproduction_images": len(refs), "same_images": sum(r["same_image"] for r in refs),
            "same_verdicts": sum(r["same_verdict"] for r in logged_refs)}.items()):
        raise ValueError("Reproduction count mismatch")

    def contrast(candidate, baseline):
        values = [(i, rowmap[i, candidate]["utility"] - rowmap[i, baseline]["utility"]) for i in ids]
        eligible = [i for i in ids if any(rowmap[i, baseline]["satisfied"])]
        damaged = [i for i in eligible if any(x and not y for x, y in zip(
            rowmap[i, baseline]["satisfied"], rowmap[i, candidate]["satisfied"]))]
        return {"utility": prompt_summary(values), "positive": sum(v > 1e-12 for _, v in values),
                "negative": sum(v < -1e-12 for _, v in values), "queries_with_existing_clause": len(eligible),
                "losing_existing_clause_ids": damaged,
                "joint_success_delta": prompt_summary([(i, int(rowmap[i, candidate]["joint_success"]) -
                                                        int(rowmap[i, baseline]["joint_success"])) for i in ids]),
                "per_query": dict(values)}

    all_actions = {n: contrast(n, "ray@0") for n in names}
    adjacent = {f"{a:g}->{b:g}": contrast(f"ray@{b:g}", f"ray@{a:g}") for a, b in zip(BETAS, BETAS[1:])}
    nonmonotonic = [i for i in ids if any(rowmap[i, f"ray@{b:g}"]["utility"] < rowmap[i, f"ray@{a:g}"]["utility"] - 1e-12
                                        for a, b in zip(BETAS, BETAS[1:]))]
    arms = {}
    for a in ARMS:
        g = [geom[i, a] for i in ids]
        arms[a] = {"actual_minus_noop": all_actions[f"{a}/actual"],
                   "projected_minus_noop": all_actions[f"{a}/projected"],
                   "projected_minus_actual": contrast(f"{a}/projected", f"{a}/actual"),
                   "target_minus_projected": contrast("ray@1", f"{a}/projected"),
                   "geometry": {"gamma_mean": float(np.mean([v["gamma"] for v in g])),
                                "gamma_range": [min(v["gamma"] for v in g), max(v["gamma"] for v in g)],
                                "outside_unit_interval": sum(v["gamma_outside_unit_interval"] for v in g),
                                "cosine_mean": float(np.mean([v["cosine"] for v in g if v["cosine"] is not None])),
                                "orthogonal_movement_energy_fraction_mean": float(np.mean([
                                    v["orthogonal_squared_norm"] / v["movement_squared_norm"] for v in g if v["movement_squared_norm"] > 0]))}}
    # Pair the same prompts, never pool four checkpoint measurements as 88 independent samples.
    step_changes = {label: {kind: contrast(f"{late}/{kind}", f"{early}/{kind}") for kind in ("actual", "projected")}
                    for label, early, late in (("keep0", "fit2", "fit4"), ("keep1", "fit2_keep", "fit4_keep"))}
    return {"run": str(run), "interpretation": "22 selected TRAIN prompts, one training seed, fixed flow10 behavior suffix. Oracle projection uses the expert target at inference; no deployment or robustness claim.",
            "statistics": "Paired prompt bootstrap, 2000 resamples, seed0; no multiplicity correction. Step-change contrasts are descriptive follow-up.",
            "integrity": {"images_verified": len(rows), "action_tensors_verified": len(actions), "geometry_rows_verified": len(geometry),
                          "reference_images": len(refs), "same_images": result["same_images"],
                          "same_verdicts": sum(r["same_verdict"] for r in refs), "runtime_logged_same_verdicts": result["same_verdicts"],
                          "comparison_flag_errors": comparison_flag_errors,
                          "mismatches": [r for r in refs if not r["same_image"] or not r["same_verdict"]],
                          "behavior_sha256": result["behavior_sha256"], "optimizer_updates": 0},
            "ray_vs_noop": {f"{b:g}": all_actions[f"ray@{b:g}"] for b in BETAS},
            "adjacent_ray": adjacent, "nonmonotonic_ray_ids": nonmonotonic, "arms": arms,
            "step2_to_step4": step_changes, "counts": result["counts"],
            "process_seconds": result["elapsed_seconds"], "source_sha256": sha256(__file__),
            "result_sha256": sha256(run / "result.json")}


def render(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    def points(ax, x, values, label, marker):
        y = np.array([v["utility"]["mean"] for v in values])
        ci = np.array([v["utility"]["ci95"] for v in values]).T
        ax.errorbar(x, y, yerr=np.stack([y-ci[0], ci[1]-y]), fmt=marker, capsize=3, label=label)

    points(axes[0], BETAS, list(summary["ray_vs_noop"].values()), "Target direction", "o-")
    axes[0].set(xlabel="Fraction of selected target correction", ylabel="Utility change vs frozen noop", title="Pure target direction")
    x = np.arange(len(ARMS))
    for offset, key, label in ((-.08, "actual_minus_noop", "Actual student"), (.08, "projected_minus_noop", "Projected student")):
        points(axes[1], x+offset, [summary["arms"][a][key] for a in ARMS], label, "o")
    axes[1].set(xticks=x, xticklabels=list(ARMS), title="Remove orthogonal movement")
    axes[1].legend()
    for ax in axes:
        ax.axhline(0, color="gray", linewidth=.8); ax.grid(alpha=.2)
    fig.suptitle("Selected TRAIN states only; oracle actions; paired prompt bootstrap 95% intervals", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"actions.{ext}", dpi=160)
    plt.close(fig)
    report = ["# Frozen action diagnostic", "", summary["interpretation"], "", summary["statistics"], "",
              "| Contrast | Mean utility delta | 95% interval | Positive / negative | Lost existing clause |",
              "|---|---:|---|---|---:|"]
    entries = [(f"ray {b} - noop", v) for b, v in summary["ray_vs_noop"].items()]
    entries += [(f"ray {k}", v) for k, v in summary["adjacent_ray"].items()]
    entries += [(f"{a}: {k}", v) for a in ARMS for k, v in summary["arms"][a].items() if k != "geometry"]
    for name, v in entries:
        report.append(f"| {name} | {v['utility']['mean']:.6f} | {v['utility']['ci95']} | {v['positive']} / {v['negative']} | {len(v['losing_existing_clause_ids'])} |")
    report += ["", f"Nonmonotonic ray queries: {len(summary['nonmonotonic_ray_ids'])}/22.", "",
               "Reproduction and tensor/image integrity:", "", "```json", json.dumps(summary["integrity"], indent=2), "```", "",
               "Every contrast uses newly generated scores. Remaining target headroom and projection gains are conditional proxy diagnostics, not held-out improvement."]
    (out / "report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", required=True); p.add_argument("--out", required=True)
    args = p.parse_args()
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    summary = summarize(run)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "summary.json", summary)
    render(summary, out)
    print(json.dumps({"integrity": summary["integrity"], "nonmonotonic_rays": len(summary["nonmonotonic_ray_ids"])}))
