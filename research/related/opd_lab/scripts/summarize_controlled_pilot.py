"""Validate completed phase-2 evidence and export the predeclared comparison matrix."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from opdlab.evaluate import adapter_identity, compare, read_evaluation
from opdlab.io import read_json, sha256, tree_fingerprint, write_json
from opdlab.report import prompt_summary
from opdlab.runtime import verify_run


def transfer_audit(queries, fit, post):
    """Descriptive repeated training-state diagnostics, not a causal mediation estimate."""
    identity = lambda r: (r["outer_epoch"], r["id"], r["seed"], r["query_index"])
    fitted, updated = ({identity(r): r for r in rows} for rows in (fit["rows"], post))
    if {identity(q) for q in queries} != fitted.keys() or fitted.keys() != updated.keys():
        raise ValueError("Missing transfer audit identities")
    groups = {}
    for accepted in (True, False):
        selected = [q for q in queries if q["accepted"] == accepted]
        if not selected:
            continue
        rows = []
        for q in selected:
            before = q["candidates"][0]
            target = next(c for c in q["candidates"] if c["name"] == q["selected"])
            f, p = fitted[identity(q)], updated[identity(q)]
            verdicts = {"target": target, "local": p["postfit_single_intervention"],
                        "full": p["postfit_full_policy_train_prompt"]}
            row = {"has_satisfied_clause": any(before["satisfied"]),
                   "loss_improved": f["post_update_target_loss"] < f["pre_update_target_loss"]}
            for name, v in verdicts.items():
                row[name + "_delta"] = v["utility"] - before["utility"]
                row[name + "_damage"] = any(old and not new for old, new in zip(before["satisfied"], v["satisfied"]))
            rows.append(row)
        improved = [r for r in rows if r["loss_improved"]]
        protected = [r for r in rows if r["has_satisfied_clause"]]
        groups["accepted" if accepted else "rejected"] = {
            "queries": len(rows), "queries_with_satisfied_clause": len(protected),
            "loss_improved_queries": len(improved),
            "local_positive_given_loss_improved": sum(r["local_delta"] > 0 for r in improved) / len(improved) if improved else None,
            **{name + "_mean_delta": float(np.mean([r[name + "_delta"] for r in rows])) for name in verdicts},
            **{name + "_damaged_queries": sum(r[name + "_damage"] for r in protected) for name in verdicts}}
    return groups


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--campaign", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--review", required=True, help="Completed predeclared validation image review")
    args = p.parse_args()
    campaign, out = Path(args.campaign).resolve(), Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    plan = read_json(campaign / "runs.json")
    if sha256(campaign / "PROTOCOL.md") != plan["protocol_sha256"]:
        raise ValueError("Predeclared protocol changed")
    training, artifacts = {}, {}
    for name, item in plan["training_runs"].items():
        run = Path(item)
        verify_run(run)
        status = read_json(run / "result.json")
        if status["status"] != "complete" or status["outer_epoch"] != 12:
            raise ValueError("Training not complete at its fixed budget")
        queries = [json.loads(s) for s in (run / "queries-rank-0.jsonl").read_text().splitlines()]
        specs = read_json(run / "training_queries.json")["queries"]
        if len(queries) != len(specs):
            raise ValueError("Training queries missing")
        for q, spec in zip(queries, specs):
            if {k: q[k] for k in spec} != spec:
                raise ValueError("Training did not follow its frozen schedule")
            for c in q["candidates"]:
                if not c["valid"] or sha256(run / c["image"]) != c["image_sha256"]:
                    raise ValueError("Training candidate evidence invalid")
        accepted = [q for q in queries if q["accepted"]]
        choices = Counter(q["selected"] for q in queries)
        fit = read_json(run / "postfit_audit.json")
        total_pre = sum(v["accepted_loss_sum_before"] for v in fit["epochs"].values())
        total_after = sum(v["accepted_loss_sum_after"] for v in fit["epochs"].values())
        training[name] = {**status, "queries": len(queries), "accepted": len(accepted),
                          "selected": dict(choices), "pooled_fitting_after_over_before": total_after / total_pre if total_pre else None,
                          "interpretation": "Fitting ratio pools evolving training states; not a validation metric."}
        counts = Counter()
        for row in map(json.loads, (run / "metrics.jsonl").read_text().splitlines()):
            counts.update(row["counts"])
        training[name]["measured_operation_counts"] = dict(counts)
        for metric in ["target_utility", "postfit_local_utility", "postfit_full_train_utility"]:
            training[name][metric + "_delta_vs_behavior"] = float(np.mean([
                row[metric] - row["baseline_utility"] for row in fit["rows"]]))
        rejected = [row for row in fit["rows"] if not row["accepted"]]
        training[name]["rejected_state_local_utility_drift"] = (float(np.mean([
            row["postfit_local_utility"] - row["baseline_utility"] for row in rejected])) if rejected else None)
        post = [json.loads(s) for s in (run / "postfit-rank-0.jsonl").read_text().splitlines()]
        training[name]["transfer_by_acceptance"] = transfer_audit(queries, fit, post)
        by_time = {}
        for lo, hi in ((0, 2), (3, 5), (6, 8)):
            group = [r for r in fit["rows"] if lo <= r["query_index"] <= hi]
            chosen = [r for r in group if r["accepted"]]
            by_time[f"{lo}-{hi}"] = {"queries": len(group), "accepted": len(chosen),
                "accepted_prefit_loss_fraction": sum(r["pre_update_target_loss"] for r in chosen) / total_pre if total_pre else None,
                **{metric + "_delta": float(np.mean([r[metric] - r["baseline_utility"] for r in chosen])) if chosen else None
                   for metric in ("target_utility", "postfit_local_utility", "postfit_full_train_utility")}}
        training[name]["exploratory_time_bins"] = {"scope": "Post-hoc equal-index bins of repeated training queries; descriptive only.", "bins": by_time}
        for filename in ["result.json", "metrics.jsonl", "queries-rank-0.jsonl", "postfit-rank-0.jsonl", "postfit_audit.json"]:
            artifacts[str(run / filename)] = sha256(run / filename)
    cells = {}
    seen_cells = set()
    for item in read_json(campaign / "evaluation_jobs.json")["jobs"]:
        cell = (item["sampler"], item["steps"])
        if cell in seen_cells or list(cell) not in plan["evaluation"]["cells"]:
            raise ValueError("Unexpected or duplicate evaluation cell")
        seen_cells.add(cell)
        job = Path(item["job"])
        command = read_json(job / "submission.json")["worker_command"]
        context = Path(command[command.index("--context") + 1])
        verify_run(context)
        if (tree_fingerprint(job / "framework/python", full=True)["inventory_digest"]
                != read_json(context / "manifest.json")["framework"]["inventory_digest"]):
            raise ValueError("Evaluation worker package differs from the frozen evaluation context")
        attestation = read_json(job / "input_attestation.json")
        result = read_json(job / "evaluations/result.json")
        if (result["status"] != "complete" or result["runs_manifest_sha256"] != attestation["runs.json"]
                or result["ocr_config_sha256"] != attestation["ocr_audit_config.json"]
                or sha256(job / "run_controlled_evaluation.py") != attestation["worker_script_sha256"]):
            raise ValueError("Evaluation result or its frozen inputs differ")
        if set(result["arms"]) != {"base", *training}:
            raise ValueError("Evaluation arms differ from the frozen plan")
        primary_tables, secondary_tables = {}, {}
        for arm in result["arms"]:
            protocol, rows = read_evaluation(job / "evaluations" / arm)
            adapter = (Path(plan["training_runs"][arm]) / f"checkpoints/epoch-{plan['evaluation']['checkpoint_epoch']:06d}/raw_lora"
                       if arm != "base" else None)
            if read_json(job / "evaluations" / arm / "checkpoint.json") != adapter_identity(adapter, "raw" if adapter else "base"):
                raise ValueError("Evaluated weights differ from the predeclared final raw checkpoint")
            if ((protocol["sampler"], protocol["steps"]) != cell
                    or protocol["split"] != plan["evaluation"]["split"]
                    or protocol["replicate_seeds"] != plan["evaluation"]["seeds"]
                    or protocol["adapter_kind"] != ("base" if arm == "base" else "raw")):
                raise ValueError("Evaluation protocol differs from the frozen plan")
            if len(rows) != 64:
                raise ValueError("Expected all 32 validation prompts and both seeds")
            primary_tables[arm] = {(r["id"], r["replicate"]): r for r in rows.values()}
            secondary = read_json(job / "evaluations" / arm / "secondary_ocr.json")["rows"]
            secondary_tables[arm] = {(r["id"], r["replicate"]): r for r in secondary}
            if (len(secondary) != 64 or secondary_tables[arm].keys() != primary_tables[arm].keys()
                    or any(r["image_sha256"] != primary_tables[arm][key]["image_sha256"]
                           for key, r in secondary_tables[arm].items())):
                raise ValueError("Secondary OCR does not cover the same unchanged images")
            for filename in ["protocol.json", "checkpoint.json", "per_image.jsonl", "secondary_ocr.json"]:
                path = job / "evaluations" / arm / filename
                artifacts[str(path)] = sha256(path)
        result["ocr_reader_disagreement"] = {}
        for arm, rows in primary_tables.items():
            result["ocr_reader_disagreement"][arm] = prompt_summary([
                (key[0], row["satisfied"][row["constraint_ids"].index("text")]
                 != secondary_tables[arm][key]["satisfied"][row["constraint_ids"].index("text")])
                for key, row in rows.items()])
        for contrast in result["contrasts"].values():
            method = Path(contrast["primary"]["method"]).name
            baseline = Path(contrast["primary"]["baseline"]).name
            if contrast["primary"] != compare(job / "evaluations" / baseline, job / "evaluations" / method):
                raise ValueError("Saved comparison differs from the underlying paired evaluation")
            contrast["secondary_ocr"]["text_accuracy_delta"] = prompt_summary([
                (key[0], int(secondary_tables[method][key]["satisfied"][row["constraint_ids"].index("text")])
                 - int(secondary_tables[baseline][key]["satisfied"][row["constraint_ids"].index("text")]))
                for key, row in primary_tables[baseline].items()])
        cells[f"{item['sampler']}{item['steps']}"] = result
    if len(cells) != len(plan["evaluation"]["cells"]):
        raise ValueError("Evaluation matrix incomplete")
    review = Path(args.review).resolve()
    observations = read_json(review / "comparison.json")
    if (observations["observations_sha256"] != sha256(review / "assistant_observations.json")
            or observations["key_sha256"] != sha256(review / "key.json")
            or read_json(review / "provenance.json")["plan_sha256"] != sha256(campaign / "qualitative_validation_plan.json")
            or len(observations["rows"]) != 80):
        raise ValueError("Predeclared image review evidence differs or is incomplete")
    for row in observations["rows"]:
        if sha256(row["image"]) != row["image_sha256"]:
            raise ValueError("Reviewed source image changed")
    for path in review.iterdir():
        if path.is_file():
            artifacts[str(path)] = sha256(path)
    out.mkdir(parents=True)
    result = {"scope": "Controlled development screen; one training seed, 32 validation prompts; no test used.",
              "training": training, "cells": cells, "artifacts_sha256": artifacts,
              "validation_image_review": {"path": str(review), "scope": observations["scope"],
                  "counts_relative_to_provisional_observations": observations["counts_relative_to_provisional_observations"]},
              "scheduler_accounting": read_json(campaign / "slurm_status.json")}
    write_json(out / "results.json", result)
    lines = ["# Phase 2 controlled screen", "", result["scope"], "",
             "All cells use final epoch-12 raw LoRA; no checkpoint or EMA selection.", "",
             "| Cell | Arm | Utility delta vs base | Prompt bootstrap 95% interval | Joint-success delta |",
             "|---|---|---:|---|---:|"]
    for cell, value in cells.items():
        for arm in training:
            contrast = value["contrasts"][arm + "_minus_base"]["primary"]
            u, j = contrast["utility"], contrast["joint_success"]
            lines.append(f"| {cell} | {arm} | {u['paired_delta']:+.6f} | {u['ci95']} | {j['paired_delta']:+.6f} |")
    lines += ["", "## Absolute base performance", "",
              "| Cell | Utility | Joint success |", "|---|---:|---:|"]
    for cell, value in cells.items():
        base = value["arms"]["base"]
        lines.append(f"| {cell} | {base['utility']:.6f} | {base['joint_success']:.6f} |")
    lines += ["", "## Predeclared mechanism contrasts", "",
              "| Cell | Contrast | Utility delta | Prompt bootstrap 95% interval | Joint-success delta |",
              "|---|---|---:|---|---:|"]
    for cell, value in cells.items():
        for name, comparison in value["contrasts"].items():
            if name.endswith("_minus_base"):
                continue
            u, j = comparison["primary"]["utility"], comparison["primary"]["joint_success"]
            lines.append(f"| {cell} | {name} | {u['paired_delta']:+.6f} | {u['ci95']} | {j['paired_delta']:+.6f} |")
    lines += ["", "## Training-state diagnostics", "",
              "Means below include all 192 evolving training queries, including rejections; they are not held-out gains.", "",
              "| Arm | Accepted | Updates | Fitting after/before | Selected target delta | Post-fit local delta | Post-fit full delta | Rejected local drift |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, stats in training.items():
        drift = stats["rejected_state_local_utility_drift"]
        drift_text = f"{drift:+.6f}" if drift is not None else "n/a"
        lines.append(f"| {arm} | {stats['accepted']}/192 | {stats['optimizer_step']} | "
                     f"{stats['pooled_fitting_after_over_before']:.6f} | "
                     f"{stats['target_utility_delta_vs_behavior']:+.6f} | "
                     f"{stats['postfit_local_utility_delta_vs_behavior']:+.6f} | "
                     f"{stats['postfit_full_train_utility_delta_vs_behavior']:+.6f} | {drift_text} |")
    lines += ["", "## Audit-only OCR sensitivity", "",
              "Tesseract replaces only the OCR clause; the GenEval clauses remain shared. This is not ground truth.", "",
              "| Cell | Arm vs base | Secondary utility delta | Secondary joint delta | Secondary text delta | Readers disagree on text |",
              "|---|---|---:|---:|---:|---:|"]
    for cell, value in cells.items():
        for arm in training:
            s = value["contrasts"][arm + "_minus_base"]["secondary_ocr"]
            lines.append(f"| {cell} | {arm} | {s['utility']['mean']:+.6f} | "
                         f"{s['joint_success']['mean']:+.6f} | {s['text_accuracy_delta']['mean']:+.6f} | "
                         f"{value['ocr_reader_disagreement'][arm]['mean']:.6f} |")
    lines += ["", "Component accuracy, losses of previously satisfied clauses, mechanism contrasts and secondary OCR results are in results.json.",
              "Intervals resample prompts, not training seeds; they are exploratory and not multiplicity-corrected.",
              "The audit-only Tesseract reader had substantial false negatives in calibration. It is not a ground-truth judge."]
    lines += ["", f"The predeclared 80-image implementing-assistant review is in [{review.name}]({review}/report.md).",
              "Its provisional observations were saved before revealing method identities and scores; uncertain readings were retained."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax, metric, title in zip(axes, ["utility", "joint_success"], ["Utility change vs base", "Joint-success change vs base"]):
        for i, arm in enumerate(training):
            stats = [cell["contrasts"][arm + "_minus_base"]["primary"][metric] for cell in cells.values()]
            means = np.array([v["paired_delta"] for v in stats]); ci = np.array([v["ci95"] for v in stats])
            ax.errorbar(np.arange(len(cells)) + (i - 1.5) * .12, means,
                        yerr=np.maximum(0, [means - ci[:, 0], ci[:, 1] - means]), fmt="o", capsize=3, label=arm)
        ax.axhline(0, color="black", linewidth=.8); ax.grid(axis="y", alpha=.2)
        ax.set_xticks(range(len(cells)), list(cells)); ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.suptitle("12 updates at most; one training seed; 32 validation prompts; prompt bootstrap 95% intervals", fontsize=10)
    fig.savefig(out / "validation_deltas.png", dpi=200); fig.savefig(out / "validation_deltas.pdf")
    plt.close(fig)
    print(json.dumps({"report": str(out), "training": training}))


if __name__ == "__main__":
    main()
