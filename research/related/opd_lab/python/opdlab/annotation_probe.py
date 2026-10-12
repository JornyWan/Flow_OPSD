"""Label-assisted fixed-target mechanism probe; not an automatic or native OPD run."""
import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
import time

import torch

from .backends import build_backend
from .config import Experiment
from .data import load_records
from .engine import atomic_torch_save, preserve_rng, seed_all
from .fitting_probe import restore, snapshot, tensor_digest, update_once
from .io import append_jsonl, assert_environment, digest, read_json, sha256, stable_seed, write_json
from .learnability import action_metrics, save_verdict, single_endpoint
from .method import transition_mean_loss
from .runtime import verify_run
from .verifiers import build_verifier


VERSION = "annotation_mechanism_v1"
ARMS = ("label_selected", "random_matched", "all_teacher")


def cohort(rows, reviewed):
    """Take every known-safe state, then outcome-blind matched preservation points."""
    labels = {q["id"]: q for q in reviewed}
    positives = {}
    for row in rows:
        q = labels[row["id"]]
        eligible = []
        for i, c in enumerate(row["candidates"]):
            if c["name"] != "noop" and q["comparisons"][c["name"]]["safe_repair"]:
                eligible.append(((len(q["comparisons"][c["name"]]["repaired_clauses"]), -c["distance"], -i), c["name"]))
        if eligible:
            positives[row["id"]] = max(eligible)[1]
    if len(positives) != 6:
        raise ValueError("Expected all six frozen phase12 safe states")
    groups = Counter(labels[i]["group"] for i in positives)
    if sorted(groups.values()) != [2, 2, 2]:
        raise ValueError("Unexpected known-positive strata")
    keep = []
    for group, count in sorted(groups.items()):
        pool = [r["id"] for r in rows if labels[r["id"]]["group"] == group and r["id"] not in positives]
        keep.extend(sorted(pool, key=lambda i: digest([VERSION, "preservation", i]))[:count])
    specs = []
    byid = {r["id"]: r for r in rows}
    for ident in sorted([*positives, *keep]):
        r, q = byid[ident], labels[ident]
        specs.append({"id": ident, "group": q["group"], "source_case": q["case"], "seed": r["seed"],
                      "index": r["query_index"], "label_target": positives.get(ident, "noop"),
                      "saved_query": r["saved_query"], "saved_query_sha256": r["saved_query_sha256"],
                      "action_hashes": r["action_hashes"], "images": {c["name"]: {"image": c["image"], "sha256": c["image_sha256"]} for c in r["candidates"]}})
    return specs


def control_targets(specs):
    """Match accepted state count, task stratum and teacher counts without outcomes."""
    selections = {a: {} for a in ARMS}
    groups = defaultdict(list)
    for q in specs:
        groups[q["group"]].append(q)
        selections["label_selected"][q["id"]] = q["label_target"]
    for group, points in sorted(groups.items()):
        teachers = sorted(q["label_target"] for q in points if q["label_target"] != "noop")
        if len(points) != 2 * len(teachers) or not teachers:
            raise ValueError("Each group needs equal positive and preservation counts")
        ordered = sorted(points, key=lambda q: digest([VERSION, "random_accept", q["id"]]))
        for block, subset in enumerate((ordered[:len(teachers)], ordered[len(teachers):])):
            routed = sorted(subset, key=lambda q: digest([VERSION, "teacher_route", q["id"]]))
            for q, teacher in zip(routed, teachers):
                selections["random_matched"][q["id"]] = teacher if block == 0 else "noop"
                selections["all_teacher"][q["id"]] = teacher
    return selections


def evaluation_pairs(specs, records, excluded):
    fitted = {q["id"]: q for q in specs}
    held = []
    for group in sorted({r["group"] for r in records}):
        pool = [r["id"] for r in records if r["group"] == group and r["id"] not in fitted]
        held.extend(sorted(pool, key=lambda i: digest([VERSION, "unfitted_prompt", i]))[:2])
    if len(held) != 8 or any(r["split"] != "train" for r in records):
        raise ValueError("Expected eight additional train prompts; no validation/test inputs")
    pairs = []
    for q in specs:
        role = "positive" if q["label_target"] != "noop" else "preservation"
        for replicate in range(2):
            pairs.append({"id": q["id"], "cohort": role + "_fresh_seed", "replicate": replicate,
                          "seed": stable_seed(15015, VERSION, q["id"], replicate)})
        if role == "positive":
            pairs.append({"id": q["id"], "cohort": "fitting_seed_diagnostic", "replicate": 0, "seed": q["seed"]})
    for ident in held:
        pairs.append({"id": ident, "cohort": "unfitted_prompt_fresh_seed", "replicate": 0,
                      "seed": stable_seed(15015, VERSION, ident, 0)})
    fresh = [p["seed"] for p in pairs if p["cohort"] != "fitting_seed_diagnostic"]
    if len(fresh) != 32 or len(set(fresh)) != len(fresh) or set(fresh) & set(excluded):
        raise ValueError("Fresh evaluation noise collides with archived data")
    for p in pairs:
        p["pair_id"] = digest([VERSION, p])[:24]
    return pairs


@torch.no_grad()
def audit_points(backend, points, targets, config, run, arm, step):
    backend.schedule(config.steps)
    for q in points:
        prediction = backend.predict(q["state"], q["index"], q["context"], "student")
        name = targets[q["record"]["id"]]
        values = {"arm": arm, "step": step, "id": q["record"]["id"], "target_name": name,
                  "fitting_loss": float(transition_mean_loss(prediction, q["actions"][name], q["dt"])),
                  "noop_drift_mse": float((prediction.double() - q["actions"]["noop"].double()).square().mean())}
        if q["label_target"] != "noop":
            values["label_target_alignment"] = action_metrics(prediction, q["actions"][q["label_target"]], q["actions"]["noop"])
        append_jsonl(run / "metrics.jsonl", values)


def fit_models(backend, points, config, plan, run):
    optimizer = torch.optim.AdamW(backend.parameters(), lr=config.learning_rate, betas=tuple(config.adam_betas),
                                 eps=config.adam_epsilon, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.backend == "sd3" and config.precision == "fp16")
    initial = snapshot(backend, optimizer, scaler, 0)
    initial_hash = tensor_digest(initial["student"])
    reports = {}
    for arm in ARMS:
        targets = plan["targets"][arm]
        restore(backend, optimizer, scaler, initial)
        if tensor_digest(backend.student_state()) != initial_hash or optimizer.state:
            raise RuntimeError("Arm weight/optimizer reset differs")
        active = [{**q, "target": q["actions"][targets[q["record"]["id"]]],
                   "accepted": targets[q["record"]["id"]] != "noop"} for q in points]
        before, start = backend.counters.copy(), time.perf_counter()
        out = run / "checkpoints" / arm
        out.mkdir(parents=True, exist_ok=False)
        for step in range(plan["updates"] + 1):
            if step:
                fitting = update_once(backend, active, optimizer, scaler, config, keep=1)
                append_jsonl(run / "updates.jsonl", {"arm": arm, "step": step, "backward_queries": len(active), **fitting})
            if step in plan["audits"]:
                with preserve_rng(), torch.no_grad():
                    if tensor_digest(backend.behavior_state()) != initial_hash:
                        raise RuntimeError("Frozen behavior changed")
                    audit_points(backend, points, targets, config, run, arm, step)
                    atomic_torch_save(backend.student_state(), out / f"step{step:03d}.pt")
                print({"event": "fit_audit", "arm": arm, "step": step}, flush=True)
            elif step % 16 == 0:
                print({"event": "fit_progress", "arm": arm, "step": step}, flush=True)
        atomic_torch_save(snapshot(backend, optimizer, scaler, plan["updates"]), out / "training.pt")
        counts = dict(backend.counters - before)
        if counts["backward_queries"] != len(active) * plan["updates"]:
            raise RuntimeError("Unmatched fitting budget")
        reports[arm] = {"counts": counts, "elapsed_seconds": time.perf_counter() - start,
                        "initial_student_sha256": initial_hash, "final_student_sha256": tensor_digest(backend.student_state()),
                        "expert_targets": sum(q["accepted"] for q in active), "optimizer_updates": plan["updates"]}
    return reports


@torch.no_grad()
def evaluate(backend, verifier, records, config, plan, run):
    records = {r["id"]: r for r in records}
    models = {"joint128": plan["student"], **{a: str(run / "checkpoints" / a / f'step{plan["updates"]:03d}.pt') for a in ARMS}}
    noises, hashes = {}, {}
    for model, path in models.items():
        state = torch.load(path, map_location="cpu", weights_only=True)
        backend.load_student(state, refresh_behavior=False)
        hashes[model] = tensor_digest(state)
        for i, pair in enumerate(plan["evaluation"]):
            noise = tensor_digest({"noise": backend.initial(pair["seed"])})
            if pair["pair_id"] in noises and noises[pair["pair_id"]] != noise:
                raise RuntimeError("Paired model noises differ")
            noises[pair["pair_id"]] = noise
            record = records[pair["id"]]
            image = backend.generate(record, pair["seed"], config.steps, sampler="flow")
            verdict = save_verdict(run, backend, verifier, record, image, [VERSION, model, pair["pair_id"]], config.utility_joint_weight)
            append_jsonl(run / "endpoints.jsonl", {**pair, "model": model, "student_sha256": hashes[model], "noise_sha256": noise,
                         "constraint_ids": [c["id"] for c in record["constraints"]], **verdict})
            if (i + 1) % 10 == 0:
                print({"event": "evaluation", "model": model, "complete": i + 1}, flush=True)
        if tensor_digest(backend.student_state()) != hashes[model]:
            raise RuntimeError("Evaluation weights changed")
    return {"model_hashes": hashes, "noise_hashes": noises}


def run_probe(run):
    run = Path(run).resolve()
    verify_run(run)
    assert_environment(read_json(run / "manifest.json")["environment"])
    for p, h in read_json(run / "probe_attestation.json")["inputs"].items():
        if sha256(p) != h:
            raise ValueError(f"Frozen input changed: {p}")
    if any((run / p).exists() for p in ("RUNNING", "result.json", "failure.json", "updates.jsonl")):
        raise FileExistsError("Use a new attempt")
    (run / "RUNNING").write_text(f"pid={os.getpid()}\n")
    start = time.perf_counter()
    try:
        plan, config, site = read_json(run / "probe_plan.json"), Experiment.load(run / "config.json"), read_json(run / "site.json")
        if plan["version"] != VERSION or plan["targets"] != control_targets(plan["points"]):
            raise ValueError("Frozen target rules changed")
        seed_all(config.seed)
        backend = build_backend(config, site, run / "upstream", "cuda", training=True)
        initial = torch.load(plan["student"], map_location="cpu", weights_only=True)
        backend.load_student(initial, refresh_behavior=True)
        initial_hash = tensor_digest(initial)
        records = load_records(config.dataset, "train")
        verifier = build_verifier(config, site, backend.device, records)
        points = []
        source = Path(plan["source"])
        for spec in plan["points"]:
            q = torch.load(source / spec["saved_query"], map_location="cpu", weights_only=False)
            if (q["record"]["id"], q["seed"], q["index"]) != (spec["id"], spec["seed"], spec["index"]):
                raise ValueError("Saved query identity mismatch")
            for name, action in q["actions"].items():
                if tensor_digest({"action": action}) != spec["action_hashes"][name]:
                    raise ValueError("Saved action mismatch")
            q["state"] = q["state"].to(backend.device)
            q["actions"] = {n: t.to(backend.device) for n, t in q["actions"].items()}
            q["label_target"] = spec["label_target"]
            backend.schedule(config.steps)
            with torch.no_grad():
                action = backend.predict(q["state"], q["index"], q["context"], "behavior")
                if not torch.equal(action, q["actions"]["noop"]):
                    raise RuntimeError("Saved noop does not match joint128")
                base = backend.generate(q["record"], q["seed"], config.steps)
                verdict = save_verdict(run, backend, verifier, q["record"], base, ["reference", spec["id"], "noop"], config.utility_joint_weight)
                if verdict["image_sha256"] != spec["images"]["noop"]["sha256"]:
                    raise RuntimeError("Baseline image failed exact replay")
                append_jsonl(run / "references.jsonl", {"id": spec["id"], "mode": "noop", "exact": True, **verdict})
                if spec["label_target"] != "noop":
                    endpoint = single_endpoint(backend, q, q["actions"][spec["label_target"]], config.steps)
                    verdict = save_verdict(run, backend, verifier, q["record"], endpoint, ["reference", spec["id"], "target"], config.utility_joint_weight)
                    if verdict["image_sha256"] != spec["images"][spec["label_target"]]["sha256"]:
                        raise RuntimeError("Selected target image failed exact replay")
                    append_jsonl(run / "references.jsonl", {"id": spec["id"], "mode": "target", "exact": True, **verdict})
            points.append(q)
        arms = fit_models(backend, points, config, plan, run)
        checks = evaluate(backend, verifier, records, config, plan, run)
        if tensor_digest(backend.behavior_state()) != initial_hash:
            raise RuntimeError("Frozen behavior changed")
        write_json(run / "result.json", {"status": "complete", "arms": arms, **checks,
                   "elapsed_seconds": time.perf_counter() - start, "counts": dict(backend.counters + verifier.counters),
                   "new_teacher_queries": 0, "source_collection_teacher_queries": 128,
                   "available_saved_teacher_actions": 2 * len(points), "distinct_fitting_queries": len(points),
                   "evaluated_images": len(plan["evaluation"]) * 4, "reference_images": 18,
                   "initial_student_sha256": initial_hash, "optimizer_policy": "identical reset AdamW/scaler/RNG; raw weight warm start, no EMA or behavior refresh",
                   "output_hashes": {p.name: sha256(p) for p in run.glob("*.jsonl")},
                   "checkpoint_hashes": {str(p.relative_to(run)): sha256(p) for p in (run / "checkpoints").rglob("*.pt")}})
    except Exception as exc:
        write_json(run / "failure.json", {"error": repr(exc), "elapsed_seconds": time.perf_counter() - start})
        raise
    finally:
        (run / "RUNNING").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True)
    run_probe(parser.parse_args().run)
