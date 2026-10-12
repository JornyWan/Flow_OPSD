"""Fit saved local actions from explicit schedules, then evaluate frozen model paths."""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import time

import torch

from .action_transfer import require_exact, trace_baseline
from .backends import build_backend
from .config import Experiment
from .data import load_records
from .engine import atomic_torch_save, preserve_rng, seed_all
from .fitting_probe import cpu_tree, restore, snapshot, tensor_digest, update_once
from .io import append_jsonl, assert_environment, digest, read_json, sha256, write_json
from .learnability import action_metrics, save_verdict, single_endpoint
from .runtime import verify_run
from .training_replay import training_execution
from .verifiers import build_verifier

VERSION = "saved_supervision_v1"


def schedules(points, updates):
    """One state per prompt/update; random control routes roles within task strata."""
    byid = defaultdict(list)
    for p in points:
        byid[p["id"]].append(p)
    for group in byid.values():
        group.sort(key=lambda p: (not p["original"], p["seed"]))
        assert sum(p["original"] for p in group) == 1
    result = {a: [] for a in ("single", "multi", "random_matched")}
    for step in range(updates):
        old = [byid[i][0] for i in sorted(byid)]
        varied = [byid[i][step % len(byid[i])] for i in sorted(byid)]
        for arm, rows in (("single", old), ("multi", varied)):
            result[arm].append([{"key": p["key"], "target": p["target"]} for p in rows])
        routed = {p["key"]: p["target"] for p in varied}
        for group in sorted({p["group"] for p in varied}):
            subset = [p for p in varied if p["group"] == group]
            order = sorted(subset, key=lambda p: digest([VERSION, "random_route", step, p["key"]]))
            for p, target in zip(order, sorted(p["target"] for p in subset)):
                routed[p["key"]] = target
        result["random_matched"].append([{"key": p["key"], "target": routed[p["key"]]} for p in varied])
    return result


@torch.no_grad()
def audit(backend, queries, config, run, arm, step):
    backend.schedule(config.steps)
    for key, q in queries.items():
        action = backend.predict(q["state"], q["index"], q["context"], "student")
        row = {"arm": arm, "step": step, "key": key, "id": q["record"]["id"], "seed": q["seed"],
               "target": q["label_target"], "noop_drift_mse": float((action.double() - q["actions"]["noop"].double()).square().mean())}
        if q["label_target"] != "noop":
            row["alignment"] = action_metrics(action, q["actions"][q["label_target"]], q["actions"]["noop"])
        append_jsonl(run / "metrics.jsonl", row)


def fit(backend, queries, config, plan, run):
    optimizer = torch.optim.AdamW(backend.parameters(), lr=config.learning_rate, betas=tuple(config.adam_betas),
                                 eps=config.adam_epsilon, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.backend == "sd3" and config.precision == "fp16")
    initial = snapshot(backend, optimizer, scaler, 0)
    frozen = tensor_digest(initial["student"])
    reports = {}
    for arm in plan["arms"]:
        schedule = plan["schedules"][arm]
        restore(backend, optimizer, scaler, initial)
        assert tensor_digest(backend.student_state()) == frozen and not optimizer.state
        before, started = backend.counters.copy(), time.perf_counter()
        out = run / "checkpoints" / arm
        out.mkdir(parents=True, exist_ok=False)
        for step in range(len(schedule) + 1):
            if step:
                active = [{**queries[e["key"]], "target": queries[e["key"]]["actions"][e["target"]],
                           "accepted": e["target"] != "noop"} for e in schedule[step - 1]]
                values = update_once(backend, active, optimizer, scaler, config, keep=1)
                append_jsonl(run / "updates.jsonl", {"arm": arm, "step": step, "queries": schedule[step - 1], **values})
            if step in plan["audits"]:
                with preserve_rng():
                    audit(backend, queries, config, run, arm, step)
                    atomic_torch_save(backend.student_state(), out / f"step{step:03d}.pt")
                print({"event": "fit_audit", "arm": arm, "step": step}, flush=True)
            elif step % 16 == 0:
                print({"event": "fit_progress", "arm": arm, "step": step}, flush=True)
        assert tensor_digest(backend.behavior_state()) == frozen
        counts = dict(backend.counters - before)
        assert counts["backward_queries"] == sum(map(len, schedule))
        final_hash = tensor_digest(backend.student_state())
        expected = plan.get("reference_model_hashes", {}).get(arm)
        if expected is not None and final_hash != expected:
            raise RuntimeError(f"{arm} failed exact historical fitting replay")
        atomic_torch_save(snapshot(backend, optimizer, scaler, len(schedule)), out / "training.pt")
        reports[arm] = {"counts": counts, "optimizer_updates": len(schedule), "initial_sha256": frozen,
                        "final_sha256": final_hash, "historical_replay_exact": final_hash == expected if expected else None,
                        "elapsed_seconds": time.perf_counter() - started}
        write_json(run / "fit_reports.json", reports)
    return reports


@torch.no_grad()
def load_queries(backend, verifier, config, plan, run):
    queries = {}
    for spec in plan["points"]:
        assert sha256(spec["saved_query"]) == spec["saved_query_sha256"]
        q = torch.load(spec["saved_query"], map_location="cpu", weights_only=False)
        assert (q["record"]["id"], q["seed"], q["index"]) == (spec["id"], spec["seed"], spec["index"])
        actions = {n: q["actions"][source] for n, source in spec["action_names"].items()}
        assert {n: tensor_digest({"action": a}) for n, a in actions.items()} == spec["action_hashes"]
        actual, image = trace_baseline(backend, q["record"], q["seed"], config.steps, q["index"])
        assert torch.equal(actual["state"].cpu(), q["state"])
        assert actual["dt"] == q["dt"] and all(torch.equal(a.cpu(), b) for a, b in zip(actual["context"], q["context"]))
        q.update(actions={n: a.to(backend.device) for n, a in actions.items()}, state=q["state"].to(backend.device), label_target=spec["target"])
        assert torch.equal(backend.predict(q["state"], q["index"], q["context"], "behavior"), q["actions"]["noop"])
        for mode in ("noop", spec["target"]) if spec["target"] != "noop" else ("noop",):
            endpoint = image if mode == "noop" else single_endpoint(backend, q, q["actions"][mode], config.steps)
            verdict = save_verdict(run, backend, verifier, q["record"], endpoint, [VERSION, "reference", spec["key"], mode], config.utility_joint_weight)
            require_exact(verdict, spec["images"][mode]["sha256"])
            append_jsonl(run / "references.jsonl", {"key": spec["key"], "mode": mode, "exact": True, **verdict})
        queries[spec["key"]] = q
    print({"event": "reference_gate_passed", "images": plan["reference_images"]}, flush=True)
    return queries


@torch.no_grad()
def evaluate(backend, verifier, records, config, plan, run):
    models = {"joint128": plan["student"], **{a: str(run / "checkpoints" / a / f"step{plan['updates']:03d}.pt") for a in plan["arms"]}}
    queries, hashes, noises = {}, {}, {}
    (run / "eval_queries").mkdir()
    for model, path in models.items():
        state = torch.load(path, map_location="cpu", weights_only=True)
        backend.load_student(state, refresh_behavior=False)
        hashes[model] = tensor_digest(state)
        for pair in plan["evaluation"]:
            pid, record = pair["pair_id"], records[pair["id"]]
            noise = tensor_digest({"noise": backend.initial(pair["seed"])})
            assert noises.get(pid, noise) == noise
            noises[pid] = noise
            if model == "joint128" and pair["single_action"]:
                q, image = trace_baseline(backend, record, pair["seed"], config.steps, plan["index"])
                queries[pid] = q
                atomic_torch_save(cpu_tree(q), run / "eval_queries" / f"{pid}.pt")
            else:
                image = backend.generate(record, pair["seed"], config.steps)
            paths = {"full": image}
            if model != "joint128" and pair["single_action"]:
                q = queries[pid]
                action = backend.predict(q["state"], q["index"], q["context"], "student")
                paths["single"] = single_endpoint(backend, q, action, config.steps)
            for mode, endpoint in paths.items():
                verdict = save_verdict(run, backend, verifier, record, endpoint, [VERSION, pid, model, mode], config.utility_joint_weight)
                append_jsonl(run / "endpoints.jsonl", {**pair, "model": model, "path": mode,
                             "student_sha256": hashes[model], "noise_sha256": noise, **verdict})
        assert tensor_digest(backend.student_state()) == hashes[model]
        print({"event": "evaluation_complete", "model": model}, flush=True)
    return {"model_hashes": hashes, "noise_hashes": noises}


def run_study(run):
    run = Path(run).resolve()
    verify_run(run)
    assert_environment(read_json(run / "manifest.json")["environment"])
    for p, h in read_json(run / "probe_attestation.json")["inputs"].items():
        assert sha256(p) == h, p
    plan, config, site = read_json(run / "probe_plan.json"), Experiment.load(run / "config.json"), read_json(run / "site.json")
    assert plan["version"] == VERSION and plan["schedules"] == schedules(plan["points"], plan["updates"])
    if any((run / n).exists() for n in ("RUNNING", "result.json", "failure.json", "references.jsonl")):
        raise FileExistsError("Use a new attempt")
    (run / "RUNNING").write_text(f"pid={os.getpid()}\n")
    start = time.perf_counter()
    try:
        seed_all(config.seed)
        backend = build_backend(config, site, run / "upstream", "cuda", training=True)
        backend.load_student(torch.load(plan["student"], map_location="cpu", weights_only=True), refresh_behavior=True)
        records = {r["id"]: r for r in load_records(config.dataset, "train")}
        verifier = build_verifier(config, site, backend.device, list(records.values()))
        queries = load_queries(backend, verifier, config, plan, run)
        with training_execution(plan) as execution:
            write_json(run / "training_execution.json", {"execution": execution})
            reports = fit(backend, queries, config, plan, run)
        write_json(run / "fitting_result.json", {"arms": reports, "training_execution": execution,
                   "counts": dict(backend.counters + verifier.counters), "elapsed_seconds": time.perf_counter() - start})
        checks = evaluate(backend, verifier, records, config, plan, run)
        assert tensor_digest(backend.behavior_state()) == checks["model_hashes"]["joint128"]
        counts = dict(backend.counters + verifier.counters)
        for key, value in plan["expected_counts"].items():
            assert counts.get(key, 0) == value, (key, counts.get(key), value)
        write_json(run / "result.json", {"status": "complete", "arms": reports, **checks, "counts": counts,
                   "training_execution": execution,
                   "historical_model_comparison": {a: checks["model_hashes"][a] == h for a, h in plan.get("historical_reference_model_hashes", {}).items()},
                   "elapsed_seconds": time.perf_counter() - start, "new_teacher_queries": 0,
                   "evaluation_images": plan["evaluation_images"], "reference_images": plan["reference_images"],
                   "optimizer_policy": "identical reset AdamW/scaler/RNG; raw joint128 warm start; no EMA/behavior refresh",
                   "output_hashes": {str(p.relative_to(run)): sha256(p) for p in run.rglob("*") if p.is_file() and
                                     (p.suffix == ".pt" and "upstream" not in p.parts or p.parent == run and p.suffix == ".jsonl")}})
    except Exception as exc:
        write_json(run / "failure.json", {"error": repr(exc), "elapsed_seconds": time.perf_counter() - start})
        raise
    finally:
        (run / "RUNNING").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True)
    run_study(parser.parse_args().run)
