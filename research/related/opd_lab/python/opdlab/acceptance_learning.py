"""Explicit saved-action regression, including batches whose targets are all noop."""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import time

import torch

from .backends import build_backend
from .config import Experiment
from .data import load_records
from .engine import atomic_torch_save, preserve_rng, seed_all
from .fitting_probe import restore, snapshot, tensor_digest
from .io import append_jsonl, assert_environment, digest, read_json, sha256, write_json
from .method import transition_mean_loss
from .runtime import verify_run
from .supervision_study import audit, evaluate, load_queries
from .training_replay import training_execution
from .verifiers import build_verifier

VERSION = "single_expert_acceptance_learning_v1"
ARMS = ["all_teacher", "certified", "random_matched"]


def schedules(points, updates):
    groups = defaultdict(list)
    for p in points:
        groups[p["id"]].append(p)
    targets, randomization = {}, {}
    for pid, group in sorted(groups.items()):
        group.sort(key=lambda p: (p["replicate"], p["key"]))
        assert len({p["key"] for p in group}) == len(group)
        if group[0]["role"] == "preservation":
            assert len(group) == 1 and group[0]["target"] == "noop"
            targets[group[0]["key"]] = "noop"
            continue
        assert len(group) == 8 and {p["replicate"] for p in group} == set(range(8))
        # One outcome-independent ordering, without retries or requiring nonidentity.
        order = sorted(group, key=lambda p: digest([VERSION, "random_accept", pid, p["key"]]))
        randomization[pid] = []
        for destination, source in zip(group, order):
            targets[destination["key"]] = source["target"]
            randomization[pid].append({"destination": destination["key"], "source": source["key"], "target": source["target"]})
    result = {a: [] for a in ARMS}
    for step in range(updates):
        active = [g[step % len(g)] for _, g in sorted(groups.items())]
        for arm in ARMS:
            row = []
            for p in active:
                target = ("geneval@1" if p["role"] == "positive" else "noop") if arm == "all_teacher" else p["target"] if arm == "certified" else targets[p["key"]]
                row.append({"key": p["key"], "target": target})
            result[arm].append(row)
    return result, randomization


def update_saved_actions(backend, active, optimizer, scaler, config):
    """Same OPD transition MSE and unit weights for every saved target, including noop.

    The older diagnostic update_once rejects all-noop batches. Acceptance schedules
    legitimately contain these batches; do not relabel targets or skip their update.
    """
    assert active
    backend.schedule(config.steps)
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for q in active:
        prediction = backend.predict(q["state"], q["index"], q["context"], "student", grad=True)
        loss = transition_mean_loss(prediction, q["target"], q["dt"]) / len(active)
        total += float(loss.detach())
        scaler.scale(loss).backward()
        backend.counters["backward_queries"] += 1
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(backend.parameters(), config.max_grad_norm,
                                         error_if_nonfinite=not scaler.is_enabled())
    previous = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    if scaler.get_scale() < previous:
        raise RuntimeError("GradScaler skipped an update")
    return {"objective": total, "gradient_norm": float(norm), "scale": scaler.get_scale()}


def fit_saved(backend, queries, config, plan, run):
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
                active = [{**queries[e["key"]], "target": queries[e["key"]]["actions"][e["target"]]} for e in schedule[step - 1]]
                values = update_saved_actions(backend, active, optimizer, scaler, config)
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
        atomic_torch_save(snapshot(backend, optimizer, scaler, len(schedule)), out / "training.pt")
        reports[arm] = {"counts": counts, "optimizer_updates": len(schedule), "initial_sha256": frozen,
                        "final_sha256": tensor_digest(backend.student_state()), "elapsed_seconds": time.perf_counter() - started}
        write_json(run / "fit_reports.json", reports)
    return reports


def run_stage(run):
    run = Path(run).resolve()
    verify_run(run)
    assert_environment(read_json(run / "manifest.json")["environment"])
    for p, h in read_json(run / "probe_attestation.json")["inputs"].items():
        assert sha256(p) == h, p
    plan, config, site = read_json(run / "probe_plan.json"), Experiment.load(run / "config.json"), read_json(run / "site.json")
    assert plan["version"] == VERSION and plan["arms"] == ARMS
    gate = read_json(plan["collection_targets"])
    assert gate["training_allowed_by_coverage_gate"]
    assert (plan["schedules"], plan["randomization"]) == schedules(plan["points"], plan["updates"])
    if any((run / n).exists() for n in ("RUNNING", "result.json", "failure.json", "references.jsonl", "updates.jsonl", "endpoints.jsonl")):
        raise FileExistsError("Use a new attempt")
    (run / "RUNNING").write_text(f"pid={os.getpid()}\n")
    started = time.perf_counter()
    try:
        seed_all(config.seed)
        backend = build_backend(config, site, run / "upstream", "cuda", training=plan["stage"] == "fit")
        evidence_plan = read_json(plan["qualification_plan"])
        evidence = read_json(evidence_plan["training_execution"]["qualification_result"])
        assert torch.cuda.get_device_name() == evidence["gpu"] and torch.__version__ == evidence["torch"]
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        backend.load_student(torch.load(plan["student"], map_location="cpu", weights_only=True), refresh_behavior=True)
        records = {r["id"]: r for r in load_records(config.dataset, "train")}
        verifier = build_verifier(config, site, backend.device, list(records.values()))
        if plan["stage"] == "fit":
            queries = load_queries(backend, verifier, config, plan, run)
            with training_execution(evidence_plan) as execution:
                write_json(run / "training_execution.json", {"execution": execution, "qualification_only": True})
                reports = fit_saved(backend, queries, config, plan, run)
            details = {"arms": reports, "training_execution": execution, "optimizer_updates": sum(r["optimizer_updates"] for r in reports.values()), "evaluation_images": 0}
        else:
            assert plan["stage"] == "evaluate"
            load_queries(backend, verifier, config, {**plan, "points": plan["references"]}, run)
            checks = evaluate(backend, verifier, records, config, plan, run)
            assert tensor_digest(backend.behavior_state()) == checks["model_hashes"]["joint128"]
            details = {**checks, "optimizer_updates": 0, "evaluation_images": 224, "fit_run": plan["fit_run"]}
        counts = dict(backend.counters + verifier.counters)
        for k, v in plan["expected_counts"].items():
            assert counts.get(k, 0) == v, (k, counts.get(k), v)
        write_json(run / "result.json", {"status": "complete", "stage": plan["stage"], **details, "counts": counts,
                   "elapsed_seconds": time.perf_counter() - started, "new_teacher_queries": 0,
                   "reference_images": plan["reference_images"], "output_hashes": {
                       str(p.relative_to(run)): sha256(p) for root in (run / "eval_queries", run / "checkpoints")
                       if root.exists() and not root.is_symlink() for p in root.rglob("*.pt")}
                   | {str(p.relative_to(run)): sha256(p) for p in run.glob("*.jsonl")}})
    except Exception as exc:
        write_json(run / "failure.json", {"error": repr(exc), "elapsed_seconds": time.perf_counter() - started})
        raise
    finally:
        (run / "RUNNING").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True)
    run_stage(parser.parse_args().run)
