"""Frozen-query 2x2 mechanism probe; deliberately separate from on-policy training."""
from __future__ import annotations

import argparse
import copy
import hashlib
import os
from pathlib import Path
import time

import torch

from .backends import build_backend
from .config import Experiment
from .data import load_records
from .diagnostic import image_writer
from .engine import atomic_torch_save, distributed_device, preserve_rng, rng_state, restore_rng, seed_all
from .io import append_jsonl, assert_environment, digest, read_json, sha256, write_json
from .method import collect_query, query_log, transition_mean_loss
from .runtime import verify_run
from .verifiers import build_verifier

ARMS = {"fit2": (2, 0), "fit2_keep": (2, 1), "fit4": (4, 0), "fit4_keep": (4, 1)}


def tensor_digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(digest([key, str(value.dtype), list(value.shape)]).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def snapshot(backend, optimizer, scaler, step):
    return {"student": backend.student_state(), "optimizer": cpu_tree(optimizer.state_dict()),
            "scaler": copy.deepcopy(scaler.state_dict()), "rng": rng_state(), "optimizer_step": step}


def restore(backend, optimizer, scaler, state):
    backend.load_student(state["student"], refresh_behavior=False)
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    scaler.load_state_dict(copy.deepcopy(state["scaler"]))
    optimizer.zero_grad(set_to_none=True)
    restore_rng(state["rng"])


def update_once(backend, queries, optimizer, scaler, config, keep):
    if keep not in (0, 1) or not queries or not any(q["accepted"] for q in queries):
        raise ValueError("Probe needs accepted queries and a predeclared keep coefficient")
    backend.schedule(config.steps)
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for q in queries:
        prediction = backend.predict(q["state"], q["index"], q["context"], "student", grad=True)
        weight = 1.0 if q["accepted"] else float(keep)
        loss = transition_mean_loss(prediction, q["target"], q["dt"]) * weight / len(queries)
        total += float(loss.detach())
        scaler.scale(loss).backward()
        backend.counters["backward_queries"] += 1
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(backend.parameters(), config.max_grad_norm,
                                         error_if_nonfinite=not scaler.is_enabled())
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    if scaler.get_scale() < old_scale:
        raise RuntimeError("GradScaler skipped a probe update; fail rather than mislabel fitting count")
    return {"objective": total, "gradient_norm": float(norm), "scale": scaler.get_scale()}


@torch.no_grad()
def fit_audit(backend, queries, config):
    backend.schedule(config.steps)
    rows = []
    for q in queries:
        p = backend.predict(q["state"], q["index"], q["context"], "student")
        rows.append({"id": q["record"]["id"], "accepted": q["accepted"],
                     "target_loss": float(transition_mean_loss(p, q["target"], q["dt"])),
                     "noop_drift": float(transition_mean_loss(p, q["candidates"][0].action, q["dt"]))})
    return rows


@torch.no_grad()
def endpoint_audit(backend, verifier, queries, config, run, arm):
    for q in queries:
        backend.schedule(config.steps)
        p = backend.predict(q["state"], q["index"], q["context"], "student")
        state = backend.step(q["state"].clone(), p, q["index"])
        for j in range(q["index"] + 1, config.steps):
            state = backend.step(state, backend.predict(state, j, q["context"], "behavior"), j)
        save = image_writer(run, digest(["postfit", arm, q["record"]["id"]])[:24])
        row = {"arm": arm, "id": q["record"]["id"], "seed": q["seed"],
               "query_index": q["index"], "accepted": q["accepted"]}
        for name, image in [("single", backend.decode(state)),
                            ("full", backend.generate(q["record"], q["seed"], config.steps))]:
            verdict = verifier(image, q["record"])
            if not verdict.valid:
                raise RuntimeError("Invalid endpoint verifier result")
            path, checksum = save(name, image)
            row[name] = {**verdict.as_dict(config.utility_joint_weight),
                         "image": path, "image_sha256": checksum}
        append_jsonl(Path(run) / "postfit.jsonl", row)


def run_probe(run):
    run = Path(run).resolve()
    config, site = Experiment.load(run / "config.json"), read_json(run / "site.json")
    verify_run(run)
    assert_environment(read_json(run / "manifest.json")["environment"])
    plan = read_json(run / "probe_plan.json")
    attestation = read_json(run / "probe_attestation.json")
    if sha256(run / "probe_plan.json") != attestation["plan_sha256"]:
        raise ValueError("Frozen probe plan changed")
    if sha256(run / "PROTOCOL.md") != attestation["protocol_sha256"]:
        raise ValueError("Frozen protocol changed")
    records = load_records(config.dataset, "train")
    if config.split != "train" or config.method != "verified" or not config.protect_satisfied:
        raise ValueError("Probe requires protected training queries")
    if (len(plan["queries"]) != len(records) or
            {q["id"] for q in plan["queries"]} != {r["id"] for r in records} or
            config.accumulation != len(records) or plan["arms"] != {k: list(v) for k, v in ARMS.items()}):
        raise ValueError("Probe query coverage, accumulation or branch design mismatch")
    if any(not 0 <= q["query_index"] < int(config.steps * config.timestep_fraction) for q in plan["queries"]):
        raise ValueError("Invalid query time")
    _, world, device = distributed_device(config.backend)
    if world != 1:
        raise ValueError("Probe is a single-GPU full-batch experiment")
    lock = run / "RUNNING"
    with lock.open("x") as f:
        f.write(f"pid={os.getpid()}\n")
    started = time.perf_counter()
    try:
        if any((run / name).exists() for name in ("queries.jsonl", "result.json", "metrics.jsonl")):
            raise ValueError("Use a fresh attempt; never append to a partial probe")
        seed_all(config.seed)
        backend = build_backend(config, site, run / "upstream", device)
        verifier = build_verifier(config, site, device, records)
        optimizer = torch.optim.AdamW(backend.parameters(), lr=config.learning_rate,
                    betas=tuple(config.adam_betas), eps=config.adam_epsilon, weight_decay=config.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=config.backend == "sd3" and config.precision == "fp16")
        frozen_behavior = tensor_digest(backend.behavior_state())

        def ensure_frozen():
            if tensor_digest(backend.behavior_state()) != frozen_behavior:
                raise RuntimeError("Behavior parameters changed during a frozen-query experiment")

        def measure(branch, step, fitting=None):
            with preserve_rng():
                ensure_frozen()
                rows = fit_audit(backend, queries, config)
                event = {"branch": branch, "optimizer_step": step, "rows": rows,
                         "student_sha256": tensor_digest(backend.student_state()),
                         "behavior_sha256": frozen_behavior, "fitting": fitting,
                         "counts_cumulative": dict(backend.counters + verifier.counters),
                         "elapsed_seconds": time.perf_counter() - started}
                append_jsonl(run / "metrics.jsonl", event)
                print({k: v for k, v in event.items() if k != "rows"}, flush=True)

        queries, record_map = [], {r["id"]: r for r in records}
        for ordinal, spec in enumerate(plan["queries"]):
            q = collect_query(backend, verifier, record_map[spec["id"]], spec["seed"],
                              spec["query_index"], config, ordinal,
                              image_writer(run, digest(["candidate", spec["id"]])[:24]))
            if any(not c.verdict.valid for c in q["candidates"]):
                raise RuntimeError("Invalid candidate set")
            queries.append(q)
            append_jsonl(run / "queries.jsonl", {"ordinal": ordinal, **query_log(q, config)})
            print({"event": "query", "number": ordinal + 1, "accepted": q["accepted"]}, flush=True)
        atomic_torch_save(cpu_tree([{k: q[k] for k in ("record", "seed", "index", "state", "context", "dt", "target", "accepted")}
                                   for q in queries]), run / "frozen_queries.pt")
        atomic_torch_save(backend.behavior_state(), run / "frozen_behavior.pt")
        measure("initial", 0)
        first_update = update_once(backend, queries, optimizer, scaler, config, keep=0)
        measure("shared", 1, first_update)
        shared = snapshot(backend, optimizer, scaler, 1)
        atomic_torch_save(shared, run / "shared_step1.pt")
        branch_counts = {}
        for keep in (0, 1):
            before = backend.counters.copy() + verifier.counters.copy()
            restore(backend, optimizer, scaler, shared)
            measure(f"keep{keep}_restored", 1)
            for step in (2, 3, 4):
                fitting = update_once(backend, queries, optimizer, scaler, config, keep)
                measure(f"keep{keep}", step, fitting)
                if step in (2, 4):
                    arm = f"fit{step}" + ("_keep" if keep else "")
                    target = run / "checkpoints" / arm
                    target.mkdir(parents=True, exist_ok=False)
                    atomic_torch_save(snapshot(backend, optimizer, scaler, step), target / "training.pt")
                    with preserve_rng():
                        endpoint_audit(backend, verifier, queries, config, run, arm)
                    ensure_frozen()
            branch_counts[f"keep{keep}"] = dict(backend.counters + verifier.counters - before)
        # Export the four predeclared states only after all fitting/endpoint audits.
        for arm, (step, keep) in ARMS.items():
            target = run / "checkpoints" / arm
            state = torch.load(target / "training.pt", weights_only=False, map_location="cpu")
            backend.load_student(state["student"], refresh_behavior=False)
            backend.export(target / "raw_lora")
            ensure_frozen()
            write_json(target / "metadata.json", {"complete": True, "raw_lora": "raw", "optimizer_step": step,
                       "rejected_noop_coefficient": keep, "student_sha256": tensor_digest(state["student"]),
                       "training_state_sha256": sha256(target / "training.pt"),
                       "shared_step1_sha256": sha256(run / "shared_step1.pt"), "probe": True})
        elapsed = time.perf_counter() - started
        write_json(run / "result.json", {"status": "complete", "queries": len(queries),
                   "accepted": sum(q["accepted"] for q in queries), "unique_optimizer_updates": 7,
                   "arms": {k: list(v) for k, v in ARMS.items()}, "behavior_sha256": frozen_behavior,
                   "counts": dict(backend.counters + verifier.counters), "branch_counts": branch_counts,
                   "elapsed_seconds": elapsed, "allocated_gpu_hours": elapsed / 3600 if device.type == "cuda" else 0,
                   "queries_sha256": sha256(run / "queries.jsonl"), "postfit_sha256": sha256(run / "postfit.jsonl"),
                   "metrics_sha256": sha256(run / "metrics.jsonl"), "plan_sha256": attestation["plan_sha256"]})
    finally:
        lock.unlink()


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", required=True)
    run_probe(p.parse_args().run)
