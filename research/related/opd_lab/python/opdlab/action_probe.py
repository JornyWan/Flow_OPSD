"""Counterfactual action projection on frozen phase-3 queries; no optimizer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from .backends import build_backend
from .config import Experiment
from .data import load_records
from .diagnostic import image_writer
from .engine import atomic_torch_save, distributed_device, seed_all
from .fitting_probe import ARMS, tensor_digest
from .io import append_jsonl, assert_environment, digest, read_json, sha256, write_json
from .runtime import verify_run
from .verifiers import build_verifier

BETAS = (0., .25, .5, .75, 1.)


def same_verdict(left, right):
    """Compare exact values across in-memory tuple / JSON list representations."""
    return (all(left[k] == right[k] for k in ("valid", "utility", "joint_success")) and
            all(tuple(left[k]) == tuple(right[k]) for k in ("satisfied", "progress")))


def project_action(base, target, actual):
    delta, movement = target.double() - base.double(), actual.double() - base.double()
    t, d, dot = delta.square().sum(), movement.square().sum(), (delta * movement).sum()
    if not all(torch.isfinite(x) for x in (t, d, dot)) or t <= 0:
        raise ValueError("Projection needs a finite, nonzero target direction")
    gamma = dot / t
    orthogonal = movement - gamma * delta
    projected = (base.double() + gamma * delta).to(base.dtype)
    return projected, {"gamma": float(gamma), "gamma_outside_unit_interval": bool(gamma < 0 or gamma > 1),
                       "target_squared_norm": float(t), "movement_squared_norm": float(d),
                       "orthogonal_squared_norm": float(orthogonal.square().sum()),
                       "cosine": float(dot / (t * d).sqrt()) if d > 0 else None,
                       "orthogonality_error": float((orthogonal * delta).sum().abs() / (t * d).sqrt()) if d > 0 else 0.}


def ray_action(base, target, beta):
    if beta == 0:
        return base
    if beta == 1:
        return target
    return (base.double() + beta * (target.double() - base.double())).to(base.dtype)


@torch.no_grad()
def finish_action(backend, query, action, steps):
    backend.schedule(steps)
    state = backend.step(query["state"].clone(), action, query["index"])
    for j in range(query["index"] + 1, steps):
        state = backend.step(state, backend.predict(state, j, query["context"], "behavior"), j)
    return backend.decode(state)


def verify_plan(run):
    run = Path(run)
    plan = read_json(run / "action_plan.json")
    attestation = read_json(run / "action_attestation.json")
    if sha256(run / "action_plan.json") != attestation["plan_sha256"] or sha256(run / "PROTOCOL.md") != attestation["protocol_sha256"]:
        raise ValueError("Action plan/protocol changed")
    parent = Path(plan["parent_run"])
    for name, expected in plan["parent_files"].items():
        if sha256(parent / name) != expected:
            raise ValueError(f"Parent artifact changed: {name}")
    if plan["betas"] != list(BETAS) or plan["arms"] != list(ARMS):
        raise ValueError("Unexpected action matrix")
    if len(plan["query_ids"]) != len(set(plan["query_ids"])):
        raise ValueError("Duplicated query identity")
    return plan


@torch.no_grad()
def run_probe(run):
    from safetensors.torch import load_file
    run = Path(run).resolve()
    verify_run(run)
    plan = verify_plan(run)
    parent = Path(plan["parent_run"])
    config, site = Experiment.load(run / "config.json"), read_json(run / "site.json")
    assert_environment(read_json(run / "manifest.json")["environment"])
    records = {r["id"]: r for r in load_records(config.dataset, "train")}
    queries = torch.load(parent / "frozen_queries.pt", map_location="cpu", weights_only=False)
    selected = {q["record"]["id"]: q for q in queries if q["accepted"]}
    if set(selected) != set(plan["query_ids"]) or config.split != "train":
        raise ValueError("Action probe must include all original accepted training queries")
    queries = [selected[i] for i in plan["query_ids"]]
    old_queries = {r["id"]: r for r in map(json.loads, (parent / "queries.jsonl").read_text().splitlines())}
    old_post = {(r["arm"], r["id"]): r for r in map(json.loads, (parent / "postfit.jsonl").read_text().splitlines())}
    _, world, device = distributed_device(config.backend)
    if world != 1:
        raise ValueError("Single-device diagnostic only")
    with (run / "RUNNING").open("x") as f:
        f.write(f"pid={os.getpid()}\n")
    started = time.perf_counter()
    try:
        if (run / "actions.jsonl").exists() or (run / "result.json").exists():
            raise ValueError("Use a fresh attempt")
        seed_all(config.seed)
        # Same adapter layout as collection, with original teachers loaded but never queried.
        backend = build_backend(config, site, run / "upstream", device)
        frozen = torch.load(parent / "frozen_behavior.pt", map_location="cpu", weights_only=True)
        backend.load_student(frozen)
        frozen_hash = tensor_digest(frozen)
        verifier = build_verifier(config, site, device, list(records.values()))
        saved_actions, reproductions = {}, []

        def ensure_frozen():
            if tensor_digest(backend.behavior_state()) != frozen_hash:
                raise RuntimeError("Behavior changed")

        def emit(q, name, action, reference=None):
            identity = q["record"]["id"]
            key = (identity, name)
            if key in saved_actions or not torch.isfinite(action).all():
                raise ValueError("Duplicate or nonfinite action")
            image = finish_action(backend, q, action, config.steps)
            verdict = verifier(image, q["record"])
            if not verdict.valid:
                raise RuntimeError("Invalid verifier outcome")
            path, checksum = image_writer(run, digest(["action_probe", identity])[:24])(name, image)
            row = {"id": identity, "seed": q["seed"], "query_index": q["index"], "action": name,
                   **verdict.as_dict(config.utility_joint_weight), "image": path, "image_sha256": checksum}
            if reference:
                if sha256(parent / reference["image"]) != reference["image_sha256"]:
                    raise ValueError("Old reference image changed")
                reproduction = {"id": identity, "action": name,
                                "same_image": checksum == reference["image_sha256"],
                                "same_verdict": same_verdict(row, reference),
                                "parent_image": reference["image"], "parent_sha256": reference["image_sha256"]}
                reproductions.append(reproduction)
                row["reproduction"] = reproduction
            saved_actions[key] = action.detach().cpu().clone()
            append_jsonl(run / "actions.jsonl", row)

        for q in queries:
            identity = q["record"]["id"]
            old = old_queries[identity]
            if q["record"] != records[identity] or q["seed"] != old["seed"] or q["index"] != old["query_index"]:
                raise ValueError("Saved query identity mismatch")
            q["state"], q["target"] = q["state"].to(device), q["target"].to(device)
            schedule = backend.schedule(config.steps)
            if abs(float(schedule[q["index"] + 1] - schedule[q["index"]]) - q["dt"]) > 1e-8:
                raise ValueError("Sigma increment changed")
            q["noop"] = backend.predict(q["state"], q["index"], q["context"], "behavior").detach()
            for beta in BETAS:
                reference = old["candidates"][0] if beta == 0 else (next(c for c in old["candidates"] if c["name"] == old["selected"]) if beta == 1 else None)
                emit(q, f"ray@{beta:g}", ray_action(q["noop"], q["target"], beta), reference)
            print({"event": "ray_query_complete", "id": identity}, flush=True)
        for arm in ARMS:
            state = load_file(str(parent / "checkpoints" / arm / "raw_lora/adapter_model.safetensors"))
            if tensor_digest(state) != read_json(parent / "checkpoints" / arm / "metadata.json")["student_sha256"]:
                raise ValueError("Checkpoint identity mismatch")
            backend.load_student(state, refresh_behavior=False)
            ensure_frozen()
            for q in queries:
                backend.schedule(config.steps)
                actual = backend.predict(q["state"], q["index"], q["context"], "student")
                projected, geometry = project_action(q["noop"], q["target"], actual)
                if geometry["orthogonality_error"] > 1e-10:
                    raise ArithmeticError("Projection did not remove the orthogonal component")
                append_jsonl(run / "geometry.jsonl", {"id": q["record"]["id"], "arm": arm, **geometry})
                emit(q, f"{arm}/actual", actual, old_post[arm, q["record"]["id"]]["single"])
                emit(q, f"{arm}/projected", projected)
            print({"event": "checkpoint_complete", "arm": arm, "images": len(saved_actions)}, flush=True)
        ensure_frozen()
        if len(saved_actions) != 13 * len(queries):
            raise ValueError("Incomplete action matrix")
        atomic_torch_save(saved_actions, run / "actions.pt")
        elapsed = time.perf_counter() - started
        write_json(run / "reproduction.json", reproductions)
        write_json(run / "result.json", {"status": "complete", "queries": len(queries), "images": len(saved_actions),
                   "optimizer_updates": 0, "behavior_sha256": frozen_hash, "elapsed_seconds": elapsed,
                   "allocated_gpu_hours": elapsed / 3600 if device.type == "cuda" else 0,
                   "counts": dict(backend.counters + verifier.counters),
                   "plan_sha256": sha256(run / "action_plan.json"),
                   "artifacts_sha256": {name: sha256(run / name) for name in ("actions.jsonl", "actions.pt", "geometry.jsonl", "reproduction.json")},
                   "reproduction_images": len(reproductions), "same_images": sum(r["same_image"] for r in reproductions),
                   "same_verdicts": sum(r["same_verdict"] for r in reproductions)})
    finally:
        (run / "RUNNING").unlink()


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", required=True)
    run_probe(p.parse_args().run)
