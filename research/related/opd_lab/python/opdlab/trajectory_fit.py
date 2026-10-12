"""Phase10: sparse versus complete verified-branch fitting, with frozen targets."""
import argparse
import os
from pathlib import Path
import time

import torch

from .backends import build_backend
from .config import Experiment
from .data import load_records
from .engine import atomic_torch_save, preserve_rng, seed_all
from .fitting_probe import cpu_tree, restore, snapshot, tensor_digest, update_once
from .io import append_jsonl, assert_environment, read_json, sha256, write_json
from .learnability import action_metrics, save_verdict, single_endpoint
from .method import transition_mean_loss
from .runtime import verify_run
from .trajectory_probe import rollout
from .verifiers import build_verifier


@torch.no_grad()
def build_case(backend, saved, query, spec, steps):
    """Keep suffix targets on their corrected states; only one action is external."""
    if (saved["record"], saved["seed"], saved["query_index"]) != (query["record"], query["seed"], query["index"]):
        raise ValueError("Saved branch/query identity differs")
    if saved["record"]["id"] != spec["id"]:
        raise ValueError("Wrong selected case")
    sigmas = backend.schedule(steps)
    branch = saved["branches"][spec["candidate"]]
    if len(branch["states"]) != steps + 1 or len(branch["actions"]) != steps:
        raise ValueError("Incomplete trajectory")
    points = []
    for index, (state, target, after) in enumerate(zip(branch["states"], branch["actions"], branch["states"][1:])):
        state, target = state.to(backend.device), target.to(backend.device)
        if not torch.equal(backend.step(state, target, index).cpu(), after):
            raise ValueError("Saved transition does not match the current schedule")
        external = spec["candidate"] != "base" and index == query["index"]
        origin = "external_expert" if external else "current_student_preservation"
        point = {"record": query["record"], "seed": query["seed"], "context": query["context"],
                 "index": index, "state": state, "target": target,
                 "dt": float(sigmas[index + 1] - sigmas[index]), "accepted": external, "origin": origin}
        point["noop"] = backend.predict(state, index, query["context"], "behavior").detach()
        if not external and not torch.equal(point["noop"], target):
            raise ValueError("Preservation target was not evaluated on its saved branch state")
        points.append(point)
    return {"spec": spec, "query": query, "saved": saved, "points": points,
            "sparse": points[query["index"]]}


@torch.no_grad()
def references(backend, verifier, cases, config, run):
    for case in cases:
        spec, query = case["spec"], case["query"]
        states, actions = rollout(backend, query, [], config.steps)
        base = case["saved"]["branches"]["base"]
        if any(not torch.equal(a.cpu(), b) for xs, ys in ((states, base["states"]), (actions, base["actions"])) for a, b in zip(xs, ys)):
            raise RuntimeError("Initial current-student trajectory does not replay phase9")
        images = {"base": backend.decode(states[-1])}
        if spec["candidate"] != "base":
            images["target"] = single_endpoint(backend, case["sparse"], case["sparse"]["target"], config.steps)
        for name, image in images.items():
            verdict = save_verdict(run, backend, verifier, query["record"], image,
                                  ["reference", spec["id"], name], config.utility_joint_weight)
            expected = spec[f"{name}_image_sha256"]
            if verdict["image_sha256"] != expected:
                raise RuntimeError("Phase9 reference image mismatch")
            append_jsonl(run / "references.jsonl", {"id": spec["id"], "mode": name, "exact": True, **verdict})


@torch.no_grad()
def audit(backend, verifier, cases, config, run, arm, step, endpoints):
    state_hash = tensor_digest(backend.student_state())
    for case in cases:
        spec, query = case["spec"], case["query"]
        predictions = []
        for point in case["points"]:
            p = backend.predict(point["state"], point["index"], point["context"], "student")
            predictions.append(p)
            row = {"arm": arm, "optimizer_step": step, "id": spec["id"], "index": point["index"],
                   "origin": point["origin"], "student_sha256": state_hash,
                   "target_mse": float((p.double() - point["target"].double()).square().mean()),
                   "preservation_drift_mse": float((p.double() - point["noop"].double()).square().mean()),
                   "target_loss": float(transition_mean_loss(p, point["target"], point["dt"]))}
            if point["accepted"]:
                row.update(action_metrics(p, point["target"], point["noop"]))
            append_jsonl(run / "metrics.jsonl", row)
        if not endpoints:
            continue
        states, actions = rollout(backend, query, list(range(config.steps)), config.steps)
        base = case["saved"]["branches"]["base"]["states"]
        target = case["saved"]["branches"][spec["candidate"]]["states"]
        drift = {"query_state_mse_vs_phase9": float((states[query["index"]].cpu() - base[query["index"]]).square().mean()),
                 "state_mse_vs_phase9_by_index": [float((s.cpu() - b).square().mean()) for s, b in zip(states, base)],
                 "state_mse_vs_verified_branch_by_index": [float((s.cpu() - b).square().mean()) for s, b in zip(states, target)]}
        atomic_torch_save(cpu_tree({"states": states, "actions": actions}),
                          run / "checkpoints" / arm / f"trajectory-step{step:03d}-case{spec['case']:02d}.pt")
        images = {"full": backend.decode(states[-1]),
                  "local": single_endpoint(backend, case["sparse"], predictions[query["index"]], config.steps)}
        for mode, image in images.items():
            verdict = save_verdict(run, backend, verifier, query["record"], image,
                                  [arm, step, spec["id"], mode], config.utility_joint_weight)
            append_jsonl(run / "endpoints.jsonl", {"arm": arm, "optimizer_step": step, "id": spec["id"],
                         "seed": query["seed"], "query_index": query["index"], "mode": mode,
                         "student_sha256": state_hash, **drift, **verdict})


def fit_arms(backend, verifier, cases, config, plan, run):
    optimizer = torch.optim.AdamW(backend.parameters(), lr=config.learning_rate, betas=tuple(config.adam_betas),
                                 eps=config.adam_epsilon, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.backend == "sd3" and config.precision == "fp16")
    initial = snapshot(backend, optimizer, scaler, 0)
    initial_hash = tensor_digest(initial["student"])
    counts = {}
    for arm, spec in plan["arms"].items():
        target = run / "checkpoints" / arm
        target.mkdir(parents=True, exist_ok=False)
        restore(backend, optimizer, scaler, initial)
        if tensor_digest(backend.student_state()) != initial_hash or optimizer.state:
            raise RuntimeError("Arm initialization or optimizer reset differs")
        active = [c["sparse"] for c in cases] if arm == "sparse" else [p for c in cases for p in c["points"]]
        if len(active) != spec["queries_per_update"]:
            raise ValueError("Wrong supervision density")
        before = backend.counters.copy() + verifier.counters.copy()
        started = time.perf_counter()
        for step in range(spec["updates"] + 1):
            if step:
                # keep=1 gives every preservation point unit weight; accepted marks expert provenance only.
                fitting = update_once(backend, active, optimizer, scaler, config, keep=1)
                append_jsonl(run / "updates.jsonl", {"arm": arm, "optimizer_step": step,
                             "backward_queries": len(active), **fitting})
            if step in spec["audits"]:
                with preserve_rng(), torch.no_grad():
                    if tensor_digest(backend.behavior_state()) != initial_hash:
                        raise RuntimeError("Frozen behavior changed")
                    atomic_torch_save(backend.student_state(), target / f"step{step:03d}.pt")
                    audit(backend, verifier, cases, config, run, arm, step, endpoints=step > 0)
                print({"event": "fit_audit", "arm": arm, "step": step, "elapsed_seconds": time.perf_counter() - started}, flush=True)
            elif step % 16 == 0:
                print({"event": "fit_progress", "arm": arm, "step": step}, flush=True)
        atomic_torch_save(snapshot(backend, optimizer, scaler, spec["updates"]), target / "training.pt")
        backend.export(target / "raw_lora")
        delta = dict(backend.counters + verifier.counters - before)
        if delta["backward_queries"] != spec["updates"] * len(active):
            raise RuntimeError("Incorrect backward budget")
        counts[arm] = {"counts": delta, "wall_seconds": time.perf_counter() - started,
                       "optimizer_updates": spec["updates"], "initial_student_sha256": initial_hash,
                       "optimizer_policy": "reset AdamW and scaler; weight warm start, not full resume"}
        write_json(target / "metadata.json", {**counts[arm], "complete": True,
                   "training_sha256": sha256(target / "training.pt"), "student_sha256": tensor_digest(backend.student_state())})
    return counts


def run_fit(run):
    run = Path(run).resolve()
    verify_run(run)
    assert_environment(read_json(run / "manifest.json")["environment"])
    for path, expected in read_json(run / "probe_attestation.json")["inputs"].items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    if any((run / name).exists() for name in ("RUNNING", "result.json", "failure.json", "updates.jsonl")):
        raise FileExistsError("Use a fresh numbered attempt")
    (run / "RUNNING").write_text(f"pid={os.getpid()}\n")
    started = time.perf_counter()
    try:
        plan, config = read_json(run / "probe_plan.json"), Experiment.load(run / "config.json")
        site = read_json(run / "site.json")
        seed_all(config.seed)
        backend = build_backend(config, site, run / "upstream", "cuda")
        initial = torch.load(plan["student"], map_location="cpu", weights_only=True)
        backend.load_student(initial, refresh_behavior=True)
        verifier = build_verifier(config, site, backend.device, load_records(config.dataset, "train"))
        queries = {q["record"]["id"]: q for q in torch.load(plan["frozen_queries"], map_location="cpu", weights_only=False)}
        cases = [build_case(backend, torch.load(s["branch_file"], map_location="cpu", weights_only=True),
                            queries[s["id"]], s, config.steps) for s in plan["selected"]]
        write_json(run / "supervision.json", [{"id": c["spec"]["id"], "candidate": c["spec"]["candidate"],
                   "index": p["index"], "origin": p["origin"], "dt": p["dt"],
                   "state_sha256": tensor_digest({"state": p["state"]}),
                   "target_sha256": tensor_digest({"target": p["target"]})} for c in cases for p in c["points"]])
        references(backend, verifier, cases, config, run)
        arms = fit_arms(backend, verifier, cases, config, plan, run)
        if tensor_digest(backend.behavior_state()) != tensor_digest(initial):
            raise RuntimeError("Frozen behavior changed")
        elapsed = time.perf_counter() - started
        write_json(run / "result.json", {"status": "complete", "arms": arms, "elapsed_seconds": elapsed,
                   "worker_gpu_hours": elapsed / 3600, "counts": dict(backend.counters + verifier.counters),
                   "initial_student_sha256": tensor_digest(initial), "new_teacher_queries": 0,
                   "source_expert_actions": 1, "optimizer_policy": "common reset, weight warm start",
                   "output_hashes": {p.name: sha256(p) for p in sorted(run.glob("*.jsonl"))},
                   "checkpoint_hashes": {str(p.relative_to(run)): sha256(p) for p in sorted((run / "checkpoints").rglob("*")) if p.is_file()},
                   "supervision_sha256": sha256(run / "supervision.json")})
    except Exception as e:
        write_json(run / "failure.json", {"error": repr(e), "elapsed_seconds": time.perf_counter() - started})
        raise
    finally:
        (run / "RUNNING").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True)
    run_fit(parser.parse_args().run)
