"""Phase8: isolate prefix and suffix changes of one frozen learned student."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import torch

from .backends import build_backend
from .config import Experiment
from .data import load_records
from .engine import atomic_torch_save, seed_all
from .fitting_probe import cpu_tree, tensor_digest
from .io import append_jsonl, assert_environment, read_json, sha256, write_json
from .learnability import save_verdict
from .runtime import verify_run
from .verifiers import build_verifier

MODES = ("local", "suffix", "prefix", "full", "window2", "window4")


def student_indices(mode, index, steps):
    if not 0 <= index < steps:
        raise ValueError("Query outside the schedule")
    choices = {"base": [], "local": [index], "suffix": list(range(index, steps)),
               "prefix": list(range(index + 1)), "full": list(range(steps)),
               "window2": list(range(index, min(index + 2, steps))),
               "window4": list(range(index, min(index + 4, steps)))}
    return choices[mode]


@torch.no_grad()
def rollout(backend, query, indices, steps):
    if len(indices) != len(set(indices)) or any(not 0 <= i < steps for i in indices):
        raise ValueError("Invalid or duplicate intervention step")
    backend.schedule(steps)
    z = backend.initial(query["seed"])
    states, actions = [z.clone()], []
    for index in range(steps):
        adapter = "student" if index in indices else "behavior"
        p = backend.predict(z, index, query["context"], adapter)
        z = backend.step(z, p, index)
        actions.append(p.clone())
        states.append(z.clone())
    return states, actions


def run_probe(run):
    run = Path(run).resolve()
    verify_run(run)
    assert_environment(read_json(run / "manifest.json")["environment"])
    for path, expected in read_json(run / "probe_attestation.json")["inputs"].items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    plan = read_json(run / "probe_plan.json")
    if plan["modes"] != list(MODES):
        raise ValueError("Unexpected experimental matrix")
    if any((run / n).exists() for n in ("RUNNING", "result.json", "failure.json", "endpoints.jsonl")):
        raise FileExistsError("Use a fresh numbered attempt")
    (run / "RUNNING").write_text(f"pid={os.getpid()}\n")
    started = time.perf_counter()
    try:
        config, site = Experiment.load(run / "config.json"), read_json(run / "site.json")
        seed_all(config.seed)
        device = "cpu" if config.backend == "toy" else "cuda"
        backend = build_backend(config, site, run / "upstream", device, training=False)
        behavior = torch.load(plan["frozen_behavior"], weights_only=True, map_location="cpu")
        student = torch.load(plan["student"], weights_only=True, map_location="cpu")
        backend.load_student(behavior, refresh_behavior=True)
        backend.load_student(student, refresh_behavior=False)
        behavior_hash, student_hash = tensor_digest(behavior), tensor_digest(student)
        all_queries = torch.load(plan["frozen_queries"], weights_only=False, map_location="cpu")
        indexed = {q["record"]["id"]: q for q in all_queries}
        records = load_records(config.dataset, "train")
        verifier = build_verifier(config, site, backend.device, records)
        replay_checks, duplicate_checks = [], []
        for spec in plan["queries"]:
            q = indexed[spec["id"]]
            if (q["seed"], q["index"]) != (spec["seed"], spec["index"]):
                raise ValueError("Frozen query identity changed")
            base_states, base_actions = rollout(backend, q, [], config.steps)
            if not torch.equal(base_states[q["index"]].cpu(), q["state"]):
                raise RuntimeError("Frozen query state failed exact replay")
            images, trajectories = {}, {"base": (base_states, base_actions)}
            for mode in ("base", *MODES):
                indices = student_indices(mode, q["index"], config.steps)
                states, actions = trajectories["base"] if mode == "base" else rollout(backend, q, indices, config.steps)
                trajectories[mode] = cpu_tree({"states": states, "actions": actions})
                image = backend.decode(states[-1])
                verdict = save_verdict(run, backend, verifier, q["record"], image,
                                      ["phase8", q["record"]["id"], mode], config.utility_joint_weight)
                images[mode] = verdict["image_sha256"]
                row = {"id": spec["id"], "mode": mode, "seed": q["seed"], "query_index": q["index"],
                       "student_indices": indices, "sigma_intervals": [[float(backend.sigmas[i]),
                       float(backend.sigmas[i + 1])] for i in indices], "nfe": config.steps,
                       "query_state_mse_vs_frozen": float((states[q["index"]] - base_states[q["index"]]).square().mean()),
                       "state_mse_vs_base_by_index": [float((s - b).square().mean()) for s, b in zip(states, base_states)],
                       "action_mse_vs_base_by_index": [float((a - b).square().mean()) for a, b in zip(actions, base_actions)],
                       **verdict}
                append_jsonl(run / "endpoints.jsonl", row)
                if mode in spec["reference_images"]:
                    expected = spec["reference_images"][mode]
                    check = {"id": spec["id"], "mode": mode, "exact": images[mode] == expected}
                    replay_checks.append(check)
                    if not check["exact"]:
                        raise RuntimeError(f"Phase7 reference image mismatch: {check}")
            if q["index"] == 0:
                for first, second in (("local", "prefix"), ("suffix", "full")):
                    check = {"id": spec["id"], "modes": [first, second], "exact": images[first] == images[second]}
                    duplicate_checks.append(check)
                    if not check["exact"]:
                        raise RuntimeError("Identical execution rules differ")
            atomic_torch_save(trajectories, run / f"trajectory-{spec['case']:02d}.pt")
            print({"event": "query_complete", "id": spec["id"], "modes": len(images)}, flush=True)
        if tensor_digest(backend.behavior_state()) != behavior_hash or tensor_digest(backend.student_state()) != student_hash:
            raise RuntimeError("A frozen adapter changed")
        elapsed = time.perf_counter() - started
        write_json(run / "result.json", {"status": "complete", "elapsed_seconds": elapsed,
                   "worker_gpu_hours": elapsed / 3600 if config.backend == "sd3" else 0,
                   "counts": dict(backend.counters + verifier.counters), "optimizer_updates": 0,
                   "student_sha256": student_hash, "behavior_sha256": behavior_hash,
                   "replay_checks": replay_checks, "duplicate_checks": duplicate_checks,
                   "endpoints_sha256": sha256(run / "endpoints.jsonl"),
                   "trajectory_hashes": {p.name: sha256(p) for p in sorted(run.glob("trajectory-*.pt"))}})
    except Exception as e:
        write_json(run / "failure.json", {"error": repr(e), "elapsed_seconds": time.perf_counter() - started})
        raise
    finally:
        (run / "RUNNING").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True)
    run_probe(parser.parse_args().run)
