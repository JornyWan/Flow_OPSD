"""Phase9: fresh external-teacher actions at a frozen learned policy's states."""
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
from .trajectory_probe import rollout
from .verifiers import build_verifier


@torch.no_grad()
def intervene(backend, states, actions, query, action, steps):
    """Preserve current prefix; execute a new action then the CURRENT student."""
    k = query["index"]
    output_states, output_actions = list(states[:k + 1]), list(actions[:k])
    z = backend.step(states[k].clone(), action, k)
    output_states.append(z)
    output_actions.append(action)
    for j in range(k + 1, steps):
        a = backend.predict(z, j, query["context"], "student")
        z = backend.step(z, a, j)
        output_actions.append(a)
        output_states.append(z)
    return output_states, output_actions


def run_probe(run):
    run = Path(run).resolve()
    verify_run(run)
    assert_environment(read_json(run / "manifest.json")["environment"])
    for path, expected in read_json(run / "probe_attestation.json")["inputs"].items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    if any((run / n).exists() for n in ("RUNNING", "result.json", "failure.json", "candidates.jsonl")):
        raise FileExistsError("Use a fresh numbered attempt")
    plan = read_json(run / "probe_plan.json")
    (run / "RUNNING").write_text(f"pid={os.getpid()}\n")
    started = time.perf_counter()
    try:
        config, site = Experiment.load(run / "config.json"), read_json(run / "site.json")
        seed_all(config.seed)
        backend = build_backend(config, site, run / "upstream", "cuda")
        state = torch.load(plan["student"], weights_only=True, map_location="cpu")
        backend.load_student(state, refresh_behavior=True)
        initial_hash = tensor_digest(state)
        queries = {q["record"]["id"]: q for q in torch.load(plan["frozen_queries"], weights_only=False, map_location="cpu")}
        records = load_records(config.dataset, "train")
        verifier = build_verifier(config, site, backend.device, records)
        replay = []
        with torch.no_grad():
            for spec in plan["queries"]:
                q = queries[spec["id"]]
                if (q["seed"], q["index"]) != (spec["seed"], spec["index"]):
                    raise ValueError("Query identity differs")
                states, actions = rollout(backend, q, list(range(config.steps)), config.steps)
                at_query = states[q["index"]]
                proposals, experts = {"base": actions[q["index"]]}, {}
                for teacher in config.teachers:
                    expert = backend.predict(at_query, q["index"], q["context"], teacher)
                    experts[teacher] = expert
                    for alpha in config.alphas:
                        proposals[f"{teacher}@{alpha:g}"] = actions[q["index"]] + alpha * (expert - actions[q["index"]])
                branches = {}
                for name, action in proposals.items():
                    ss, aa = (states, actions) if name == "base" else intervene(backend, states, actions, q, action, config.steps)
                    verdict = save_verdict(run, backend, verifier, q["record"], backend.decode(ss[-1]),
                                          ["phase9", spec["id"], name], config.utility_joint_weight)
                    append_jsonl(run / "candidates.jsonl", {"id": spec["id"], "candidate": name,
                                 "seed": q["seed"], "query_index": q["index"],
                                 "sigma": float(backend.sigmas[q["index"]]),
                                 "query_state_sha256": tensor_digest({"state": at_query}),
                                 "query_state_mse_vs_old_training_state": float((at_query.cpu() - q["state"]).square().mean()),
                                 **verdict})
                    if name == "base":
                        check = {"id": spec["id"], "exact": verdict["image_sha256"] == spec["baseline_image_sha256"]}
                        replay.append(check)
                        if not check["exact"]:
                            raise RuntimeError("Current student baseline failed exact replay")
                    branches[name] = cpu_tree({"states": ss, "actions": aa})
                atomic_torch_save({"record": q["record"], "seed": q["seed"], "query_index": q["index"],
                                   "experts": cpu_tree(experts), "branches": branches}, run / f"branches-{spec['case']:02d}.pt")
                print({"event": "query_complete", "id": spec["id"], "candidates": len(proposals)}, flush=True)
        if tensor_digest(backend.student_state()) != initial_hash or tensor_digest(backend.behavior_state()) != initial_hash:
            raise RuntimeError("Frozen current policy changed")
        elapsed = time.perf_counter() - started
        write_json(run / "result.json", {"status": "complete", "elapsed_seconds": elapsed,
                   "worker_gpu_hours": elapsed / 3600, "optimizer_updates": 0, "student_sha256": initial_hash,
                   "counts": dict(backend.counters + verifier.counters), "baseline_replay": replay,
                   "candidates_sha256": sha256(run / "candidates.jsonl"),
                   "branches_sha256": {p.name: sha256(p) for p in sorted(run.glob("branches-*.pt"))}})
    except Exception as e:
        write_json(run / "failure.json", {"error": repr(e), "elapsed_seconds": time.perf_counter() - started})
        raise
    finally:
        (run / "RUNNING").unlink()


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", required=True)
    run_probe(p.parse_args().run)
