"""Phase17 frozen joint128 candidate coverage across new seeds, times and strengths."""
import argparse
import os
from pathlib import Path
import time

import torch

from .action_transfer import require_exact, trace_baseline
from .backends import build_backend
from .config import Experiment
from .data import load_records
from .engine import atomic_torch_save, seed_all
from .fitting_probe import cpu_tree, tensor_digest
from .io import append_jsonl, assert_environment, digest, read_json, sha256, stable_seed, write_json
from .learnability import save_verdict, single_endpoint
from .runtime import verify_run
from .verifiers import build_verifier

VERSION = "candidate_coverage_v1"
INDICES = (1, 3, 5)
ALPHAS = (.5, 1.)
TEACHERS = ("ocr", "geneval")


def make_pairs(points, excluded):
    assert len(points) == len({p["id"] for p in points}) == 6
    pairs = []
    for p in sorted(points, key=lambda p: p["id"]):
        for replicate in range(2):
            r = {"id": p["id"], "replicate": replicate,
                 "seed": stable_seed(17017, VERSION, "collection", p["id"], replicate)}
            pairs.append({**r, "pair_id": digest(r)[:24]})
    seeds = {p["seed"] for p in pairs}
    if len(seeds) != 12 or seeds & set(excluded):
        raise ValueError("Collection seed collision")
    return pairs


def candidate_specs():
    return [{"mode": f"i{i}_{t}@{a:g}", "index": i, "teacher": t, "alpha": a}
            for i in INDICES for t in TEACHERS for a in ALPHAS]


@torch.no_grad()
def trace(backend, record, seed, steps):
    backend.schedule(steps)
    z, context = backend.initial(seed), backend.encode(record["prompt"])
    states, actions = [z.clone()], []
    for i in range(steps):
        a = backend.predict(z, i, context, "behavior")
        z = backend.step(z, a, i)
        actions.append(a.clone())
        states.append(z.clone())
    return {"record": record, "seed": seed, "context": context, "states": states,
            "actions": actions, "sigmas": backend.sigmas.clone()}, backend.decode(z)


@torch.no_grad()
def propose(backend, trajectory, index):
    noop = trajectory["actions"][index]
    q = {"record": trajectory["record"], "seed": trajectory["seed"], "index": index,
         "state": trajectory["states"][index], "context": trajectory["context"],
         "dt": float(trajectory["sigmas"][index + 1] - trajectory["sigmas"][index]),
         "actions": {"noop": noop}, "experts": {}}
    for teacher in TEACHERS:
        expert = backend.predict(q["state"], index, q["context"], teacher)
        q["experts"][teacher] = expert
        for alpha in ALPHAS:
            # Same FP32 interpolation expression as opdlab.method.collect_query.
            q["actions"][f"i{index}_{teacher}@{alpha:g}"] = noop + alpha * (expert - noop)
    return q


@torch.no_grad()
def execute(backend, verifier, config, plan, run):
    initial = torch.load(plan["student"], map_location="cpu", weights_only=True)
    backend.load_student(initial, refresh_behavior=True)
    initial_hash = tensor_digest(initial)
    records = {r["id"]: r for r in load_records(config.dataset, "train")}
    for p in plan["points"]:
        q, image = trace_baseline(backend, records[p["id"]], p["seed"], config.steps, 3)
        old = torch.load(Path(plan["query_source"]) / p["saved_query"], map_location="cpu", weights_only=False)
        assert torch.equal(q["state"].cpu(), old["state"])
        assert all(torch.equal(a, b) for a, b in zip(q["context"], old["context"]))
        noop = backend.predict(q["state"], 3, q["context"], "behavior")
        assert torch.equal(noop.cpu(), old["actions"]["noop"])
        for name, a in old["actions"].items():
            assert tensor_digest({"action": a}) == p["action_hashes"][name]
        for mode, im in (("noop", image), (p["label_target"], single_endpoint(backend, q, old["actions"][p["label_target"]].to(backend.device), config.steps))):
            v = save_verdict(run, backend, verifier, records[p["id"]], im, [VERSION, "reference", p["id"], mode], config.utility_joint_weight)
            require_exact(v, p["images"][mode]["sha256"])
            append_jsonl(run / "references.jsonl", {"id": p["id"], "seed": p["seed"], "mode": mode, "exact": True, **v})
    write_json(run / "replay_gate.json", {"all_exact": True, "images": 12, "old_queries_exact": 6,
               "counts_before_collection": dict(backend.counters + verifier.counters)})
    print({"event": "references_exact", "images": 12}, flush=True)
    (run / "queries").mkdir(exist_ok=False)
    (run / "trajectories").mkdir(exist_ok=False)
    for pair in plan["pairs"]:
        before = backend.counters + verifier.counters
        trajectory, image = trace(backend, records[pair["id"]], pair["seed"], config.steps)
        saved = Path("trajectories") / f"{pair['pair_id']}.pt"
        atomic_torch_save(cpu_tree(trajectory), run / saved)
        noise = tensor_digest({"noise": trajectory["states"][0]})
        base = save_verdict(run, backend, verifier, records[pair["id"]], image, [VERSION, pair["pair_id"], "noop"], config.utility_joint_weight)
        append_jsonl(run / "endpoints.jsonl", {**pair, "mode": "noop", "index": None, "teacher": None, "alpha": 0, "noise_sha256": noise, **base})
        for index in INDICES:
            q = propose(backend, trajectory, index)
            qpath = Path("queries") / f"{pair['pair_id']}-i{index}.pt"
            atomic_torch_save(cpu_tree(q), run / qpath)
            append_jsonl(run / "queries.jsonl", {**pair, "index": index, "dt": q["dt"], "noise_sha256": noise,
                         "state_sha256": tensor_digest({"state": q["state"]}),
                         "action_hashes": {n: tensor_digest({"action": a}) for n, a in q["actions"].items()},
                         "expert_hashes": {n: tensor_digest({"action": a}) for n, a in q["experts"].items()},
                         "saved_query": str(qpath), "saved_query_sha256": sha256(run / qpath),
                         "trajectory": str(saved), "trajectory_sha256": sha256(run / saved),
                         "prefix_sha256": initial_hash, "suffix_sha256": initial_hash})
            for spec in plan["candidates"]:
                if spec["index"] != index:
                    continue
                action = q["actions"][spec["mode"]]
                endpoint = single_endpoint(backend, q, action, config.steps)
                v = save_verdict(run, backend, verifier, records[pair["id"]], endpoint, [VERSION, pair["pair_id"], spec["mode"]], config.utility_joint_weight)
                append_jsonl(run / "endpoints.jsonl", {**pair, **spec, "noise_sha256": noise,
                             "action_distance": float((action - q["actions"]["noop"]).double().square().mean()), **v})
        append_jsonl(run / "costs.jsonl", {**pair, "counts": dict(backend.counters + verifier.counters - before)})
        print({"event": "state_complete", "pair_id": pair["pair_id"], "images": 13}, flush=True)
    assert tensor_digest(backend.student_state()) == tensor_digest(backend.behavior_state()) == initial_hash
    counts = dict(backend.counters + verifier.counters)
    for k, v in plan["expected_counts"].items():
        assert counts.get(k, 0) == v, (k, counts)
    return {"status": "complete", "counts": counts, "student_sha256": initial_hash, "teacher_queries": 72,
            "new_images": 156, "reference_images": 12, "optimizer_updates": 0,
            "output_hashes": {str(p.relative_to(run)): sha256(p) for p in run.rglob("*") if p.is_file() and
                              (p.parent.name in ("queries", "trajectories") or p.name in
                               ("endpoints.jsonl", "queries.jsonl", "costs.jsonl", "references.jsonl", "replay_gate.json"))}}


def run_probe(run):
    run = Path(run).resolve()
    verify_run(run)
    assert_environment(read_json(run / "manifest.json")["environment"])
    for p, h in read_json(run / "probe_attestation.json")["inputs"].items():
        assert sha256(p) == h, p
    if any((run / n).exists() for n in ("RUNNING", "failure.json", "result.json", "references.jsonl")):
        raise FileExistsError("Use a new attempt")
    (run / "RUNNING").write_text(f"pid={os.getpid()}\n")
    start = time.perf_counter()
    try:
        config, plan, site = Experiment.load(run / "config.json"), read_json(run / "probe_plan.json"), read_json(run / "site.json")
        assert plan["version"] == VERSION and plan["pairs"] == make_pairs(plan["points"], plan["excluded_seeds"])
        assert plan["candidates"] == candidate_specs() and config.steps == 10
        seed_all(config.seed)
        backend = build_backend(config, site, run / "upstream", "cuda", training=True)
        verifier = build_verifier(config, site, backend.device, load_records(config.dataset, "train"))
        result = execute(backend, verifier, config, plan, run)
        write_json(run / "result.json", {**result, "elapsed_seconds": time.perf_counter() - start})
    except Exception as exc:
        write_json(run / "failure.json", {"error": repr(exc), "elapsed_seconds": time.perf_counter() - start})
        raise
    finally:
        (run / "RUNNING").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True)
    run_probe(parser.parse_args().run)
